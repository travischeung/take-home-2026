"""
Image URL extraction and async filtering for the product extraction pipeline.

Responsibilities:
- Extract candidate image URLs from HTML (img, srcset, meta, JSON-LD).
- Filter to product-quality images: ~1:1 aspect, both sides ≥ MIN_SIDE, valid types.
"""

from __future__ import annotations

import asyncio
import io
import logging
import json
import re
from pathlib import Path
from typing import Any, Optional, Tuple

import aiohttp
from PIL import Image
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


# Product-quality criteria per plan: ~1:1 aspect, both sides ≥ 500px, valid image types.
MIN_SIDE = 500
ASPECT_LOW, ASPECT_HIGH = 0.8, 1.25  # aspect ratio tolerance around 1:1
VALID_IMAGE_TYPES = {"jpeg", "jpg", "png", "webp", "gif"}


# --- Helpers functions --- 
def _normalize_url(url: str, base: Optional[str]) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if base and not url.startswith(("http://", "https://", "//")):
        return urljoin(base, url)
    if url.startswith("//"):
        return "https:" + url
    return url

def _parse_best_from_srcset(srcset_str: str) -> str | None:
    if not srcset_str:
        return None
    candidates = []
    for entry in srcset_str.split(','):
        parts = entry.strip().split()
        if not parts:
            continue
        url = parts[0].strip()
        if not url:
            continue

        score = 0        
        if len(parts) > 1:
            descriptor = parts[1].lower()
            # Extract digits from '1200w' or '2x'
            nums = re.findall(r'\d+', descriptor)
            if nums:
                score = int(nums[0])
        
        candidates.append({"url": url, "score": score})

    if not candidates:
        return None

    # Sort by score descending; among ties, first in srcset is kept.
    best = sorted(candidates, key=lambda x: x['score'], reverse=True)[0]
    return best["url"]

def _dedupe_images(urls: list[str]) -> list[str]:
    """
    Groups images by their base identity to avoid redundant resolutions 
    (e.g., shoe-100x100.jpg and shoe-max.jpg are treated as the same asset).
    """
    if not urls:
        return []
    best_candidates = {}

    for url in urls:
            # Strip query params and resolution-specific suffixes
            base = url.split("?")[0]
            identity = re.sub(
                r'[-_](\d+x\d+|thumb|small|medium|max|large|original)', 
                '', base, flags=re.IGNORECASE
            )
            # Prioritize the version with the longest URL (likely containing higher-res markers)
            if identity not in best_candidates or len(url) > len(best_candidates[identity]):
                best_candidates[identity] = url
    return list[str](best_candidates.values())

# --- Image processing --- 

def extract_image_urls(html_path: Path, base_url: Optional[str] = None) -> list[str]:
    """
    Collect candidate image URLs from HTML for downstream filtering.
    Sources per plan: <img> (src, data-src, srcset, data-srcset), meta (og:image, twitter:image)
    Base URL from <base href> or caller.
    """
    html_content = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html_content, "html.parser")
    base = base_url
    # If no base_url is provided, search for a base_url within the html itself.
    if not base:
        base_tag = soup.find("base", attrs={"href": True})
        if base_tag:
            base = base_tag.get("href")
    seen: set[str] = set[str]()
    urls: list[str] = []
    
    # Helper to dedupe images and ensure urls are properly formatted before being added to list of valid image urls.
    def add(url_candidate: str) -> None:
        url_candidate = _normalize_url(url_candidate, base)
        # Exclude unresolved relative URLs to avoid extraneous entries and optimize token usage.
        if not url_candidate.startswith(("http://", "https://", "//")):
            return
        if url_candidate and url_candidate not in seen and urlparse(url_candidate).path.strip("/"):
            seen.add(url_candidate)
            urls.append(url_candidate)

    # Start with <img src> tags, srcset, data-src, data-srcset.
    for img in soup.find_all("img"):
        for attr in ("src", "data-src", "data-lazy-src", "data-original"):
            img_url = img.get(attr)
            if img_url:
                add(img_url)
        for attr in ("srcset", "data-srcset"):
            img_url = img.get(attr)
            if img_url:
                best_url = _parse_best_from_srcset(img_url)
                if best_url:
                    add(best_url)

    # Next, search for image urls inside of metadata tags. 
    for meta in soup.find_all("meta"):
        key = (meta.get("property") or meta.get("name") or "").strip().lower()
        if key in ("og:image", "og:image:secure_url", "twitter:image"):
            img_url = meta.get("content")
            if img_url:
                add(img_url)

    # Finally, search through structured data (json-ld) for img urls.
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string
        if not raw or not raw.strip():
            continue
        try:
            data = json.loads(raw)
            # json-ld will either be a dict or a list of dicts. If not, continue.
            if isinstance(data, list):
                items = [x for x in data if isinstance(x, dict)]
            elif isinstance(data, dict):
                items = [data]
            else:
                continue

            for item in items:
                for key in ("image", "images"):
                    val = item.get(key)
                    if val is None:
                        continue
                    # If value is a single url string, just add the string.
                    if isinstance(val, str):
                        add(val)
                        continue
                    # If value is ImageObject: e.g. "image": {"@type": "ImageObject", "url": "https://..."}
                    if isinstance(val, dict) and "url" in val:
                        add(val["url"])
                        continue
                    # If value is a list, add each url string in the list.
                    if isinstance(val, list):
                        for v in val:
                            if isinstance(v, str):
                                add(v)
                            elif isinstance(v, dict) and "url" in v:
                                add(v["url"])
        except (json.JSONDecodeError, TypeError):
            continue
    # TODO: add deduping logic for images that may share a base image
    return _dedupe_images(urls)
