"""
Hybrid Product Distillation Pipeline (Textual Pass)

This module converts messy HTML into high-signal Markdown by combining 
deterministic metadata extraction with heuristic content distillation.

Pipeline flows as follows:
1. Deterministic Extraction (BeautifulSoup): Harvest machine-readable 
   metadata (JSON-LD, OpenGraph) that heuristics may discard.
2. Heuristic Distillation (Trafilatura): Extract the 'core' product story 
   and specs while stripping navigation, ads, and boilerplate.

Output:
A condensed Markdown and metadata context for token-optimized AI hydration.
"""

from __future__ import annotations

import json
from pathlib import Path
from bs4 import BeautifulSoup
import trafilatura

# Pick out the high value metadata before the heuristic distillation process.
def extract_metadata(html_path: Path) -> dict:
    """Extract high-certainty machine-readable data (JSON-LD, OpenGraph, Twitter, data-*) using BeautifulSoup. Returns a dict of metadata."""
    html_content = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html_content, "html.parser")
    output: dict = {
        "json_ld": [],
        "meta": {},
        "product_attributes": {},
    }

    # JSON-LD: highest value machine readable metadata. typically used for SEO for merchant sites.
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string
        if not raw or not raw.strip():
            continue
        try:
            data = json.loads(raw)
            # prevent nested lists if the json_ld data is already a list.
            if isinstance(data, list):
                output["json_ld"].extend(data) 
            else:
                output["json_ld"].append(data)
        except (json.JSONDecodeError, TypeError):
            continue

    # Check for high value tags ie: "og:*", "product:*", "twitter:*", and "name"
    for tag in soup.find_all("meta"):
        if "property" not in tag.attrs and "name" not in tag.attrs:
            continue
        key = (tag.get("property") or tag.get("name") or "").strip().lower()
        content = tag.get("content")
        if key and content is not None and key not in output["meta"]:
            output["meta"][key] = content.strip()
    
    # Check all tags for common data-* attributes, a convention used by ecommerce sites that may contain product data.
    for tag in soup.find_all(True):
        if not any(cur_key.startswith("data-") for cur_key in tag.attrs):
            continue
        for key, val in tag.attrs.items():
            # iterate through the attributes in the tag for relevant content
            if not key.startswith("data-") or val is None:
                continue
            cur_key = key.lower()
            if any(
                x in cur_key
                for x in ("product", "price", "sku", "id", "image", "brand")
            ):
                output["product_attributes"][key] = str(val)
    
    return output

# Extract main content as Markdown using Trafilatura (Reader Mode heuristics).
def extract_distilled_content(html_path: Path) -> str:
    """Extract main content as Markdown using Trafilatura (Reader Mode heuristics). Returns markdown-formatted string."""
    html_content = html_path.read_text(encoding="utf-8", errors="replace")
    if not html_content.strip():
        return ""
    result = trafilatura.extract(
        html_content,
        output_format="markdown",
        include_links=True,
        include_images=True,
        include_tables=True,
        favor_recall=True,
    )
    return result or ""
