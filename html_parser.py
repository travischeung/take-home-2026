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

# Helper functions for parsing messy web data
def _to_list(val):
    """Normalize value to list: None -> [], str -> [str], iterable -> list, else [val]."""
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    try:
        return list(val)
    except (TypeError, ValueError):
        return [val]

def _norm_float(val):
    """Parse value to float or None."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

# Pick out the high value metadata before the heuristic distillation process.
def extract_metadata(html_path: Path) -> dict:
    """
    Extract high-certainty machine-readable data (JSON-LD, OpenGraph, Twitter, data-*) using BeautifulSoup. 
    Returns a dict of metadata.
    """
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
            # Prevent nested lists if the json-ld data is already a list.
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
    """
    Extract main content as Markdown using Trafilatura (Reader Mode heuristics). 
    Returns markdown-formatted string.
    """
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

# Stage 1: Contextual Anchoring
# Extracts high-fidelity deterministic data (JSON-LD) to anchor the AI hydration stage.
# Adheres to Schema.org standards to ensure cross-merchant compatibility.
def get_hybrid_context(html_path: Path) -> dict:
    """
    Unified entrypoint for stage 1 of pipeline.
    Returns a dict with 'truth_sheet' (anchors for a "soft" pre-hydration) and 'markdown' (context for LLM reasoning)
    """
    raw_meta = extract_metadata(html_path)
    md_content = extract_distilled_content(html_path)

    # Extract the relevant data from the json_ld. eCommerce conventions dictate that the "@type" value will be "Products".
    # NB: Truth sheet will be filled following the conventions outlined on https://schema.org/Product.
    json_ld_list = raw_meta.get("json_ld") or []
    json_ld = {}
    for script in json_ld_list:
        obj_type = script.get("@type", "")
        if obj_type == "Product" or (isinstance(obj_type, list) and "Product" in obj_type):
            json_ld = script
            break

    # Build truth_sheet from json_ld, following Product/ProductVariant schema.
    # Leave values None or empty when not present in json_ld.
    truth_sheet: dict = {
        "name": json_ld.get("name") or None,
        "price": None,
        "description": json_ld.get("description") or None,
        "key_features": [],
        "image_urls": [],
        "video_url": None,
        "category": json_ld.get("category") or None,
        "brand": None,
        "colors": [],
        "variants": [],
    }

    # brand: schema.org uses { "@type": "Brand", "name": "X" } or plain string
    brand_val = json_ld.get("brand")
    if isinstance(brand_val, dict) and brand_val.get("name"):
        truth_sheet["brand"] = str(brand_val["name"]).strip() or None
    elif isinstance(brand_val, str) and brand_val.strip():
        truth_sheet["brand"] = brand_val.strip()

    # price: from offers (single object or array)
    offers = json_ld.get("offers")
    if offers is not None:
        offer_list = offers if isinstance(offers, list) else [offers]
        for offer in offer_list:
            if not isinstance(offer, dict):
                continue
            price_val = _norm_float(offer.get("price"))
            if price_val is not None:
                currency = offer.get("priceCurrency") or "USD"
                truth_sheet["price"] = {
                    "price": price_val,
                    "currency": str(currency).strip() if currency else "USD",
                    "compare_at_price": _norm_float(offer.get("highPrice")),
                }
                break

    # key_features: positiveNotes (list of strings or {name} dicts) or additionalProperty
    notes = json_ld.get("positiveNotes")
    if isinstance(notes, list):
        for x in notes:
            if x is None:
                continue
            s = str(x.get("name", x)).strip() if isinstance(x, dict) else str(x).strip()
            if s:
                truth_sheet["key_features"].append(s)
    add_props = json_ld.get("additionalProperty") or []
    if not truth_sheet["key_features"]:
        add_props = add_props if isinstance(add_props, list) else ([add_props] if isinstance(add_props, dict) else [])
        for p in add_props:
            if isinstance(p, dict):
                val = p.get("value") or p.get("name")
                if val is not None and str(val).strip():
                    truth_sheet["key_features"].append(str(val).strip())

    # image_urls: from JSON-LD (fallback when Verified Media is empty or sparse)
    imgs = json_ld.get("images") or json_ld.get("image")
    for u in _to_list(imgs):
        if isinstance(u, str):
            u = u.strip() if u else None
        elif isinstance(u, dict):
            raw = u.get("url") or u.get("contentUrl")
            u = raw.strip() if isinstance(raw, str) else None
        else:
            u = str(u).strip() if u is not None else None
        if u and u not in truth_sheet["image_urls"]:
            truth_sheet["image_urls"].append(u)
    if not truth_sheet["image_urls"] and json_ld.get("image"):
        u = json_ld["image"]
        u = u.strip() if isinstance(u, str) else None
        if u:
            truth_sheet["image_urls"].append(u)
    # Drop non-product paths (e.g. email signup) so we don't feed bad URLs to the LLM.
    from image_processor import _drop_non_product_urls
    truth_sheet["image_urls"] = _drop_non_product_urls(truth_sheet["image_urls"])
    # When JSON-LD had no product image (or only bad ones), use og:image as fallback (e.g. L.L.Bean).
    if not truth_sheet["image_urls"]:
        og_image = (raw_meta.get("meta") or {}).get("og:image")
        if isinstance(og_image, str) and og_image.strip():
            truth_sheet["image_urls"].append(og_image.strip())

    # video_url: schema.org video can be string, VideoObject {embedUrl, contentUrl}, or array
    vid = json_ld.get("video")
    if isinstance(vid, list) and vid:
        vid = vid[0]
    if isinstance(vid, dict):
        vid = vid.get("embedUrl") or vid.get("contentUrl")
    truth_sheet["video_url"] = vid.strip() if isinstance(vid, str) else None

    # colors
    color_val = json_ld.get("color")
    truth_sheet["colors"] = [str(x).strip() for x in _to_list(color_val) if x is not None and str(x).strip()]

    # variants: hasVariant or similar
    has_variant = json_ld.get("hasVariant", [])
    variant_list = has_variant if isinstance(has_variant, list) else [has_variant]
    for v in variant_list:
        if not isinstance(v, dict):
            continue
        var = {
            "sku": v.get("sku") or None,
            "color": v.get("color") or None,
            "size": v.get("size") or v.get("width") or None,
            "price": _norm_float(v.get("price")),
            "image_url": None,
        }
        v_img = v.get("image") or v.get("image_url")
        if v_img:
            v_url = v_img if isinstance(v_img, str) else (v_img.get("url") or v_img.get("contentUrl") if isinstance(v_img, dict) else None)
            var["image_url"] = v_url.strip() if isinstance(v_url, str) else None
        truth_sheet["variants"].append(var)

    # If no hasVariant but product has sku, treat as single variant
    if not truth_sheet["variants"] and json_ld.get("sku"):
        truth_sheet["variants"].append({
            "sku": json_ld.get("sku"),
            "color": json_ld.get("color") or None,
            "size": None,
            "price": truth_sheet["price"]["price"] if isinstance(truth_sheet.get("price"), dict) else None,
            "image_url": truth_sheet["image_urls"][0] if truth_sheet["image_urls"] else None,
        })
        
    # Final pruning to ensure the Truth Sheet is strictly factual and token-efficient
    truth_sheet = {k: v for k, v in truth_sheet.items() if v not in [None, [], {}]}

    return {
        "truth_sheet": truth_sheet,
        "md_content": md_content
    }
