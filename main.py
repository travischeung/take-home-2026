import ai
import argparse
import asyncio
import json
import logging
from pathlib import Path
from pydantic import ValidationError

from html_parser import get_hybrid_context
from image_processor import get_filtered_media, normalize_product_image_urls
from models import Product, DEFAULT_PRODUCT

ai_instructions = """
# Role
You are a Senior Data Integrity Agent. Your task is to reconcile raw web extraction data into a single, high-fidelity JSON Product Object.

# Inputs
1. **Truth Sheet (Deterministic)**: Extracted directly from Schema.org JSON-LD. This is the primary source for pricing and identifiers.
2. **Product Context (Markdown)**: Distilled main content of the page. Use this to verify features, materials, and descriptions.
3. **Verified Media**: Image URLs that passed quality gates (dimensions, aspect). Prefer these for image_urls.
4. **Image Candidates**: All page image URLs that passed the non-product path filter (og:image, img tags, etc.). Use this list when Verified Media is empty so you can still pick a product image (e.g. og:image). Prefer product shots; exclude marketing, banner, or email-signup imagery.

# Instructions
- **Reconciliation**: If the Truth Sheet is missing a field (e.g., 'material'), find it in the Markdown.
- **Image Selection**: Populate image_urls from Verified Media when non-empty. When Verified Media is empty, choose from Image Candidates (and/or truth sheet image_urls / variant image_url). Only include product imagery—never marketing, banner, or email-signup. The pipeline will strip non-product URLs. 
- **Formatting**: Output ONLY valid JSON. No prose.
- **Constraint**: If a value is not found in either source, return `null`. Do not hallucinate.

# Schema Requirements
{
  "name": "string",
  "brand": "string",
  "price": {"price": number, "currency": "string", "compare_at_price": number | null},
  "description": "string (concise, focus on specs)",
  "key_features": ["list", "of", "key", "points"],
  "primary_image": "url",
  "gallery": ["url", "url"]
}

# Input Data
<truth_sheet>
{{truth_sheet}}
</truth_sheet>

<product_context>
{{markdown}}
</product_context>

<verified_media>
{{verified_images}}
</verified_media>

<image_candidates>
{{image_candidates}}
</image_candidates>

# Response
"""

async def run_pipeline(html_path: str):
    path = Path(html_path)
    try:
        context, media = await asyncio.gather(
            asyncio.to_thread(get_hybrid_context, path),
            get_filtered_media(path),
        )
    except Exception as e:
        logging.warning("Pipeline context/media failed for %s: %s", html_path, e)
        return DEFAULT_PRODUCT

    truth_sheet = context["truth_sheet"]
    markdown = context["md_content"]
    verified_images = media["images"]
    image_candidates = media.get("candidates", [])

    try:
        response = await ai.responses(
            "gpt-5-nano",
            [
                {
                    "role": "system",
                    "content": ai_instructions
                        .replace("{{truth_sheet}}", str(truth_sheet))
                        .replace("{{markdown}}", markdown)
                        .replace("{{verified_images}}", str(verified_images))
                        .replace("{{image_candidates}}", str(image_candidates))
                }
            ],
            text_format=Product
        )
    except ValidationError as e:
        logging.warning("Schema validation failed for %s: %s", html_path, e)
        return DEFAULT_PRODUCT
    except Exception as e:
        logging.warning("AI request failed for %s: %s", html_path, e)
        return DEFAULT_PRODUCT
    if not response.image_urls and truth_sheet.get("image_urls"):
        response = response.model_copy(update={"image_urls": truth_sheet["image_urls"][:1]})

    response = normalize_product_image_urls(response)
    return response



async def run_all_pipelines(html_paths: list[str]):
    tasks = [run_pipeline(path) for path in html_paths]
    return await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run product extraction pipeline on data/*.html")
    parser.add_argument(
        "--export",
        type=str,
        metavar="PATH",
        help="Write successful products to JSON (e.g. output/products.json) with an 'id' per product",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    sample_files = sorted(str(file_path) for file_path in Path("data").glob("*.html"))
    results = asyncio.run(run_all_pipelines(sample_files))

    for path, result in zip(sample_files, results):
        if isinstance(result, BaseException):
            logging.error(f"Failed {path}: {result}")
        else:
            name = getattr(result, "name", str(result)[:50])
            logging.info(f"Result for {path}: {name}")

    if args.export:
        products = [
            res for res in results
            if not isinstance(res, BaseException) and isinstance(res, Product)
        ]
        out_path = Path(args.export)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = []
        for i, p in enumerate(products):
            d = p.model_dump()
            d["id"] = i
            payload.append(d)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logging.info(f"Exported {len(payload)} products to {out_path}")
