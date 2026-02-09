import ai
import asyncio
import logging
from pathlib import Path
from html_parser import get_hybrid_context
from image_processor import get_filtered_media
from models import Product

ai_instructions = """
# Role
You are a Senior Data Integrity Agent. Your task is to reconcile raw web extraction data into a single, high-fidelity JSON Product Object.

# Inputs
1. **Truth Sheet (Deterministic)**: Extracted directly from Schema.org JSON-LD. This is the primary source for pricing and identifiers.
2. **Product Context (Markdown)**: Distilled main content of the page. Use this to verify features, materials, and descriptions.
3. **Verified Media**: High-resolution image URLs that have passed quality gates.

# Instructions
- **Reconciliation**: If the Truth Sheet is missing a field (e.g., 'material'), find it in the Markdown.
- **Image Selection**: The 'primary_image' must come from the Verified Media list. 
- **Formatting**: Output ONLY valid JSON. No prose.
- **Constraint**: If a value is not found in either source, return `null`. Do not hallucinate.

# Schema Requirements
{
  "name": "string",
  "brand": "string",
  "price": {"price": number, "currency": "string", "compare_at_price": number | null},
  "description": "string (concise, focus on specs)",
  "features": ["list", "of", "key", "points"],
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

# Response
"""

async def run_pipeline(html_path: str):
    path = Path(html_path)
    context, media = await asyncio.gather(
        get_hybrid_context(path),
        get_filtered_media(path)
    )
    
    truth_sheet = context["truth_sheet"]
    markdown = context["md_content"]
    verified_images = media["images"]


    response = await ai.responses(
        "gpt-5-nano",
        [
            {
                "role": "system",
                "content": ai_instructions
                    .replace("{{truth_sheet}}", str(truth_sheet))
                    .replace("{{markdown}}", markdown)
                    .replace("{{verified_images}}", str(verified_images))
            }
        ],
        text_format=Product
    )
    return response



async def run_all_pipelines(html_paths: list[str]):
    tasks = [run_pipeline(path) for path in html_paths]
    return await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_files = [str(file_path) for file_path in Path("data").glob("*.html")]
    results = asyncio.run(run_all_pipelines(sample_files))
    for path, result in zip(sample_files, results):
        if isinstance(result, BaseException):
            logging.error(f"Failed {path}: {result}")
        else:
            logging.info(f"Result for {path}: {result}")
