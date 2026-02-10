import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from main import run_all_pipelines
from models import Product

# Browser-like User-Agent to reduce CDN blocking
IMAGE_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run pipeline on startup and write output/products.json
    sample_files = sorted(str(file) for file in Path("data").glob("*.html"))
    if sample_files:
        results = await run_all_pipelines(sample_files)
        for path, r in zip(sample_files, results):
            if isinstance(r, BaseException):
                logging.getLogger("uvicorn.error").warning(f"Pipeline failed for {path}: {type(r).__name__}: {r}")
            elif not isinstance(r, Product):
                logging.getLogger("uvicorn.error").warning(f"Pipeline returned non-Product for {path}: {type(r).__name__}")
        products = [r for r in results if not isinstance(r, BaseException) and isinstance(r, Product)]
        out_path = Path("output/products.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = []
        for i, p in enumerate(products):
            d = p.model_dump()
            d["id"] = i
            payload.append(d)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    yield


app = FastAPI(lifespan=lifespan)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalize_image_url(url: str) -> str:
    """Normalize protocol-relative URLs to https."""
    if url.startswith("//"):
        return "https:" + url
    return url


@app.get("/image")
def proxy_image(url: str = Query(..., description="Image URL to proxy")):
    """Proxy image requests to avoid CORS; forwards upstream status when not 200."""
    normalized = _normalize_image_url(url)
    try:
        r = requests.get(
            normalized,
            timeout=10,
            headers={"User-Agent": IMAGE_UA},
            stream=True,
        )
    except requests.RequestException:
        return Response(status_code=502)
    if r.status_code != 200:
        return Response(status_code=r.status_code)
    content_type = r.headers.get("Content-Type") or "application/octet-stream"
    return Response(content=r.content, media_type=content_type)


@app.get("/products")
def get_products():
    """Return all extracted products"""
    output_file = Path("output/products.json")
    if output_file.exists():
        return json.loads(output_file.read_text())
    return []


@app.get("/products/{product_id}")
def get_product(product_id: int):
    """Return a single product by index"""
    products = get_products()
    if 0 <= product_id < len(products):
        return products[product_id]
    return {"error": "Product not found"}
