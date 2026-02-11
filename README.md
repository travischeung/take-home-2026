```markdown
# Travis's Submission for Channel3 Take Home Assignment

Product extraction pipeline: HTML → structured Product JSON.

## Setup

### Backend

The backend uses Python 3.12+ and [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv if needed: https://docs.astral.sh/uv/getting-started/installation/
uv sync
```

Create a `.env` file in the project root with `OPEN_ROUTER_API_KEY` for AI extraction (OpenRouter/LLM calls).

### API

The API is a FastAPI app served with uvicorn. It runs the extraction pipeline on startup and exposes product data.

```bash
uv run uvicorn api:app --reload
```

The API listens on `http://localhost:8000` by default. See `api.py` for endpoints.

### Frontend

The frontend is a React + Vite + TypeScript app.

```bash
cd frontend
npm install
npm run dev
```

The dev server runs at `http://localhost:5173`. The frontend calls the API at `http://localhost:8000`—ensure the API is running for product data to load.

---

## Architecture Overview

### High-Level Flow

**User experience:** Open the app → browse product grid → click a product → 
see full detail page (images, variants, features, pricing, description).

**Extraction:** The pipeline runs once on API startup. It processes each HTML 
file in `/data`, extracts product data through a two-stage approach (structured 
extraction + content distillation), calls the LLM to hydrate the Product schema, 
and writes results to `products.json`. After startup, the API just serves this 
cached JSON—no re-extraction per request.

**Data flow:**
1. Raw HTML → Parse structured data (JSON-LD, meta tags) + distill content (Markdown)
2. Extract images → Filter for quality → Deduplicate
3. LLM receives: structured data + distilled content + verified images
4. LLM outputs: Complete Product JSON matching schema
5. API exposes at `/products` and `/products/{id}`
6. React frontend fetches and renders

**Trigger:** Extraction happens automatically when you start the API server 
(`uvicorn api:app`). Takes ~30 seconds for 6 products. Frontend polls the API 
once loaded—if extraction isn't done yet, you'll see a loading state.


### Extraction Pipeline

The pipeline has two stages:

**Stage 1: Deterministic extraction** (html_parser.py)
- BeautifulSoup pulls structured data: JSON-LD, meta tags, embedded JSON
- Trafilatura distills the main content to clean Markdown (Think Readability or Reader Mode on browsers)
- Why both? BeautifulSoup alone gives you noisy HTML (ads, navigation). 
  Trafilatura alone might discard important structured metadata. The 
  combination preserves high-fidelity data while getting clean content.

**Stage 2: LLM hydration** (ai.py)
- Receives: structured data (truth sheet) + distilled content + verified images
- Outputs: Complete Product JSON matching the Pydantic schema
- Why LLM? Product pages are messy and diverse. A single generic prompt 
  handles a wide variety of unrelated merchants without site-specific logic.

**Image processing** (image_processor.py) runs in parallel:
- Extracts image URLs from HTML
- Async filters for quality (dimensions, aspect ratio, file type)
- Deduplicates based on e-commerce URL patterns
- Only verified images go to the LLM for final selection

### Data Flow

**Where does input data come from?**
- The input data is a collection of raw HTML product pages, stored in the `/data` directory. There are no live API fetches or third-party scraping; extraction runs locally on these files when the API server starts.

**How does data move between backend, API, and frontend?**
- On server startup, the backend parses and processes each HTML file, runs content extraction and LLM-based schema population, and writes the results to a single `products.json` file. This JSON is then served by the API at `/products` (all products) and `/products/{id}` (single product). The frontend fetches product data by calling these API endpoints, and renders product detail and grids from the cached JSON. There is no per-request extraction or recomputation; everything is precomputed and cached until the API is restarted.

**What is the shape of the Product model and where is it defined?**
- The Product model is a strict Pydantic schema (see `models.py`) with fields for name, brand, price (object with currency and compare-at price), description, key features (list), image URLs, video URL, category (object), colors (list), and variants (list of SKU/color/size/price/image). The LLM is prompted to return outputs that match this schema exactly, which is then validated before being written to `products.json`.

### Component Responsibilities

**html_parser.py**  
- Responsible for extracting structured data from each HTML file.  
- Pulls JSON-LD (schema.org), embedded JSON (like `__NEXT_DATA__`), meta tags, and other high-confidence fields.  
- Also distills the main content to Markdown using Trafilatura, striking a balance between structured metadata (which can be incomplete) and readable page content (which can be noisy).  
- Outputs a “truth sheet” with the best-known answers from machine-readable sources, plus the distilled Markdown for context.

**image_processor.py**  
- Takes raw image URLs from HTML and applies filtering logic.  
- Filters out obvious non-product images (tiny dimensions, ultra-wide banners, bad file types, ad beacons).  
- Prefers candidates that look like e-commerce product URLs (by path, dimensions, etc.).  
- Deduplicates by image identity to avoid repeated content and selects for highest quality (bigger, cleaner images, less likely to be banners or collages).

**api.py**  
- On startup, coordinates the extraction pipeline: parses HTML, processes images, runs the LLM, and writes everything out to `products.json`.  
- API endpoints just read this cached JSON (no live recompute).  
- No per-request scraping; data is only updated when you restart the server (re-extracts everything on boot).

## Known Limitations

**Size variant extraction:** While the pipeline successfully extracts color 
variants across all products, size information is inconsistently captured. 
This is due to sites encoding sizes differently—some in JSON-LD variant 
arrays, others in separate JavaScript state, others in dropdown options.

**Impact:** Product schema populates, but size field in variants may be null 
for some products.

**Production approach:** I'd add extraction quality metrics (field population 
rate per site) and iterate on the parsing logic for sites with low size 
extraction rates. Likely need to parse `<select>` elements for size dropdowns 
as a fallback when JSON-LD doesn't have size data.


## System Design:

Context:
The system I've built uses a page agnostic 2 stage pipeline that first extracts machine readable structured data, then extracts human readable high-value text before handing it to a LLM model to handle the reasoning. The page agnostic nature of the design enables this system to be reused across the internet. Emphasis is placed on extracting data that follow ecommerce conventions (ala schema.org), while fallbacks exist in the case of messy websites that may not conform to the expected structures.

Scaling issues:
My pipeline already processes files asynchronously, meaning that HTML parsing, image processing, and LLM schema hydration happen concurrently. With these 5 sample products, this takes ~30 seconds, but it is bottlenecked by the LLM, not IO. Because we can't consistently speed up LLM response times, I will focus on increasing parallel runs. We can increase the number of concurrent calls by having an SQS queue feed URLs in to an ECS worker fleet, which then writes the extracted products to a PostgreSQL database. The pipeline is already stateless, making this kind of horizontal scaling simple. At optimized speed (~6s per product after variant deduplication), one worker processes 14,400 products daily. Scale to 1,000 concurrent workers and you hit 14.4 million products per day, completing a 50M product backfill in roughly 3.5 days.

What will scale from my solution:
- Stateless
- Asynchronous parallel processing of files
- Generic Extraction
- Deterministic Filtering

What won't scale from my solution:
- Single machine constraints
- Large, sometimes unoptimized payloads to the LLM
- Lack of caching
- Variant deduplication

Frontend:
To agentic apps, I'd provide semantic search (/products/search?q="comfortable running shoes under $150"), product comparison (/products/compare returning structured feature diffs), similarity matching (/products/{id}/similar), and category filtering. Developers could benefit from webhook subscriptions (to stay up to date with price changes, stock alerts, etc), extraction SDK (to import competitor data, enrich catalogs, build comparison tools, etc), and structured data export (bulk access to product catalog with embeddings).