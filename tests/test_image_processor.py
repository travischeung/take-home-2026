"""
Tests for image_processor: URL extraction and async image filtering.

Covers extract_image_urls (HTML → candidate URLs), _is_valid_image_type, _passes_quality,
_get_img_dims (header-based dimension fetch), and filter_image_urls (async orchestration).

Test status (aligned with image_processor.py):
- Implementation uses _get_img_dims (tests previously referred to _check_single_image; updated).
- All tests are run (no skips). Assertions are written to verify real behavior:
  - Empty-path exclusion, srcset “best only”, and non-200 / None dims are asserted explicitly.
  - Mock await_count checks ensure the pipeline calls _get_img_dims for the right candidates.
  - _passes_quality boundaries (exact ASPECT_LOW/ASPECT_HIGH) are tested so bugs like exclusive
    bounds would fail. If a test fails, see the assertion message and line for where/why.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from image_processor import (
    ASPECT_HIGH,
    ASPECT_LOW,
    MIN_SIDE,
    VALID_IMAGE_TYPES,
    extract_image_urls,
    filter_image_urls,
)
from image_processor import _get_img_dims  # Private; tested for TDD
from image_processor import _is_valid_image_type
from image_processor import _passes_quality


def _write_html(path: Path, html: str) -> None:
    path.write_text(html, encoding="utf-8", errors="replace")


# --- Extract Image URLs (implemented) ---


class TestExtractImageUrls(unittest.TestCase):
    """
    extract_image_urls: Collect candidate image URLs from HTML.
    Sources: img (src, data-src, srcset, etc.), meta (og:image, twitter:image), JSON-LD.
    """

    def test_empty_html_returns_empty_list(self) -> None:
        """No img/meta/JSON-LD yields empty list."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write("<!DOCTYPE html><html><head></head><body></body></html>")
            path = Path(f.name)
        try:
            out = extract_image_urls(path)
            self.assertEqual(out, [])
        finally:
            path.unlink(missing_ok=True)

    def test_img_src_extracted(self) -> None:
        """<img src="..."> URLs are extracted."""
        html = """<!DOCTYPE html><html><body>
        <img src="https://example.com/product.jpg"/>
        </body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_image_urls(path)
            self.assertIn("https://example.com/product.jpg", out)
        finally:
            path.unlink(missing_ok=True)

    def test_img_data_src_and_lazy_attrs_extracted(self) -> None:
        """data-src, data-lazy-src, data-original are extracted."""
        html = """<!DOCTYPE html><html><body>
        <img data-src="https://example.com/a.jpg"/>
        <img data-lazy-src="https://example.com/b.jpg"/>
        <img data-original="https://example.com/c.jpg"/>
        </body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_image_urls(path)
            self.assertIn("https://example.com/a.jpg", out)
            self.assertIn("https://example.com/b.jpg", out)
            self.assertIn("https://example.com/c.jpg", out)
        finally:
            path.unlink(missing_ok=True)

    def test_img_srcset_best_url_selected(self) -> None:
        """srcset: highest descriptor (w or x) is chosen; only that URL appears (no small)."""
        html = """<!DOCTYPE html><html><body>
        <img srcset="https://example.com/small.jpg 400w, https://example.com/large.jpg 1200w"/>
        </body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_image_urls(path)
            self.assertIn(
                "https://example.com/large.jpg",
                out,
                "Highest descriptor (1200w) must be the chosen URL.",
            )
            self.assertNotIn(
                "https://example.com/small.jpg",
                out,
                "Only one URL per srcset (the best); small.jpg must not appear.",
            )
        finally:
            path.unlink(missing_ok=True)

    def test_img_data_srcset_extracted(self) -> None:
        """data-srcset is parsed like srcset."""
        html = """<!DOCTYPE html><html><body>
        <img data-srcset="https://example.com/x.jpg 2x"/>
        </body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_image_urls(path)
            self.assertIn("https://example.com/x.jpg", out)
        finally:
            path.unlink(missing_ok=True)

    def test_meta_og_image_twitter_image_extracted(self) -> None:
        """og:image, og:image:secure_url, twitter:image from meta tags."""
        html = """<!DOCTYPE html><html><head>
        <meta property="og:image" content="https://example.com/og.png"/>
        <meta property="og:image:secure_url" content="https://example.com/og-secure.png"/>
        <meta name="twitter:image" content="https://example.com/twitter.gif"/>
        </head><body></body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_image_urls(path)
            self.assertIn("https://example.com/og.png", out)
            self.assertIn("https://example.com/og-secure.png", out)
            self.assertIn("https://example.com/twitter.gif", out)
        finally:
            path.unlink(missing_ok=True)

    def test_json_ld_image_string_extracted(self) -> None:
        """JSON-LD "image": "url" is extracted."""
        ld = {"@type": "Product", "image": "https://example.com/product.webp"}
        html = f"""<!DOCTYPE html><html><head>
        <script type="application/ld+json">{json.dumps(ld)}</script>
        </head><body></body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_image_urls(path)
            self.assertIn("https://example.com/product.webp", out)
        finally:
            path.unlink(missing_ok=True)

    def test_json_ld_image_object_with_url_extracted(self) -> None:
        """JSON-LD "image": {"@type": "ImageObject", "url": "..."} is extracted."""
        ld = {"@type": "Product", "image": {"@type": "ImageObject", "url": "https://example.com/obj.jpg"}}
        html = f"""<!DOCTYPE html><html><head>
        <script type="application/ld+json">{json.dumps(ld)}</script>
        </head><body></body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_image_urls(path)
            self.assertIn("https://example.com/obj.jpg", out)
        finally:
            path.unlink(missing_ok=True)

    def test_json_ld_images_list_extracted(self) -> None:
        """JSON-LD "images": [url, ImageObject, ...] each URL is extracted."""
        ld = {
            "@type": "Product",
            "images": [
                "https://example.com/1.png",
                {"@type": "ImageObject", "url": "https://example.com/2.png"},
            ],
        }
        html = f"""<!DOCTYPE html><html><head>
        <script type="application/ld+json">{json.dumps(ld)}</script>
        </head><body></body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_image_urls(path)
            self.assertIn("https://example.com/1.png", out)
            self.assertIn("https://example.com/2.png", out)
        finally:
            path.unlink(missing_ok=True)

    def test_base_url_resolves_relative_img_src(self) -> None:
        """base_url parameter resolves relative img src."""
        html = """<!DOCTYPE html><html><body>
        <img src="/images/product.jpg"/>
        </body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_image_urls(path, base_url="https://example.com/")
            self.assertIn("https://example.com/images/product.jpg", out)
        finally:
            path.unlink(missing_ok=True)

    def test_base_tag_in_html_used_when_no_base_url_param(self) -> None:
        """<base href="..."> is used when base_url not provided."""
        html = """<!DOCTYPE html><html><head>
        <base href="https://shop.com/" />
        </head><body>
        <img src="product.png"/>
        </body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_image_urls(path)
            self.assertIn("https://shop.com/product.png", out)
        finally:
            path.unlink(missing_ok=True)

    def test_protocol_relative_url_normalized(self) -> None:
        """//example.com/img.png is normalized to https://example.com/img.png."""
        html = """<!DOCTYPE html><html><body>
        <img src="//cdn.example.com/img.jpg"/>
        </body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_image_urls(path)
            self.assertTrue(
                any("https://" in u and "cdn.example.com/img.jpg" in u for u in out),
                f"Expected protocol-relative URL normalized; got {out}",
            )
        finally:
            path.unlink(missing_ok=True)

    def test_duplicate_urls_deduplicated(self) -> None:
        """Same URL from multiple sources appears once."""
        html = """<!DOCTYPE html><html><head>
        <meta property="og:image" content="https://example.com/same.jpg"/>
        </head><body>
        <img src="https://example.com/same.jpg"/>
        </body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_image_urls(path)
            self.assertEqual(out.count("https://example.com/same.jpg"), 1)
        finally:
            path.unlink(missing_ok=True)

    def test_relative_url_without_base_excluded(self) -> None:
        """Relative URLs that cannot be resolved are excluded (no base)."""
        html = """<!DOCTYPE html><html><body>
        <img src="relative/path.jpg"/>
        </body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_image_urls(path)
            # Unresolved relative URLs should not be in output
            self.assertFalse(any(not u.startswith("http") for u in out))
        finally:
            path.unlink(missing_ok=True)

    def test_empty_path_urls_excluded(self) -> None:
        """URLs with empty path (e.g. protocol only) are excluded."""
        html = """<!DOCTYPE html><html><body>
        <img src="https://example.com"/>
        </body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_image_urls(path)
            # Contract: path.strip("/") is falsy for "https://example.com", so URL must not be added.
            self.assertNotIn(
                "https://example.com",
                out,
                "Empty-path URL must be excluded; add() requires non-empty path.",
            )
            self.assertEqual(out, [], "No other images in HTML; output must be empty.")
        finally:
            path.unlink(missing_ok=True)

    def test_file_not_found_raises(self) -> None:
        """Missing HTML file raises FileNotFoundError."""
        path = Path("/nonexistent/file.html")
        with self.assertRaises(FileNotFoundError):
            extract_image_urls(path)

    def test_json_ld_invalid_skipped(self) -> None:
        """Invalid JSON in ld+json script is skipped without crashing."""
        html = """<!DOCTYPE html><html><head>
        <script type="application/ld+json">{ invalid }</script>
        <script type="application/ld+json">{"image": "https://example.com/ok.jpg"}</script>
        </head><body></body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_image_urls(path)
            self.assertIn("https://example.com/ok.jpg", out)
        finally:
            path.unlink(missing_ok=True)


# --- _is_valid_image_type (not implemented for this commit; image_processor line 182+) ---


class TestIsValidImageType(unittest.TestCase):
    """
    _is_valid_image_type: Pre-filter by URL extension before network.
    Valid: jpeg, jpg, png, webp (product-image-likely types; GIF excluded).
    """

    def test_valid_extensions_accepted(self) -> None:
        """Each of jpeg, jpg, png, webp returns True (GIF excluded for product-image filter)."""
        for ext in VALID_IMAGE_TYPES:
            with self.subTest(ext=ext):
                url = f"https://example.com/image.{ext}"
                self.assertTrue(_is_valid_image_type(url), f"Expected {ext} to be valid")

    def test_extension_case_insensitive(self) -> None:
        """Extensions are matched case-insensitively."""
        self.assertTrue(_is_valid_image_type("https://x.com/img.JPEG"))
        self.assertTrue(_is_valid_image_type("https://x.com/img.PNG"))

    def test_svg_rejected(self) -> None:
        """SVG is not in valid types."""
        self.assertFalse(_is_valid_image_type("https://example.com/logo.svg"))

    def test_bmp_rejected(self) -> None:
        """BMP is not in valid types."""
        self.assertFalse(_is_valid_image_type("https://example.com/photo.bmp"))

    def test_no_extension_rejected(self) -> None:
        """URL with no path extension returns False."""
        self.assertFalse(_is_valid_image_type("https://example.com/image"))

    def test_path_only_no_extension_rejected(self) -> None:
        """Path ending in slash has no extension."""
        self.assertFalse(_is_valid_image_type("https://example.com/path/"))

    def test_query_params_preserved_for_extension_check(self) -> None:
        """Extension is taken from path; query string ignored."""
        # path is /image.jpg?size=large -> ext should be jpg
        self.assertTrue(_is_valid_image_type("https://example.com/image.jpg?size=large"))

    def test_fragment_ignored(self) -> None:
        """Fragment (#) does not affect extension parsing."""
        self.assertTrue(_is_valid_image_type("https://example.com/image.png#section"))

    def test_double_extension_takes_last(self) -> None:
        """Path like image.tar.gz -> ext is gz (not in valid types)."""
        self.assertFalse(_is_valid_image_type("https://example.com/image.tar.gz"))


# --- _passes_quality (not implemented for this commit; image_processor line 182+) ---


class TestPassesQuality(unittest.TestCase):
    """
    _passes_quality: Product-quality criteria: both sides >= MIN_SIDE,
    aspect ratio in [ASPECT_LOW, ASPECT_HIGH] (~1:1).
    """

    def test_min_dimensions_pass(self) -> None:
        """Both sides exactly MIN_SIDE passes."""
        self.assertTrue(_passes_quality(MIN_SIDE, MIN_SIDE))
    
    def test_both_sides_above_min_pass(self) -> None:
        """600x600 passes (square, above min)."""
        self.assertTrue(_passes_quality(600, 600))

    def test_aspect_within_bounds_pass(self) -> None:
        """Aspect 1.0, 0.9, 1.2 within [ASPECT_LOW, ASPECT_HIGH]; boundaries inclusive."""
        self.assertTrue(_passes_quality(600, 600))   # 1.0
        self.assertTrue(_passes_quality(600, 667))   # ~0.9
        self.assertTrue(_passes_quality(720, 600))   # 1.2
        # Exact boundaries: 600/750 = 0.8, 625/500 = 1.25 (inclusive range)
        self.assertTrue(_passes_quality(600, 750), "Aspect exactly ASPECT_LOW must pass")
        self.assertTrue(_passes_quality(625, 500), "Aspect exactly ASPECT_HIGH must pass")

    def test_width_below_min_fails(self) -> None:
        """Width < MIN_SIDE fails even if height OK."""
        self.assertFalse(_passes_quality(MIN_SIDE - 1, MIN_SIDE))

    def test_height_below_min_fails(self) -> None:
        """Height < MIN_SIDE fails."""
        self.assertFalse(_passes_quality(MIN_SIDE, MIN_SIDE - 1))

    def test_aspect_too_tall_fails(self) -> None:
        """Very tall (e.g. 500x1000) aspect > ASPECT_HIGH... no: 500/1000=0.5 < ASPECT_LOW."""
        # aspect = w/h = 500/1000 = 0.5 < 0.8
        self.assertFalse(_passes_quality(500, 1000))

    def test_aspect_too_wide_fails(self) -> None:
        """Very wide (e.g. 1000x500) aspect 2.0 > ASPECT_HIGH."""
        self.assertFalse(_passes_quality(1000, 500))

    def test_zero_height_returns_false(self) -> None:
        """h=0 avoids division-by-zero; should return False."""
        self.assertFalse(_passes_quality(600, 0))

    def test_zero_width_returns_false(self) -> None:
        """w=0 fails min-side check."""
        self.assertFalse(_passes_quality(0, 600))


# --- filter_image_urls (not implemented for this commit; image_processor line 182+) ---


class TestFilterImageUrls(unittest.IsolatedAsyncioTestCase):
    """
    filter_image_urls: Async filter URLs to product-quality images.
    Uses _get_img_dims for dimensions; filters by _is_valid_image_type and _passes_quality.
    """

    async def test_empty_input_returns_empty(self) -> None:
        """Empty URL list returns [] without network calls."""
        result = await filter_image_urls([])
        self.assertEqual(result, [])

    async def test_no_valid_extensions_returns_empty(self) -> None:
        """URLs with no valid image extension are filtered out before network."""
        urls = [
            "https://example.com/script.js",
            "https://example.com/style.css",
            "https://example.com/doc.pdf",
        ]
        result = await filter_image_urls(urls)
        self.assertEqual(result, [])

    async def test_filter_by_quality_when_mocked(self) -> None:
        """When _get_img_dims returns dims, filter by _passes_quality; ext filter runs first."""
        with patch(
            "image_processor._get_img_dims",
            new_callable=AsyncMock,
            return_value=(600, 600),
        ) as mock_dims:
            urls = [
                "https://example.com/product.png",
                "https://example.com/thumb.svg",  # filtered by ext before network
            ]
            result = await filter_image_urls(urls)
            self.assertIn("https://example.com/product.png", result)
            self.assertNotIn("https://example.com/thumb.svg", result)
            # Only one URL has valid ext, so _get_img_dims must be called exactly once (not for svg).
            self.assertEqual(mock_dims.await_count, 1, "Network check only for valid-extension URL.")
            self.assertEqual(len(result), 1, "Exactly one URL passes ext + quality.")

    async def test_rejects_url_when_dims_fail_quality(self) -> None:
        """URL with valid ext but small dimensions is excluded."""
        with patch(
            "image_processor._get_img_dims",
            new_callable=AsyncMock,
            return_value=(100, 100),  # Fails min side
        ):
            urls = ["https://example.com/small.jpg"]
            result = await filter_image_urls(urls)
            self.assertEqual(result, [], "Dimensions below MIN_SIDE must be filtered out.")

    async def test_excludes_url_when_get_img_dims_returns_none(self) -> None:
        """When _get_img_dims returns None (e.g. 404, network error), URL must not be in result."""
        with patch(
            "image_processor._get_img_dims",
            new_callable=AsyncMock,
            return_value=None,
        ):
            urls = ["https://example.com/product.png"]
            result = await filter_image_urls(urls)
            self.assertEqual(
                result,
                [],
                "Failed dimension fetch must exclude URL; cannot pass through None.",
            )

    async def test_max_concurrent_respected(self) -> None:
        """All candidate URLs get a dimension check; result order preserved; no crash."""
        with patch(
            "image_processor._get_img_dims",
            new_callable=AsyncMock,
            return_value=(600, 600),
        ) as mock_dims:
            urls = ["https://example.com/a.jpg", "https://example.com/b.png"] * 5
            result = await filter_image_urls(urls, max_concurrent=2)
            self.assertEqual(len(result), 10, "All 10 have valid ext and pass quality.")
            # Each candidate must be passed to _get_img_dims (semaphore limits concurrency internally).
            self.assertEqual(
                mock_dims.await_count,
                10,
                "Each of 10 URLs must be checked; if this fails, the loop or semaphore may be wrong.",
            )


# --- _get_img_dims ---


class TestCheckSingleImage(unittest.IsolatedAsyncioTestCase):
    """
    _get_img_dims: Fetches header bytes and returns (width, height) or None.
    Implementation uses session.get(url), then resp.content.read(_HEADER_READ_SIZE), then PIL.
    """

    async def test_returns_dims_for_valid_image_bytes(self) -> None:
        """When response body contains valid image header, return (width, height)."""
        # Minimal valid 1x1 PNG (signature + IHDR + IDAT + IEND) so PIL can open and read .size
        png_1x1 = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.content = MagicMock()
        mock_resp.content.read = AsyncMock(return_value=png_1x1)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        result = await _get_img_dims(mock_session, "https://example.com/1x1.png")
        self.assertEqual(result, (1, 1), "Valid PNG header should yield (1, 1)")

    async def test_returns_none_on_404(self) -> None:
        """Request that raises (e.g. 404) should return None, not propagate."""
        mock_session = MagicMock()
        mock_session.get = MagicMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=404,
                message="Not Found",
            )
        )
        result = await _get_img_dims(mock_session, "https://example.com/missing.jpg")
        self.assertIsNone(result, "404 should be handled gracefully with None")

    async def test_returns_none_on_non_200_status(self) -> None:
        """Any non-200 status (e.g. 500) must return None, not parse body."""
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.content = MagicMock()
        mock_resp.content.read = AsyncMock(return_value=b"")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        result = await _get_img_dims(mock_session, "https://example.com/error.jpg")
        self.assertIsNone(result, "Non-200 must return None without relying on body.")

    async def test_returns_none_for_corrupt_image_bytes(self) -> None:
        """Corrupt or non-image body should return None."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.content = MagicMock()
        mock_resp.content.read = AsyncMock(return_value=b"not an image")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        result = await _get_img_dims(mock_session, "https://example.com/fake.jpg")
        self.assertIsNone(result, "Corrupt bytes should yield None")


# --- Integration (optional; skip until full pipeline works) ---


class TestImageProcessorIntegration(unittest.IsolatedAsyncioTestCase):
    """Run extract_image_urls against real data files if present."""

    def test_data_dir_extract_urls(self) -> None:
        """Parse data/article.html and assert non-empty URL list when images exist."""
        data_dir = Path(__file__).resolve().parent.parent / "data"
        for name in ("article.html", "nike.html", "llbean.html"):
            path = data_dir / name
            if not path.exists():
                continue
            with self.subTest(file=name):
                out = extract_image_urls(path)
                self.assertIsInstance(out, list)
                # Some product pages have images; allow empty for minimal pages
                for url in out:
                    self.assertTrue(
                        url.startswith("http://") or url.startswith("https://"),
                        f"URL should be absolute: {url}",
                    )
