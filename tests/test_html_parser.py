"""
Tests for html_parser.extract_metadata and html_parser.extract_distilled_content.

Covers JSON-LD extraction, meta tags (og:*, name), data-* product attributes,
and Trafilatura-based main-content extraction to Markdown.
"""

import json
import tempfile
import unittest
from pathlib import Path

from html_parser import extract_metadata, extract_distilled_content


def _write_html(path: Path, html: str) -> None:
    path.write_text(html, encoding="utf-8", errors="replace")


class TestExtractMetadataStructure(unittest.TestCase):
    """Output structure and edge cases."""

    def test_return_structure(self) -> None:
        """Result has json_ld, meta, and product_attributes keys."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write("<!DOCTYPE html><html><body></body></html>")
            path = Path(f.name)
        try:
            out = extract_metadata(path)
            self.assertIn("json_ld", out)
            self.assertIn("meta", out)
            self.assertIn("product_attributes", out)
            self.assertIsInstance(out["json_ld"], list)
            self.assertIsInstance(out["meta"], dict)
            self.assertIsInstance(out["product_attributes"], dict)
        finally:
            path.unlink(missing_ok=True)

    def test_empty_html_returns_empty_containers(self) -> None:
        """Minimal HTML yields empty list/dicts."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write("<!DOCTYPE html><html><head></head><body></body></html>")
            path = Path(f.name)
        try:
            out = extract_metadata(path)
            self.assertEqual(out["json_ld"], [])
            self.assertEqual(out["meta"], {})
            self.assertEqual(out["product_attributes"], {})
        finally:
            path.unlink(missing_ok=True)

    def test_file_not_found_raises(self) -> None:
        """Missing file raises FileNotFoundError."""
        path = Path("/nonexistent/path/to/file.html")
        with self.assertRaises(FileNotFoundError):
            extract_metadata(path)


class TestJsonLdExtraction(unittest.TestCase):
    """JSON-LD script extraction."""

    def test_json_ld_single_object(self) -> None:
        """Single JSON object in script is appended as one item."""
        ld = {"@context": "https://schema.org", "@type": "Product", "name": "Test"}
        html = f"""<!DOCTYPE html><html><head>
        <script type="application/ld+json">{json.dumps(ld)}</script>
        </head><body></body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_metadata(path)
            self.assertEqual(len(out["json_ld"]), 1)
            self.assertEqual(out["json_ld"][0]["@type"], "Product")
            self.assertEqual(out["json_ld"][0]["name"], "Test")
        finally:
            path.unlink(missing_ok=True)

    def test_json_ld_array_extended(self) -> None:
        """Array in script is extended into json_ld (no nested list)."""
        ld_list = [
            {"@type": "Product", "name": "A"},
            {"@type": "Organization", "name": "B"},
        ]
        html = f"""<!DOCTYPE html><html><head>
        <script type="application/ld+json">{json.dumps(ld_list)}</script>
        </head><body></body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_metadata(path)
            self.assertEqual(len(out["json_ld"]), 2)
            self.assertEqual(out["json_ld"][0]["name"], "A")
            self.assertEqual(out["json_ld"][1]["name"], "B")
        finally:
            path.unlink(missing_ok=True)

    def test_json_ld_invalid_skipped(self) -> None:
        """Invalid JSON in ld+json script is skipped without failing."""
        html = """<!DOCTYPE html><html><head>
        <script type="application/ld+json">{ invalid json }</script>
        <script type="application/ld+json">{"valid": true}</script>
        </head><body></body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_metadata(path)
            self.assertEqual(len(out["json_ld"]), 1)
            self.assertEqual(out["json_ld"][0]["valid"], True)
        finally:
            path.unlink(missing_ok=True)

    def test_json_ld_empty_script_skipped(self) -> None:
        """Empty or whitespace-only script content is skipped."""
        html = """<!DOCTYPE html><html><head>
        <script type="application/ld+json"></script>
        <script type="application/ld+json">   </script>
        </head><body></body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_metadata(path)
            self.assertEqual(out["json_ld"], [])
        finally:
            path.unlink(missing_ok=True)

    def test_other_script_types_ignored(self) -> None:
        """Only application/ld+json scripts are parsed for JSON-LD."""
        html = """<!DOCTYPE html><html><head>
        <script type="text/javascript">{"@type": "Product"}</script>
        </head><body></body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_metadata(path)
            self.assertEqual(out["json_ld"], [])
        finally:
            path.unlink(missing_ok=True)


class TestMetaExtraction(unittest.TestCase):
    """Meta tags (og:*, name, etc.)."""

    def test_meta_og_and_name_extracted(self) -> None:
        """og: and name meta tags are extracted; keys lowercased."""
        html = """<!DOCTYPE html><html><head>
        <meta property="og:title" content="Product Title"/>
        <meta property="og:image" content="https://example.com/img.png"/>
        <meta name="description" content="A product description"/>
        </head><body></body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_metadata(path)
            self.assertEqual(out["meta"]["og:title"], "Product Title")
            self.assertEqual(out["meta"]["og:image"], "https://example.com/img.png")
            self.assertEqual(out["meta"]["description"], "A product description")
        finally:
            path.unlink(missing_ok=True)

    def test_meta_first_wins(self) -> None:
        """First occurrence of a key is kept; duplicates not overwritten."""
        html = """<!DOCTYPE html><html><head>
        <meta property="og:title" content="First"/>
        <meta property="og:title" content="Second"/>
        </head><body></body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_metadata(path)
            self.assertEqual(out["meta"]["og:title"], "First")
        finally:
            path.unlink(missing_ok=True)

    def test_meta_without_content_skipped(self) -> None:
        """Meta tags without content are not added."""
        html = """<!DOCTYPE html><html><head>
        <meta property="og:title"/>
        <meta name="description" content="Has content"/>
        </head><body></body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_metadata(path)
            self.assertNotIn("og:title", out["meta"])
            self.assertEqual(out["meta"]["description"], "Has content")
        finally:
            path.unlink(missing_ok=True)


class TestProductAttributes(unittest.TestCase):
    """data-* attribute harvesting for product-related keys."""

    def test_product_attributes_harvested(self) -> None:
        """data-* containing product, price, sku, id, image, brand are captured."""
        html = """<!DOCTYPE html><html><body>
        <div data-product-id="123" data-price="99.99" data-sku="SKU-X"
             data-image="https://example.com/p.jpg" data-brand="Nike"></div>
        </body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_metadata(path)
            self.assertEqual(out["product_attributes"]["data-product-id"], "123")
            self.assertEqual(out["product_attributes"]["data-price"], "99.99")
            self.assertEqual(out["product_attributes"]["data-sku"], "SKU-X")
            self.assertEqual(out["product_attributes"]["data-image"], "https://example.com/p.jpg")
            self.assertEqual(out["product_attributes"]["data-brand"], "Nike")
        finally:
            path.unlink(missing_ok=True)

    def test_product_attributes_irrelevant_ignored(self) -> None:
        """data-* that don't contain product/price/sku/id/image/brand are ignored."""
        html = """<!DOCTYPE html><html><body>
        <div data-foo="bar" data-analytics-id="a1"></div>
        </body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_metadata(path)
            # analytics-id contains "id" so it is included
            self.assertIn("data-analytics-id", out["product_attributes"])
            self.assertNotIn("data-foo", out["product_attributes"])
        finally:
            path.unlink(missing_ok=True)

    def test_product_attributes_value_coerced_to_str(self) -> None:
        """Attribute values are stored as strings."""
        html = """<!DOCTYPE html><html><body>
        <div data-product-id="456"></div>
        </body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_metadata(path)
            self.assertEqual(out["product_attributes"]["data-product-id"], "456")
            self.assertIsInstance(out["product_attributes"]["data-product-id"], str)
        finally:
            path.unlink(missing_ok=True)


class TestExtractDistilledContent(unittest.TestCase):
    """Trafilatura-based main-content extraction to Markdown."""

    def test_return_type_is_str(self) -> None:
        """Result is always a string."""
        html = """<!DOCTYPE html><html><body><article><p>Hello world.</p></article></body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_distilled_content(path)
            self.assertIsInstance(out, str)
        finally:
            path.unlink(missing_ok=True)

    def test_empty_html_returns_empty_string(self) -> None:
        """Empty HTML body yields empty string."""
        html = """<!DOCTYPE html><html><head></head><body></body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_distilled_content(path)
            self.assertEqual(out, "")
        finally:
            path.unlink(missing_ok=True)

    def test_whitespace_only_returns_empty_string(self) -> None:
        """HTML that is only whitespace yields empty string."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write("   \n\t  ")
            path = Path(f.name)
        try:
            out = extract_distilled_content(path)
            self.assertEqual(out, "")
        finally:
            path.unlink(missing_ok=True)

    def test_file_not_found_raises(self) -> None:
        """Missing file raises FileNotFoundError."""
        path = Path("/nonexistent/path/to/file.html")
        with self.assertRaises(FileNotFoundError):
            extract_distilled_content(path)

    def test_html_with_main_content_returns_markdown(self) -> None:
        """Article-like HTML produces non-empty Markdown with main text."""
        html = """<!DOCTYPE html><html><head><title>Test</title></head><body>
        <nav>Menu</nav>
        <main><article><h1>Product Name</h1><p>This is the main product description.</p></article></main>
        <footer>Footer</footer>
        </body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_distilled_content(path)
            self.assertIsInstance(out, str)
            self.assertGreater(len(out.strip()), 0)
            # Trafilatura often keeps main content; exact format may vary
            self.assertTrue(
                "Product Name" in out or "product description" in out.lower(),
                f"Expected main content in output: {out!r}",
            )
        finally:
            path.unlink(missing_ok=True)

    def test_minimal_html_returns_string(self) -> None:
        """HTML with little or no main content still returns a string (possibly empty)."""
        html = """<!DOCTYPE html><html><head><script>var x=1;</script></head>
        <body><nav><a href="/">Home</a></nav></body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_distilled_content(path)
            self.assertIsInstance(out, str)
            # Trafilatura may return "" or a short extraction; both are valid
        finally:
            path.unlink(missing_ok=True)

    def test_utf8_content_read_correctly(self) -> None:
        """UTF-8 characters in HTML are preserved in extracted Markdown."""
        html = """<!DOCTYPE html><html><body><article><p>Café résumé — 日本語</p></article></body></html>"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = Path(f.name)
        try:
            out = extract_distilled_content(path)
            self.assertIsInstance(out, str)
            # At least one of the UTF-8 snippets should appear
            self.assertTrue(
                "Café" in out or "résumé" in out or "日本語" in out,
                f"Expected UTF-8 content in output: {out!r}",
            )
        finally:
            path.unlink(missing_ok=True)


class TestExtractMetadataIntegration(unittest.TestCase):
    """Run against real data files if present."""

    def test_data_dir_article_structure(self) -> None:
        """Parse data/article.html and assert expected keys and types."""
        data_dir = Path(__file__).resolve().parent.parent / "data"
        article_path = data_dir / "article.html"
        if not article_path.exists():
            self.skipTest("data/article.html not found")
        out = extract_metadata(article_path)
        self.assertIsInstance(out["json_ld"], list)
        self.assertIsInstance(out["meta"], dict)
        self.assertIsInstance(out["product_attributes"], dict)
        # Article.com product pages typically have og/twitter meta
        self.assertGreater(len(out["meta"]), 0)


class TestExtractDistilledContentIntegration(unittest.TestCase):
    """Run extract_distilled_content against real data files if present."""

    def test_data_dir_returns_markdown_string(self) -> None:
        """Parse data/article.html and assert result is non-empty Markdown string."""
        data_dir = Path(__file__).resolve().parent.parent / "data"
        article_path = data_dir / "article.html"
        if not article_path.exists():
            self.skipTest("data/article.html not found")
        out = extract_distilled_content(article_path)
        self.assertIsInstance(out, str)
        self.assertGreater(len(out.strip()), 0)
