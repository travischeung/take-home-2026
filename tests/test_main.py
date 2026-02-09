"""
Smoke tests for main orchestration: run_pipeline and run_all_pipelines.
Uses real HTML in data/ and mocks only the AI so extraction runs; verifies the pipeline is wired correctly.
"""

import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from models import Category, Price, Product

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VALID_CATEGORY = "Animals & Pet Supplies"


def _make_product(name: str = "Test Product", brand: str = "TestBrand") -> Product:
    return Product(
        name=name,
        price=Price(price=99.99, currency="USD", compare_at_price=None),
        description="A product.",
        key_features=[],
        image_urls=[],
        video_url=None,
        category=Category(name=VALID_CATEGORY),
        brand=brand,
        colors=[],
        variants=[],
    )


class TestMainSmoke(unittest.IsolatedAsyncioTestCase):
    """Happy-path smoke tests: pipeline runs and returns a Product; run_all_pipelines returns list in order."""

    @patch("main.ai.responses", new_callable=AsyncMock)
    async def test_run_pipeline_returns_product(self, mock_ai):
        """Run pipeline on a real HTML file (extraction runs); AI mocked. Asserts we get a Product back."""
        html_path = DATA_DIR / "article.html"
        if not html_path.exists():
            self.skipTest("data/article.html not found")
        mock_ai.return_value = _make_product(name="Article Product")

        from main import run_pipeline

        result = await run_pipeline(str(html_path))

        self.assertIsInstance(result, Product)
        self.assertEqual(result.name, "Article Product")

    @patch("main.ai.responses", new_callable=AsyncMock)
    async def test_run_all_pipelines_returns_results_in_order(self, mock_ai):
        """Run all pipelines on two real HTML files; AI mocked per call. Asserts two Products in same order as paths."""
        paths = [DATA_DIR / "article.html", DATA_DIR / "ace.html"]
        for p in paths:
            if not p.exists():
                self.skipTest(f"{p} not found")
        path_strs = [str(p) for p in paths]
        mock_ai.side_effect = [
            _make_product(name="First"),
            _make_product(name="Second"),
        ]

        from main import run_all_pipelines

        results = await run_all_pipelines(path_strs)

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], Product)
        self.assertIsInstance(results[1], Product)
        self.assertEqual(results[0].name, "First")
        self.assertEqual(results[1].name, "Second")

    async def test_run_all_pipelines_empty_list_returns_empty(self):
        """Empty path list returns empty results."""
        from main import run_all_pipelines

        results = await run_all_pipelines([])
        self.assertEqual(results, [])
