from dataclasses import Field
from typing import Any, Optional
from pathlib import Path
from pydantic import BaseModel, field_validator, Field

# Load categories once at module level
CATEGORIES_FILE = Path(__file__).parent / "categories.txt"
VALID_CATEGORIES = set()
if CATEGORIES_FILE.exists():
    with open(CATEGORIES_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                VALID_CATEGORIES.add(line)

class Category(BaseModel):
    # A category from Google's Product Taxonomy when possible
    # https://www.google.com/basepages/producttype/taxonomy.en-US.txt
    name: str

    @field_validator("name")
    @classmethod
    def validate_name_exists(cls, v: str) -> str:
        if v is None or not isinstance(v, str):
            return "Uncategorized"
        v = v.strip()
        if not v:
            return "Uncategorized"
        # Accept any category string; if not in taxonomy, use as-is so pipeline doesn't fail.
        # Avoid substring coercion: it picks arbitrary paths (e.g. "Lighting" -> "Motor Vehicle Lighting", "Pants" -> "Motorcycle Pants").
        return v

class Price(BaseModel):
    price: float
    currency: str
    # If a product is on sale, this is the original price
    compare_at_price: float | None = None

# make sure that this actually works and is the types of variants we are actually looking for lol
class ProductVariant(BaseModel):
    sku: Optional[str] = Field(default=None, description="Unique identifier for this specific version")
    color: Optional[str] = None
    size: Optional[str] = None
    price: Optional[float] = None
    image_url: Optional[str] = None

# This is the final product schema that you need to output. 
# You may add additional models as needed.
class Product(BaseModel):
    name: str
    price: Price
    description: str
    key_features: list[str]
    image_urls: list[str]
    video_url: str | None = None
    category: Category
    brand: str
    colors: list[str]
    variants: list[ProductVariant]

DEFAULT_PRODUCT = Product(
    name="Unknown Product",
    price=Price(price=0.0, currency="USD", compare_at_price=None),
    description="",
    key_features=[],
    image_urls=[],
    video_url=None,
    category=Category(name="Uncategorized"),
    brand="",
    colors=[],
    variants=[],
)
