
from typing import TypedDict, List

class Nutrient(TypedDict):
    name: str
    value: float

class FoodItem(TypedDict):
    fdc_id: str
    description: str
    brand: str
    category: str
    nutrients: List[Nutrient]
