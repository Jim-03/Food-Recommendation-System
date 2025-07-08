
import os
import requests
from dotenv import load_dotenv
from typing import Dict, List, Optional

load_dotenv()

class USDAAPIClient:
    def __init__(self):
        self.base_url = "https://api.nal.usda.gov/fdc/v1/"
        self.api_key = os.getenv("USDA_API_KEY", "DEMO_KEY")

    def search_foods(self, query: str, page_size: int = 50) -> Optional[List[Dict]]:
        params = {
            "api_key": self.api_key,
            "query": query,
            "pageSize": page_size,
            "dataType": ["Survey (FNDDS)", "Foundation", "Branded"]
        }

        try:
            response = requests.get(f"{self.base_url}foods/search", params=params)
            response.raise_for_status()
            return response.json().get("foods", [])
        except requests.exceptions.RequestException as e:
            print(f"API ERROR: {e}")
            return None
