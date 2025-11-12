# src/scrapers/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class Scraper(ABC):
    @abstractmethod
    async def collect(self, player:str, opponent:str, date:str) -> Dict[str, Any]:
        """Return raw dicts (not yet normalized) needed for the report."""
