# src/loaders.py
import asyncio
from scrapers.site_x import SiteXScraper
from normalize import normalize
from models import HitterReportData

async def _gather(player, opponent, date):
    s1 = SiteXScraper()
    raw = await s1.collect(player, opponent, date)
    return normalize(raw, player, opponent, date)

def load_player_data(player, opponent, date) -> HitterReportData:
    return asyncio.run(_gather(player, opponent, date))
