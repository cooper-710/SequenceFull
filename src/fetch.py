# src/fetch.py
from __future__ import annotations
from typing import Optional

import asyncio
import time
import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

# Suppress urllib3 warnings about OpenSSL
warnings.filterwarnings('ignore', category=UserWarning, module='urllib3')

import httpx

try:
    # only needed if you call browser_get()
    from playwright.async_api import async_playwright
    _HAS_PLAYWRIGHT = True
except Exception:
    _HAS_PLAYWRIGHT = False

DEFAULT_HEADERS = {
    "user-agent": "SequenceBiolab-ReportBot/1.0 (+https://sequencebiolab.example) "
                  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/123 Safari/537.36",
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
}

@dataclass
class RateLimiter:
    """Simple leaky-bucket limiter: ~rps requests/second across awaited tasks."""
    rps: float = 3.0
    _t: float = 0.0

    async def wait(self) -> None:
        period = 1.0 / max(self.rps, 0.0001)
        now = time.time()
        sleep = max(0.0, self._t + period - now)
        if sleep:
            await asyncio.sleep(sleep)
        self._t = time.time()

class FetchError(RuntimeError):
    pass

async def _request(
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 25.0,
    rl: Optional[RateLimiter] = None,
    follow_redirects: bool = True,
) -> httpx.Response:
    if rl:
        await rl.wait()

    h = dict(DEFAULT_HEADERS)
    if headers:
        h.update(headers)

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=follow_redirects, headers=h) as s:
        # manual retry loop with jitter
        for attempt in range(5):
            try:
                resp = await s.request(method.upper(), url, params=params)
                # retry on server throttling / transient errors
                if resp.status_code in (429, 500, 502, 503, 504):
                    # honor Retry-After when present
                    ra = resp.headers.get("retry-after")
                    base = float(ra) if ra and ra.isdigit() else (1.0 + attempt * 1.5)
                    await asyncio.sleep(base + random.random() * 0.4)
                    continue
                resp.raise_for_status()
                return resp
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                if attempt == 4:
                    raise FetchError(f"Network error fetching {url}: {e}") from e
                await asyncio.sleep(0.6 * (attempt + 1) + random.random() * 0.3)
            except httpx.HTTPStatusError as e:
                # non-retryable 4xx
                if 400 <= e.response.status_code < 500 and e.response.status_code not in (429,):
                    raise
                if attempt == 4:
                    raise
                await asyncio.sleep(0.8 * (attempt + 1) + random.random() * 0.3)

async def get_bytes(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    cache: Optional[Any] = None,     # expects SnapshotCache-like with get()/put()
    cache_ttl: Optional[float] = 60 * 60 * 24,  # 24h
    rl: Optional[RateLimiter] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 25.0,
) -> bytes:
    """
    Fetch URL as bytes with optional disk cache & rate limiter.
    """
    if cache:
        cached = cache.get(url, params)
        if cached is not None:
            return cached

    resp = await _request("GET", url, params=params, headers=headers, timeout=timeout, rl=rl)
    content = resp.content
    if cache and content:
        cache.put(url, params, content, ttl=cache_ttl or 0)
    return content

async def get_json(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    cache: Optional[Any] = None,
    cache_ttl: Optional[float] = 60 * 60,
    rl: Optional[RateLimiter] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 25.0,
) -> Any:
    """
    Fetch URL and decode JSON. Caches raw bytes.
    """
    b = await get_bytes(url, params=params, cache=cache, cache_ttl=cache_ttl, rl=rl, headers=headers, timeout=timeout)
    # httpx does .json() but we cache bytes; decode here to avoid re-fetch.
    import json
    return json.loads(b.decode("utf-8", errors="replace")) if b else None

# ---------- Browser fetch (optional; for JS-rendered pages) ----------

async def browser_get(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    cache: Optional[Any] = None,
    cache_ttl: Optional[float] = 60 * 60 * 6,
    wait_selector: Optional[str] = None,
    rl: Optional[RateLimiter] = None,
) -> str:
    """
    Render a page with headless Chromium and return page HTML.
    Only import/playwright if you actually use this.
    """
    if not _HAS_PLAYWRIGHT:
        raise ImportError("Playwright not installed. Run: pip install playwright && playwright install chromium")

    # naive param join
    if params:
        from urllib.parse import urlencode
        url = f"{url}?{urlencode(params)}"

    if cache:
        cached = cache.get(url, params=None)
        if cached is not None:
            return cached.decode("utf-8", errors="replace")

    if rl:
        await rl.wait()

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        if wait_selector:
            try:
                await page.wait_for_selector(wait_selector, timeout=10_000)
            except Exception:
                # continue anyway; caller can validate HTML
                pass
        html = await page.content()
        await browser.close()

    if cache:
        cache.put(url, None, html.encode("utf-8"), ttl=cache_ttl or 0)
    return html

# ---------- Convenience sync wrappers ----------

def get_bytes_sync(*args, **kwargs) -> bytes:
    return asyncio.run(get_bytes(*args, **kwargs))

def get_json_sync(*args, **kwargs) -> Any:
    return asyncio.run(get_json(*args, **kwargs))

def browser_get_sync(*args, **kwargs) -> str:
    return asyncio.run(browser_get(*args, **kwargs))
