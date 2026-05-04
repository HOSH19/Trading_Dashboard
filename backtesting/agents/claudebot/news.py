"""Alpaca News API — fetch historical headlines for a symbol on a given date."""

from __future__ import annotations

import os

import requests


def fetch_headlines(symbol: str, date: str, n: int = 5) -> list[str]:
    """Return up to n headline strings for symbol on date (YYYY-MM-DD). Returns [] on any error."""
    api_key = os.getenv("ALPACA_API_KEY", "")
    api_secret = os.getenv("ALPACA_SECRET_KEY", "")
    base = os.getenv("ALPACA_DATA_ENDPOINT", "https://data.alpaca.markets")
    if not api_key or not api_secret:
        return []
    try:
        resp = requests.get(
            f"{base}/v1beta1/news",
            headers={"Apca-Api-Key-Id": api_key, "Apca-Api-Secret-Key": api_secret},
            params={
                "symbols": symbol,
                "start": f"{date}T00:00:00Z",
                "end": f"{date}T23:59:59Z",
                "limit": n,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return [item["headline"] for item in resp.json().get("news", [])]
    except Exception:
        return []
