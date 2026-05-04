"""Claude API prompt building and score parsing for Claudebot."""

from __future__ import annotations

import json
import re

from backtesting.agents.claudebot.news import fetch_headlines

TRADING_STRATEGY_RUBRIC = """
Score each stock 0-10 using exactly these five factors (0, 1, or 2 points each):

1. Catalyst strength
   0 = no news / vague rumor
   1 = scheduled event (earnings, macro release, product launch scheduled)
   2 = confirmed catalyst (earnings beat, FDA approval, analyst upgrade, major contract)

2. Sector rank YTD
   0 = bottom third of the 20-symbol universe by YTD return
   1 = middle third
   2 = top third

3. Technical setup (distance from 20-day SMA)
   0 = more than 10% above 20-day SMA (extended, risky entry)
   1 = 5-10% above 20-day SMA
   2 = at or below 20-day SMA (ideal entry point)

4. Volume confirmation (today vs 20-day average volume)
   0 = below average volume
   1 = 1.0x - 1.5x average volume
   2 = more than 1.5x average volume

5. Risk/reward ratio
   0 = R:R < 1.5
   1 = R:R 1.5 - 2.0
   2 = R:R > 2.0

Only enter trades with a score >= 7. Return ONLY a valid JSON object mapping ticker to integer score.
"""


def _build_prompt(date: str, candidates: dict[str, dict], headlines: dict[str, list[str]]) -> str:
    lines = [TRADING_STRATEGY_RUBRIC, f"\nDate: {date}\n\nCandidate data:\n"]
    for sym, data in candidates.items():
        news_str = "; ".join(headlines.get(sym, [])[:3]) or "no news found"
        lines.append(
            f"{sym}: price=${data['price']}, SMA20_dist={data['sma20_dist_pct']:+.1f}%, "
            f"5d_momentum={data['momentum_5d_pct']:+.1f}%, vol_ratio={data['volume_ratio']:.2f}x, "
            f"R:R={data['rr_estimate']:.2f}, YTD_rank={data['ytd_rank_pct']:.0f}th_pct | "
            f"News: {news_str}"
        )
    lines.append('\nRespond with only a JSON object like {"AAPL": 7, "MSFT": 4, ...} for all symbols listed.')
    return "\n".join(lines)


def _parse_scores(text: str) -> dict[str, int]:
    match = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        raw = json.loads(match.group())
        return {k: int(v) for k, v in raw.items() if isinstance(v, (int, float))}
    except json.JSONDecodeError:
        return {}


def fetch_scores(date: str, candidates: dict[str, dict]) -> dict[str, int]:
    """Fetch Alpaca headlines for candidates and score them via the Claude API."""
    headlines = {sym: fetch_headlines(sym, date) for sym in candidates}
    from anthropic import Anthropic
    resp = Anthropic().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": _build_prompt(date, candidates, headlines)}],
    )
    return _parse_scores(resp.content[0].text)
