import logging
import json
import re
from openai import OpenAI

logger = logging.getLogger(__name__)

ADVISOR_PROMPT = """You are a senior equity trader. Review this symbol's technical data and news summary.
Decide if it's a "high-conviction" buy, a "standard" buy, or "avoid".
Reply ONLY with JSON: {{"judgment":"buy"|"avoid","conviction":0.0-1.0,"reason":"short explanation"}}

Symbol: {symbol}
Technical Signal: {tech_score:+.2f} ({trend})
News Sentiment: {sentiment:+.2f}
Headlines:
{headlines}
{pdt_section}"""

PDT_SECTION_TIGHT = """
CONSTRAINT: This account is PDT-restricted with {remaining} day-trade(s) remaining this week.
You MUST only recommend stocks with strong overnight or multi-day catalysts.
Prefer: earnings beats, analyst upgrades, sector momentum, clear breakouts with volume.
Avoid: pure intraday momentum plays, high-volatility names likely to reverse,
stocks at resistance without fundamental support.
If you are not confident this stock will hold value overnight, answer "avoid"."""

PDT_SECTION_NORMAL = """
Note: PDT account ({remaining} day-trades remaining). Prefer positions with multi-day
holding potential over pure intraday plays."""


class LLMAdvisor:
    def __init__(self, provider: str = "claude", api_key: str = "", model: str = "mimo-v3-pro"):
        base_url = "https://api.xiaomimimo.com/v1"
        self._client = OpenAI(api_key=api_key or "no-key", base_url=base_url)
        self._model = model

    def advise(
        self,
        symbol: str,
        tech_score: float,
        trend: str,
        sentiment: float,
        headlines: list[str],
        pdt_remaining: int | None = None,
    ) -> dict:
        if not headlines:
            return {"judgment": "avoid", "conviction": 0, "reason": "no data"}

        # Budget exhausted — skip LLM entirely, no buys today
        if pdt_remaining == 0:
            return {
                "judgment": "avoid",
                "conviction": 0.0,
                "reason": "PDT budget exhausted — no buys today",
            }

        pdt_section = self._build_pdt_section(pdt_remaining)
        headlines_str = "\n".join(f"- {h}" for h in headlines[:15])
        prompt = ADVISOR_PROMPT.format(
            symbol=symbol,
            tech_score=tech_score,
            trend=trend,
            sentiment=sentiment,
            headlines=headlines_str,
            pdt_section=pdt_section,
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=100,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            logger.warning(f"LLM Advisor failed for {symbol}: {e}")
        return {"judgment": "avoid", "conviction": 0, "reason": "failed"}

    def _build_pdt_section(self, pdt_remaining: int | None) -> str:
        if pdt_remaining is None or pdt_remaining >= 3:
            return ""
        if pdt_remaining <= 1:
            return PDT_SECTION_TIGHT.format(remaining=pdt_remaining)
        return PDT_SECTION_NORMAL.format(remaining=pdt_remaining)
