import logging
import json
import re
from openai import OpenAI

logger = logging.getLogger(__name__)

ADVISOR_PROMPT = """You are a senior crypto trader. Review this symbol's technical data and news summary.
Decide if it's a "high-conviction" buy, a "standard" buy, or "avoid".
Reply ONLY with JSON: {{"judgment":"buy"|"avoid","conviction":0.0-1.0,"reason":"short explanation"}}

Symbol: {symbol}
Technical Signal: {tech_score:+.2f} ({trend})
News Sentiment: {sentiment:+.2f}
Headlines:
{headlines}"""

class LLMAdvisor:
    def __init__(self, provider: str = "claude", api_key: str = "", model: str = "mimo-v3-pro"):
        # We use Mimo (OpenAI-compatible) even for "claude" if that's the user's config
        base_url = "https://api.xiaomimimo.com/v1"
        self._client = OpenAI(api_key=api_key or "no-key", base_url=base_url)
        self._model = model

    def advise(self, symbol: str, tech_score: float, trend: str, sentiment: float, headlines: list[str]) -> dict:
        if not headlines:
             return {"judgment": "avoid", "conviction": 0, "reason": "no data"}
             
        headlines_str = "\n".join(f"- {h}" for h in headlines[:15])
        prompt = ADVISOR_PROMPT.format(
            symbol=symbol, tech_score=tech_score, trend=trend, 
            sentiment=sentiment, headlines=headlines_str
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
