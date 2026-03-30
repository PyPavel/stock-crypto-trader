import hashlib
import json
import logging
import re
import time
from openai import OpenAI

logger = logging.getLogger(__name__)

MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"

# Compact prompt — fewer tokens, same accuracy
PROMPT_TEMPLATE = """Rate the market sentiment of these headlines.
Reply ONLY with JSON: {{"sentiment":"bullish"|"bearish"|"neutral","confidence":0.0-1.0}}

{texts}"""

SENTIMENT_SCORES = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
CHUNK_SIZE = 10        # headlines per LLM call (was 5 — halves API calls)
MAX_HEADLINE_LEN = 120 # truncate long headlines to save input tokens
CACHE_TTL = 600        # 10 minutes — news sentiment changes slowly


class SentimentAnalyzer:
    def __init__(self, model: str = "mimo-v2-flash", api_key: str = ""):
        self._model = model
        self._client = OpenAI(api_key=api_key or "no-key", base_url=MIMO_BASE_URL)
        # cache: text_hash → (score, timestamp)
        self._cache: dict[str, tuple[float, float]] = {}

    def score_texts(self, texts: list[str]) -> float:
        if not texts:
            return 0.0

        # Deduplicate and truncate before sending to LLM
        seen: set[str] = set()
        clean: list[str] = []
        for t in texts:
            t = t[:MAX_HEADLINE_LEN]
            if t not in seen:
                seen.add(t)
                clean.append(t)

        # Check cache using a hash of the deduplicated text set
        cache_key = hashlib.md5("|".join(sorted(clean)).encode()).hexdigest()
        cached_score, cached_ts = self._cache.get(cache_key, (None, 0))
        if cached_score is not None and (time.time() - cached_ts) < CACHE_TTL:
            logger.debug(f"Sentiment cache hit ({len(clean)} texts)")
            return cached_score

        chunk_scores = []
        for i in range(0, len(clean), CHUNK_SIZE):
            chunk = clean[i:i + CHUNK_SIZE]
            score = self._score_chunk(chunk)
            if score is not None:
                chunk_scores.append(score)

        if not chunk_scores:
            return cached_score if cached_score is not None else 0.0

        result = sum(chunk_scores) / len(chunk_scores)
        self._cache[cache_key] = (result, time.time())
        return result

    def _score_chunk(self, texts: list[str]) -> float | None:
        prompt = PROMPT_TEMPLATE.format(texts="\n".join(f"- {t}" for t in texts))
        content = ""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=32,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content.strip()

            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                content = match.group(0)

            data = json.loads(content)
            if isinstance(data, list):
                data = data[0] if data else {}

            sentiment = data.get("sentiment", "neutral")
            confidence = float(data.get("confidence", 0.5))
            return SENTIMENT_SCORES.get(sentiment, 0.0) * confidence
        except Exception as e:
            logger.debug(f"Sentiment analysis failed: {e}. Content: {content[:100]}")
            return None
