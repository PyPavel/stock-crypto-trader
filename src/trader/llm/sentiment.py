import json
import logging
import ollama

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Analyze the sentiment of the following crypto news/posts.
Respond ONLY with valid JSON in this exact format:
{{"sentiment": "bullish" | "bearish" | "neutral", "confidence": 0.0-1.0}}

Texts:
{texts}"""

SENTIMENT_SCORES = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}


class SentimentAnalyzer:
    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url

    def score_texts(self, texts: list[str]) -> float:
        if not texts:
            return 0.0

        chunk_scores = []
        for i in range(0, len(texts), 5):
            chunk = texts[i:i + 5]
            score = self._score_chunk(chunk)
            if score is not None:
                chunk_scores.append(score)

        if not chunk_scores:
            return 0.0
        return sum(chunk_scores) / len(chunk_scores)

    def _score_chunk(self, texts: list[str]) -> float | None:
        prompt = PROMPT_TEMPLATE.format(texts="\n".join(f"- {t}" for t in texts))
        try:
            response = ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1},
            )
            content = response.message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content)
            sentiment = data.get("sentiment", "neutral")
            confidence = float(data.get("confidence", 0.5))
            return SENTIMENT_SCORES.get(sentiment, 0.0) * confidence
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return None
