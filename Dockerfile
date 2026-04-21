FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir pandas-ta --no-deps && \
    pip install --no-cache-dir -e . --no-deps && \
    pip install --no-cache-dir ccxt pandas numpy praw requests openai \
        fastapi uvicorn pyyaml pydantic jinja2 apscheduler alpaca-py pytz pytrends \
        lightgbm scikit-learn pyarrow yfinance "tastytrade>=8.0,<9.0"

COPY src/ src/
COPY config.yaml .
COPY models/ models/

EXPOSE 8000

CMD ["python", "-m", "trader", "--config", "config.yaml"]
