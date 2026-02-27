FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir pandas-ta --no-deps && \
    pip install --no-cache-dir -e . --no-deps && \
    pip install --no-cache-dir ccxt pandas numpy praw requests ollama anthropic \
        fastapi uvicorn pyyaml pydantic jinja2 apscheduler

COPY src/ src/
COPY config.yaml .

EXPOSE 8000

CMD ["python", "-m", "trader", "--config", "config.yaml"]
