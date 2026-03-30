import logging
import requests

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self._token = bot_token
        self._chat_id = chat_id

    def send(self, message: str) -> None:
        """Send a Telegram message. Silently logs on failure — never raises."""
        if not self._token or not self._chat_id:
            return
        url = TELEGRAM_API.format(token=self._token)
        try:
            resp = requests.post(url, json={"chat_id": self._chat_id, "text": message}, timeout=5)
            if not resp.ok:
                logger.warning("Telegram send failed: %s", resp.text)
        except Exception as exc:
            logger.warning("Telegram send error: %s", exc)
