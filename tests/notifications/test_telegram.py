from unittest.mock import patch, MagicMock
from trader.notifications.telegram import TelegramNotifier


def test_send_calls_requests_post():
    notifier = TelegramNotifier(bot_token="tok123", chat_id="42")
    with patch("trader.notifications.telegram.requests.post") as mock_post:
        mock_post.return_value.ok = True
        notifier.send("BUY BTC-USD 0.001 @ 50000.00")
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "tok123" in call_args[0][0]  # URL contains token
        assert call_args[1]["json"]["chat_id"] == "42"
        assert "BUY BTC-USD" in call_args[1]["json"]["text"]


def test_send_no_op_when_unconfigured():
    notifier = TelegramNotifier(bot_token="", chat_id="")
    with patch("trader.notifications.telegram.requests.post") as mock_post:
        notifier.send("anything")
        mock_post.assert_not_called()


def test_send_logs_warning_on_http_error():
    notifier = TelegramNotifier(bot_token="tok", chat_id="42")
    with patch("trader.notifications.telegram.requests.post") as mock_post:
        mock_post.return_value.ok = False
        mock_post.return_value.text = "Bad Request"
        import logging
        with patch.object(logging.getLogger("trader.notifications.telegram"), "warning") as mock_warn:
            notifier.send("test")
            mock_warn.assert_called_once()


def test_send_logs_warning_on_exception():
    notifier = TelegramNotifier(bot_token="tok", chat_id="42")
    with patch("trader.notifications.telegram.requests.post", side_effect=Exception("timeout")):
        import logging
        with patch.object(logging.getLogger("trader.notifications.telegram"), "warning") as mock_warn:
            notifier.send("test")
            mock_warn.assert_called_once()
