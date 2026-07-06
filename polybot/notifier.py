"""Telegram notifications (alerts only — control commands belong to the
external risk-gate/supervisor bot, kept as a separate process on purpose).

Outbound messages pass through a redaction filter so a private key or token
can never leak into a chat, even by bug.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request

log = logging.getLogger("notify")

_HEX_KEY = re.compile(r"0x[0-9a-fA-F]{64}")
_BOT_TOKEN = re.compile(r"\d{8,10}:[A-Za-z0-9_-]{30,}")


def redact(text: str) -> str:
    text = _HEX_KEY.sub("[REDACTED_KEY]", text)
    text = _BOT_TOKEN.sub("[REDACTED_TOKEN]", text)
    return text


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str) -> None:
        self.token = bot_token
        self.chat_id = chat_id

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str) -> None:
        text = redact(text)
        if not self.enabled:
            log.info("[notify-disabled] %s", text)
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = urllib.parse.urlencode({"chat_id": self.chat_id, "text": text}).encode()
        try:
            with urllib.request.urlopen(url, payload, timeout=5) as r:
                json.loads(r.read())
        except Exception as e:  # noqa: BLE001 — notification failure must not kill the engine
            log.warning("telegram send failed: %s", e)
