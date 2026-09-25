"""Envío de alertas por Telegram."""
from __future__ import annotations

import requests

MAX_LEN = 4000  # Telegram permite 4096 caracteres por mensaje


def send_message(bot_token: str, chat_id: str, text: str, timeout: float = 15) -> None:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    for start in range(0, len(text), MAX_LEN):
        resp = requests.post(url, data={"chat_id": chat_id, "text": text[start:start + MAX_LEN]},
                             timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Telegram respondió {resp.status_code}: {resp.text[:300]}")


class Notifier:
    """Imprime siempre en pantalla y, si está configurado, también envía a Telegram."""

    def __init__(self, bot_token: str = "", chat_id: str = "", send: bool = False):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.send = send and bool(bot_token and chat_id)

    def __call__(self, text: str) -> None:
        print(text, flush=True)
        if self.send:
            send_message(self.bot_token, self.chat_id, text)
