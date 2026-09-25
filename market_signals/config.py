"""Configuración leída desde variables de entorno o desde un archivo .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

TRADIER_URLS = {
    "sandbox": "https://sandbox.tradier.com/v1",
    "live": "https://api.tradier.com/v1",
}


def load_dotenv(path: str | Path = ".env") -> None:
    """Carga KEY=VALUE de un archivo .env sin sobrescribir variables ya definidas."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    tradier_token: str
    tradier_env: str
    tradier_account_id: str
    telegram_bot_token: str
    telegram_chat_id: str
    account_size: float
    risk_per_trade_pct: float
    option_stop_pct: float

    @property
    def tradier_base_url(self) -> str:
        return TRADIER_URLS[self.tradier_env]

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


def load_settings(env_file: str | Path = ".env") -> Settings:
    load_dotenv(env_file)
    env = os.environ.get("TRADIER_ENV", "sandbox").lower()
    if env not in TRADIER_URLS:
        raise ValueError(f"TRADIER_ENV debe ser 'sandbox' o 'live', no '{env}'")
    return Settings(
        tradier_token=os.environ.get("TRADIER_TOKEN", ""),
        tradier_env=env,
        tradier_account_id=os.environ.get("TRADIER_ACCOUNT_ID", ""),
        telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
        account_size=float(os.environ.get("ACCOUNT_SIZE", "8000")),
        risk_per_trade_pct=float(os.environ.get("RISK_PER_TRADE_PCT", "1.0")),
        option_stop_pct=float(os.environ.get("OPTION_STOP_PCT", "30")),
    )
