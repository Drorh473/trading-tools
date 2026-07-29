import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    bitget_api_key: str
    bitget_api_secret: str
    bitget_api_passphrase: str
    bitget_demo_mode: bool
    bitget_demo_api_key: str
    bitget_demo_api_secret: str
    bitget_demo_api_passphrase: str
    telegram_bot_token: str
    telegram_chat_id: str
    trades_db_path: str


def load_settings() -> Settings:
    return Settings(
        bitget_api_key=os.getenv("BITGET_API_KEY", ""),
        bitget_api_secret=os.getenv("BITGET_API_SECRET", ""),
        bitget_api_passphrase=os.getenv("BITGET_API_PASSPHRASE", ""),
        bitget_demo_mode=os.getenv("BITGET_DEMO_MODE", "false").strip().lower() == "true",
        bitget_demo_api_key=os.getenv("BITGET_DEMO_API_KEY", ""),
        bitget_demo_api_secret=os.getenv("BITGET_DEMO_API_SECRET", ""),
        bitget_demo_api_passphrase=os.getenv("BITGET_DEMO_API_PASSPHRASE", ""),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        trades_db_path=os.getenv("TRADES_DB_PATH", "data/trades.db"),
    )


settings = load_settings()
