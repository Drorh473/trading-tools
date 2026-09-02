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
    # Off unless explicitly enabled, so no deploy or restart ever silently
    # comes up placing real orders. With this false the bot still sends every
    # signal and still reports the exact payload it would have placed.
    auto_execute: bool
    # Off unless explicitly enabled, matching auto_execute above - a chart
    # that renders successfully but shows a misleading picture (wrong scale,
    # mislabeled level) wouldn't be caught by chart.build()'s own fail-soft
    # handling, since that only guards against a crash. Flip this on once the
    # live pictures have actually been watched and trusted.
    send_chart_images: bool


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
        auto_execute=os.getenv("AUTO_EXECUTE", "false").strip().lower() == "true",
        send_chart_images=os.getenv("SEND_CHART_IMAGES", "false").strip().lower() == "true",
    )


settings = load_settings()
