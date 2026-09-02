from config import load_settings


def _clear_bitget_env(monkeypatch):
    # load_settings reads every field from os.environ; only the one under
    # test needs a specific value, but the others must exist (as empty
    # strings) so load_settings doesn't blow up on an unrelated KeyError.
    for name in (
        "BITGET_API_KEY", "BITGET_API_SECRET", "BITGET_API_PASSPHRASE",
        "BITGET_DEMO_API_KEY", "BITGET_DEMO_API_SECRET", "BITGET_DEMO_API_PASSPHRASE",
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
    ):
        monkeypatch.delenv(name, raising=False)


def test_send_chart_images_defaults_false(monkeypatch):
    _clear_bitget_env(monkeypatch)
    monkeypatch.delenv("SEND_CHART_IMAGES", raising=False)
    assert load_settings().send_chart_images is False


def test_send_chart_images_reads_env_true(monkeypatch):
    _clear_bitget_env(monkeypatch)
    monkeypatch.setenv("SEND_CHART_IMAGES", "true")
    assert load_settings().send_chart_images is True
