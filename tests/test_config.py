from neko.config import Settings


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("NEKO_WS", "ws://server")
    monkeypatch.setenv("NEKO_LOGLEVEL", "DEBUG")
    settings = Settings.from_env()
    assert settings.neko_ws == "ws://server"
    assert settings.log_level == "DEBUG"
