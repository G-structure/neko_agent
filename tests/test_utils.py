import os
from datetime import datetime

from neko.utils import now_iso, safe_mkdir, env_bool


def test_now_iso_format():
    value = now_iso()
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None


def test_safe_mkdir(tmp_path):
    path = tmp_path / "a" / "b"
    result = safe_mkdir(path)
    assert result == path
    assert path.exists()


def test_env_bool(monkeypatch):
    monkeypatch.setenv("FLAG", "true")
    assert env_bool("FLAG", False) is True
    monkeypatch.setenv("FLAG", "0")
    assert env_bool("FLAG", True) is False
    monkeypatch.delenv("FLAG")
    assert env_bool("FLAG", True) is True
