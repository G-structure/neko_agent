import json

from neko.logging import setup_logging


def test_setup_logging_json_format(capsys):
    logger = setup_logging(level="INFO", fmt="json")
    logger.info("hello")
    err = capsys.readouterr().err.strip().splitlines()[-1]
    data = json.loads(err)
    assert data["msg"] == "hello"
    assert data["level"] == "INFO"


def test_setup_logging_text_level(capsys):
    logger = setup_logging(level="DEBUG", fmt="text")
    logger.debug("hi")
    captured = capsys.readouterr().err
    assert "hi" in captured
