"""Unit tests for main entry point and exit codes."""

from unittest.mock import patch
from main import ExitCode, main


def test_main_invalid_config():
    code = main(["-d", "invalid_device"])
    assert code == ExitCode.INVALID_CONFIG


def test_main_help():
    code = main(["--help"])
    assert code == ExitCode.SUCCESS


def test_main_keyboard_interrupt():
    with patch("main.parse_args", side_effect=KeyboardInterrupt):
        code = main([])
        assert code == ExitCode.INTERRUPTED


def test_main_value_error():
    with patch("main.parse_args", side_effect=ValueError("Bad value")):
        code = main([])
        assert code == ExitCode.INVALID_CONFIG
