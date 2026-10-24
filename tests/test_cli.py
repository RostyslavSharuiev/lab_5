"""Unit tests for CLI argument parsing."""

from pathlib import Path
import pytest

from cli import parse_args


def test_parse_args_defaults():
    app_config = parse_args([])
    assert app_config.audio.input_path == Path("audio.mp3")
    assert app_config.audio.output_dir == Path("output")
    assert app_config.audio.output_prefix == "output_"
    assert app_config.audio.output_format == "wav"
    assert app_config.model.name == "htdemucs"
    assert app_config.model.device == "auto"
    assert app_config.separation.split is True
    assert app_config.separation.overlap == 0.25
    assert app_config.separation.shifts == 1


def test_parse_args_custom_flags():
    args = [
        "-i", "input_track.wav",
        "-o", "custom_dir",
        "-p", "track_",
        "-f", "flac",
        "-m", "htdemucs_ft",
        "-d", "cpu",
        "--overlap", "0.5",
        "--shifts", "2",
        "--no-split",
    ]
    app_config = parse_args(args)
    assert app_config.audio.input_path == Path("input_track.wav")
    assert app_config.audio.output_dir == Path("custom_dir")
    assert app_config.audio.output_prefix == "track_"
    assert app_config.audio.output_format == "flac"
    assert app_config.model.name == "htdemucs_ft"
    assert app_config.model.device == "cpu"
    assert app_config.separation.split is False
    assert app_config.separation.overlap == 0.5
    assert app_config.separation.shifts == 2


def test_parse_args_invalid_device():
    with pytest.raises(SystemExit):
        parse_args(["-d", "invalid_device"])
