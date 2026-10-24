"""Unit tests for configuration models."""

from pathlib import Path
import pytest

from config import AppConfig, AudioConfig, ModelConfig, SeparationConfig


def test_default_audio_config():
    config = AudioConfig()
    assert config.input_path == Path("audio.mp3")
    assert config.output_dir == Path("output")
    assert config.output_prefix == "output_"
    assert config.output_format == "wav"
    assert config.get_stem_path("drums") == Path("output/output_drums.wav")


def test_custom_audio_config():
    config = AudioConfig(
        input_path=Path("my_song.flac"),
        output_dir=Path("custom_stems"),
        output_prefix="stem_",
        output_format=".flac",
    )
    assert config.input_path == Path("my_song.flac")
    assert config.output_dir == Path("custom_stems")
    assert config.output_prefix == "stem_"
    assert config.output_format == "flac"
    assert config.get_stem_path("vocals") == Path("custom_stems/stem_vocals.flac")


def test_invalid_audio_config():
    with pytest.raises(ValueError, match="Input audio path cannot be empty"):
        AudioConfig(input_path=Path("   "))

    with pytest.raises(ValueError, match="Output format extension cannot be empty"):
        AudioConfig(output_format="...")


def test_invalid_stem_name():
    config = AudioConfig()
    with pytest.raises(ValueError, match="Stem name cannot be empty"):
        config.get_stem_path("   ")


def test_default_model_config():
    config = ModelConfig()
    assert config.name == "htdemucs"
    assert config.device == "auto"


def test_custom_model_config():
    config = ModelConfig(name="mdx_extra", device="CPU")
    assert config.name == "mdx_extra"
    assert config.device == "cpu"


def test_invalid_model_config():
    with pytest.raises(ValueError, match="Model name cannot be empty"):
        ModelConfig(name="")

    with pytest.raises(ValueError, match="Device must be one of"):
        ModelConfig(device="tpu")


def test_model_device_resolution():
    config_auto = ModelConfig(device="auto")
    assert config_auto.get_resolved_device(cuda_available=False) == "cpu"
    assert config_auto.get_resolved_device(cuda_available=True) == "cuda"

    config_cpu = ModelConfig(device="cpu")
    assert config_cpu.get_resolved_device(cuda_available=True) == "cpu"


def test_separation_config_defaults():
    config = SeparationConfig()
    assert config.split is True
    assert config.overlap == 0.25
    assert config.shifts == 1


def test_separation_config_validation():
    with pytest.raises(ValueError, match="Overlap must be in range"):
        SeparationConfig(overlap=1.5)

    with pytest.raises(ValueError, match="Overlap must be in range"):
        SeparationConfig(overlap=-0.1)

    with pytest.raises(ValueError, match="Shifts must be an integer >= 1"):
        SeparationConfig(shifts=0)


def test_app_config_aggregate():
    audio = AudioConfig(input_path=Path("test.wav"))
    model = ModelConfig(name="htdemucs_ft")
    sep = SeparationConfig(overlap=0.5)

    app_config = AppConfig(audio=audio, model=model, separation=sep)
    assert app_config.audio.input_path == Path("test.wav")
    assert app_config.model.name == "htdemucs_ft"
    assert app_config.separation.overlap == 0.5
