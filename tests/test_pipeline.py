"""Unit tests for SeparationPipeline and pipeline orchestration."""

from pathlib import Path
import numpy as np
import pytest
import torch

from config import AppConfig, AudioConfig, ModelConfig, SeparationConfig
from pipeline import SeparationPipeline


class DummyReader:
    def __init__(self, data: np.ndarray, sample_rate: int = 44100) -> None:
        self.data = data
        self.sample_rate = sample_rate
        self.read_called = False

    def read(self, file_path: Path) -> tuple[np.ndarray, int]:
        self.read_called = True
        return self.data, self.sample_rate


class DummyWriter:
    def __init__(self) -> None:
        self.written_files: dict[Path, np.ndarray] = {}

    def write(self, file_path: Path, data: np.ndarray, sample_rate: int) -> Path:
        self.written_files[file_path] = data
        return file_path


class DummyPreprocessor:
    def __init__(self) -> None:
        self.process_called = False

    def process(self, audio_data: np.ndarray) -> torch.Tensor:
        self.process_called = True
        return torch.zeros((2, len(audio_data)), dtype=torch.float32)


class DummySeparator:
    def __init__(self, stems: list[str]) -> None:
        self._stems = stems
        self.separate_called = False

    @property
    def sources(self) -> list[str]:
        return self._stems

    def separate(self, wav: torch.Tensor) -> dict[str, np.ndarray]:
        self.separate_called = True
        samples = wav.shape[1]
        return {name: np.zeros((samples, 2), dtype=np.float32) for name in self._stems}


def test_pipeline_execution(tmp_path: Path):
    audio_config = AudioConfig(
        input_path=Path("song.mp3"),
        output_dir=tmp_path,
        output_prefix="out_",
        output_format="wav",
    )
    app_config = AppConfig(audio=audio_config)

    audio_samples = 1000
    dummy_data = np.ones((audio_samples, 2), dtype=np.float32)

    reader = DummyReader(dummy_data, sample_rate=44100)
    writer = DummyWriter()
    preprocessor = DummyPreprocessor()
    separator = DummySeparator(["drums", "bass", "other", "vocals"])

    pipeline = SeparationPipeline(
        config=app_config,
        reader=reader,
        writer=writer,
        preprocessor=preprocessor,
        separator=separator,
    )

    result = pipeline.run()

    assert reader.read_called is True
    assert preprocessor.process_called is True
    assert separator.separate_called is True

    assert set(result.keys()) == {"drums", "bass", "other", "vocals"}
    for stem_name, path in result.items():
        assert path == tmp_path / f"out_{stem_name}.wav"
        assert path in writer.written_files
