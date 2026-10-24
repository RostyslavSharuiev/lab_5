"""Unit tests for AudioIO operations."""

from pathlib import Path
import numpy as np
import pytest

from audio_io import AudioIO, AudioReadError, AudioWriteError


def test_read_nonexistent_file(tmp_path: Path):
    io = AudioIO()
    non_existent = tmp_path / "does_not_exist.wav"
    with pytest.raises(AudioReadError, match="Audio file does not exist"):
        io.read(non_existent)


def test_write_and_read_audio(tmp_path: Path):
    io = AudioIO()
    test_file = tmp_path / "subdir" / "test_output.wav"
    sample_rate = 44100
    duration_samples = 4410
    audio_data = np.zeros((duration_samples, 2), dtype=np.float32)

    saved_path = io.write(test_file, audio_data, sample_rate)
    assert saved_path.exists()
    assert saved_path.is_file()

    read_data, read_sr = io.read(saved_path)
    assert read_sr == sample_rate
    assert read_data.shape == (duration_samples, 2)


def test_write_empty_data(tmp_path: Path):
    io = AudioIO()
    test_file = tmp_path / "empty.wav"
    with pytest.raises(AudioWriteError, match="Cannot write empty audio data"):
        io.write(test_file, np.array([]), 44100)


def test_write_invalid_sample_rate(tmp_path: Path):
    io = AudioIO()
    test_file = tmp_path / "invalid_sr.wav"
    with pytest.raises(AudioWriteError, match="Sample rate must be positive"):
        io.write(test_file, np.zeros((100, 2)), 0)
