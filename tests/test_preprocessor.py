"""Unit tests for AudioPreprocessor."""

import numpy as np
import pytest
import torch

from preprocessor import AudioPreprocessor, PreprocessingError


def test_to_tensor_2d():
    preprocessor = AudioPreprocessor()
    data = np.ones((100, 2), dtype=np.float32)
    tensor = preprocessor.to_tensor(data)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (2, 100)
    assert tensor.dtype == torch.float32


def test_to_tensor_1d():
    preprocessor = AudioPreprocessor()
    data = np.ones(100, dtype=np.float32)
    tensor = preprocessor.to_tensor(data)

    assert tensor.shape == (1, 100)


def test_to_tensor_empty():
    preprocessor = AudioPreprocessor()
    with pytest.raises(PreprocessingError, match="Audio data cannot be None or empty"):
        preprocessor.to_tensor(np.array([]))


def test_ensure_stereo_mono():
    preprocessor = AudioPreprocessor()
    mono_wav = torch.ones((1, 50), dtype=torch.float32)
    stereo_wav = preprocessor.ensure_stereo(mono_wav)

    assert stereo_wav.shape == (2, 50)
    assert torch.equal(stereo_wav[0], stereo_wav[1])


def test_ensure_stereo_already_stereo():
    preprocessor = AudioPreprocessor()
    original_stereo = torch.randn((2, 50), dtype=torch.float32)
    result = preprocessor.ensure_stereo(original_stereo)

    assert result.shape == (2, 50)
    assert torch.equal(original_stereo, result)


def test_normalize():
    preprocessor = AudioPreprocessor()
    wav = torch.tensor([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]], dtype=torch.float32)
    normalized = preprocessor.normalize(wav)

    assert normalized.shape == wav.shape
    # Check that mean across channels/samples is approximately 0
    ref = normalized.mean(0)
    assert abs(ref.mean().item()) < 1e-4


def test_process_pipeline():
    preprocessor = AudioPreprocessor()
    raw = np.random.randn(200, 2).astype(np.float32)
    processed = preprocessor.process(raw)

    assert isinstance(processed, torch.Tensor)
    assert processed.shape == (2, 200)
