"""Unit tests for DemucsSeparator."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

from config import ModelConfig, SeparationConfig
from separator import DemucsSeparator, InferenceError, ModelLoadError


def test_separator_init():
    model_cfg = ModelConfig(name="htdemucs", device="cpu")
    sep_cfg = SeparationConfig(overlap=0.5)
    separator = DemucsSeparator(model_cfg, sep_cfg)

    assert separator.is_loaded is False
    assert separator.device == "cpu"
    assert separator.model_config.name == "htdemucs"
    assert separator.separation_config.overlap == 0.5


def test_separator_load_model_failure():
    model_cfg = ModelConfig(name="non_existent_model_xyz", device="cpu")
    separator = DemucsSeparator(model_cfg)

    with patch("demucs.pretrained.get_model", side_effect=RuntimeError("Model not found")):
        with pytest.raises(ModelLoadError, match="Could not load Demucs model"):
            separator.load_model()


def test_separator_separate_mocked():
    model_cfg = ModelConfig(name="htdemucs", device="cpu")
    sep_cfg = SeparationConfig()
    separator = DemucsSeparator(model_cfg, sep_cfg)

    # Mock model
    mock_model = MagicMock()
    mock_model.sources = ["drums", "bass", "other", "vocals"]
    separator._model = mock_model

    # Simulated output from apply_model: shape (1, 4, 2, 100) -> [0] gives (4, 2, 100)
    mock_sources_tensor = torch.zeros((4, 2, 100), dtype=torch.float32)

    with patch("demucs.apply.apply_model", return_value=[mock_sources_tensor]):
        input_wav = torch.zeros((2, 100), dtype=torch.float32)
        stems = separator.separate(input_wav)

        assert set(stems.keys()) == {"drums", "bass", "other", "vocals"}
        assert stems["drums"].shape == (100, 2)
        assert isinstance(stems["drums"], np.ndarray)


def test_separator_inference_error():
    model_cfg = ModelConfig(name="htdemucs", device="cpu")
    separator = DemucsSeparator(model_cfg)
    separator._model = MagicMock()
    separator._model.sources = ["drums", "bass", "other", "vocals"]

    with patch("demucs.apply.apply_model", side_effect=RuntimeError("CUDA OOM")):
        with pytest.raises(InferenceError, match="Demucs track separation inference failed"):
            separator.separate(torch.zeros((2, 50)))
