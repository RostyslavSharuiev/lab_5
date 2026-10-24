"""Source separation module encapsulating Demucs model inference."""

import logging
from typing import Any
import numpy as np
import torch

from config import ModelConfig, SeparationConfig

logger = logging.getLogger(__name__)


class SeparatorError(Exception):
    """Base exception for track separation operations."""


class ModelLoadError(SeparatorError):
    """Raised when loading Demucs neural network model fails."""


class InferenceError(SeparatorError):
    """Raised when running model inference fails."""


class DemucsSeparator:
    """Audio source separation engine using pretrained Demucs models.

    Implements TrackSeparatorProtocol.
    """

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        separation_config: SeparationConfig | None = None,
    ) -> None:
        """Initialize DemucsSeparator with configuration settings.

        Args:
            model_config: Model configuration (name, device).
            separation_config: Inference separation settings (split, overlap, shifts).
        """
        self.model_config = model_config or ModelConfig()
        self.separation_config = separation_config or SeparationConfig()
        self._model: Any = None
        self._device: str = self.model_config.get_resolved_device()

    @property
    def device(self) -> str:
        """Target execution device ('cpu' or 'cuda')."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Whether the Demucs model is currently loaded in memory."""
        return self._model is not None

    @property
    def sources(self) -> list[str]:
        """Names of the separated stem sources (e.g. ['drums', 'bass', 'other', 'vocals'])."""
        if not self.is_loaded:
            self.load_model()
        return list(self._model.sources)

    def load_model(self) -> None:
        """Load the pretrained Demucs model and transfer it to the target device.

        Raises:
            ModelLoadError: If loading the pretrained weights fails.
        """
        if self._model is not None:
            return

        logger.info(
            "Loading Demucs model '%s' onto device '%s'...",
            self.model_config.name,
            self._device,
        )
        try:
            from demucs.pretrained import get_model

            model = get_model(self.model_config.name)
            model.to(self._device)
            model.eval()

            self._model = model
            logger.info(
                "Demucs model '%s' loaded successfully. Available stems: %s",
                self.model_config.name,
                model.sources,
            )
        except Exception as err:
            logger.error(
                "Failed to load Demucs model '%s': %s",
                self.model_config.name,
                err,
            )
            raise ModelLoadError(
                f"Could not load Demucs model '{self.model_config.name}': {err}"
            ) from err

    def separate(self, wav: torch.Tensor) -> dict[str, np.ndarray]:
        """Separate preprocessed audio tensor into component stems.

        Args:
            wav: Preprocessed PyTorch tensor of shape (channels, samples).

        Returns:
            dict[str, np.ndarray]: Mapping from stem name to 2D NumPy array (samples, channels).

        Raises:
            InferenceError: If model inference fails.
        """
        if not self.is_loaded:
            self.load_model()

        logger.info(
            "Running Demucs source separation (split=%s, overlap=%.2f, shifts=%d)...",
            self.separation_config.split,
            self.separation_config.overlap,
            self.separation_config.shifts,
        )

        try:
            from demucs.apply import apply_model

            # apply_model expects batch format: (batch_size, channels, samples)
            wav_batch = wav[None].to(self._device)

            with torch.no_grad():
                sources_tensor = apply_model(
                    self._model,
                    wav_batch,
                    split=self.separation_config.split,
                    overlap=self.separation_config.overlap,
                    shifts=self.separation_config.shifts,
                )[0]

            source_names = self._model.sources
            stems: dict[str, np.ndarray] = {}

            for idx, source_tensor in enumerate(sources_tensor):
                name = source_names[idx]
                # Convert from (channels, samples) back to (samples, channels) numpy array
                audio_data = source_tensor.cpu().numpy().T
                stems[name] = audio_data

            logger.info(
                "Separation complete. Generated %d stems: %s",
                len(stems),
                list(stems.keys()),
            )
            return stems

        except Exception as err:
            logger.error("Error during Demucs model inference: %s", err)
            raise InferenceError(
                f"Demucs track separation inference failed: {err}"
            ) from err
