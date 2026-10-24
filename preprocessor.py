"""Audio preprocessing and tensor normalization module."""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Raised when audio tensor preparation or normalization fails."""


class AudioPreprocessor:
    """Prepares and normalizes raw audio signals for neural network inference.

    Implements AudioPreprocessorProtocol.
    """

    def to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """Convert numpy array of shape (samples, channels) to torch tensor of shape (channels, samples).

        Args:
            data: 2D NumPy array of audio samples (samples x channels).

        Returns:
            torch.Tensor: PyTorch float32 tensor (channels x samples).

        Raises:
            PreprocessingError: If input data is None or empty.
        """
        if data is None or data.size == 0:
            raise PreprocessingError("Audio data cannot be None or empty.")

        if data.ndim == 1:
            data = data[:, np.newaxis]

        if data.ndim != 2:
            raise PreprocessingError(
                f"Expected 1D or 2D array, got shape {data.shape}"
            )

        # Transpose from (samples, channels) to (channels, samples)
        return torch.tensor(data.T, dtype=torch.float32)

    def ensure_stereo(self, wav: torch.Tensor) -> torch.Tensor:
        """Ensure audio tensor has at least 2 channels (duplicate mono to stereo).

        Args:
            wav: PyTorch tensor of shape (channels, samples).

        Returns:
            torch.Tensor: PyTorch tensor with at least 2 channels.

        Raises:
            PreprocessingError: If tensor is not 2D.
        """
        if wav.ndim != 2:
            raise PreprocessingError(
                f"Expected 2D tensor (channels, samples), got ndim={wav.ndim}"
            )

        if wav.shape[0] == 1:
            logger.debug("Duplicating mono channel to stereo.")
            return wav.repeat(2, 1)

        return wav

    def normalize(self, wav: torch.Tensor) -> torch.Tensor:
        """Apply zero-mean and unit-variance normalization as performed internally by Demucs.

        Args:
            wav: PyTorch tensor of shape (channels, samples).

        Returns:
            torch.Tensor: Normalized PyTorch tensor.
        """
        ref = wav.mean(0)
        std = ref.std() + 1e-8
        normalized = (wav - ref.mean()) / std
        return normalized

    def process(self, audio_data: np.ndarray) -> torch.Tensor:
        """Run full preprocessing pipeline on numpy audio data.

        Args:
            audio_data: Raw NumPy audio data.

        Returns:
            torch.Tensor: Preprocessed and normalized PyTorch tensor ready for model inference.
        """
        try:
            logger.debug("Converting audio data to PyTorch tensor...")
            wav = self.to_tensor(audio_data)

            logger.debug("Ensuring stereo channels...")
            wav = self.ensure_stereo(wav)

            logger.debug("Applying audio volume normalization...")
            wav = self.normalize(wav)

            logger.info("Audio preprocessed successfully into tensor shape %s", list(wav.shape))
            return wav

        except PreprocessingError:
            raise
        except Exception as err:
            logger.error("Audio preprocessing error: %s", err)
            raise PreprocessingError(
                f"Failed to preprocess audio signal: {err}"
            ) from err
