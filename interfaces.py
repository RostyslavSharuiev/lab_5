"""Domain protocols (interfaces) defining audio processing and separation contracts.

Follows the Interface Segregation Principle (ISP) and Dependency Inversion Principle (DIP).
"""

from pathlib import Path
from typing import Protocol, runtime_checkable
import numpy as np
import torch


@runtime_checkable
class AudioReaderProtocol(Protocol):
    """Interface for reading audio files from disk."""

    def read(self, file_path: Path) -> tuple[np.ndarray, int]:
        """Read audio file and return audio data as numpy array and sample rate.

        Args:
            file_path: Path to the target audio file.

        Returns:
            tuple[np.ndarray, int]: Audio array of shape (samples, channels) and sample rate in Hz.
        """
        ...


@runtime_checkable
class AudioWriterProtocol(Protocol):
    """Interface for writing audio stems to disk."""

    def write(self, file_path: Path, data: np.ndarray, sample_rate: int) -> Path:
        """Write audio sample array to disk.

        Args:
            file_path: Destination path for the audio file.
            data: Audio sample array of shape (samples, channels).
            sample_rate: Sampling frequency in Hz.

        Returns:
            Path: Resolved absolute path to the written file.
        """
        ...


@runtime_checkable
class AudioPreprocessorProtocol(Protocol):
    """Interface for audio preprocessing and normalization."""

    def process(self, audio_data: np.ndarray) -> torch.Tensor:
        """Transform raw numpy audio data into a normalized PyTorch tensor ready for model inference.

        Args:
            audio_data: Raw audio data from audio reader, shape (samples, channels).

        Returns:
            torch.Tensor: Normalized float32 tensor of shape (channels, samples).
        """
        ...


@runtime_checkable
class TrackSeparatorProtocol(Protocol):
    """Interface for neural audio source separation models."""

    @property
    def sources(self) -> list[str]:
        """List of track stem names output by the model (e.g. ['drums', 'bass', 'other', 'vocals'])."""
        ...

    def separate(self, wav: torch.Tensor) -> dict[str, np.ndarray]:
        """Separate multi-channel audio tensor into component stems.

        Args:
            wav: Preprocessed audio tensor of shape (channels, samples).

        Returns:
            dict[str, np.ndarray]: Mapping from stem name to numpy array (samples, channels).
        """
        ...
