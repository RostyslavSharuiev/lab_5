"""Audio input/output module for loading and saving audio files via soundfile."""

import logging
from pathlib import Path
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioIOError(Exception):
    """Base exception for audio I/O operations."""


class AudioReadError(AudioIOError):
    """Raised when loading or decoding an audio file fails."""


class AudioWriteError(AudioIOError):
    """Raised when writing or encoding an audio file fails."""


class AudioIO:
    """Handles audio loading and disk persistence using SoundFile.

    Implements AudioReaderProtocol and AudioWriterProtocol.
    """

    def read(self, file_path: Path) -> tuple[np.ndarray, int]:
        """Read audio file from disk into a 2D NumPy array.

        Args:
            file_path: Path to the audio file.

        Returns:
            tuple[np.ndarray, int]: (audio_data, sample_rate) where audio_data has shape (samples, channels).

        Raises:
            AudioReadError: If the file does not exist, is empty, or decoding fails.
        """
        resolved_path = Path(file_path).resolve()

        if not resolved_path.exists():
            raise AudioReadError(f"Audio file does not exist: {resolved_path}")

        if not resolved_path.is_file():
            raise AudioReadError(f"Path is not a regular file: {resolved_path}")

        logger.info("Loading audio file from: %s", resolved_path)
        try:
            data, sample_rate = sf.read(str(resolved_path), always_2d=True)

            if data.size == 0:
                raise AudioReadError(f"Audio file contains no samples: {resolved_path}")

            logger.info(
                "Audio loaded: %d samples, %d channels, %d Hz",
                data.shape[0],
                data.shape[1],
                sample_rate,
            )
            return data, sample_rate

        except AudioReadError:
            raise
        except Exception as err:
            logger.error("Failed to read audio file %s: %s", resolved_path, err)
            raise AudioReadError(
                f"Could not read audio file {resolved_path}: {err}"
            ) from err

    def write(self, file_path: Path, data: np.ndarray, sample_rate: int) -> Path:
        """Write 2D NumPy array of audio samples to disk.

        Args:
            file_path: Destination path for the audio file.
            data: NumPy array of shape (samples, channels).
            sample_rate: Audio sampling frequency in Hz.

        Returns:
            Path: Resolved absolute path where audio was written.

        Raises:
            AudioWriteError: If directory creation or audio encoding fails.
        """
        dest_path = Path(file_path)

        if data is None or data.size == 0:
            raise AudioWriteError("Cannot write empty audio data.")

        if sample_rate <= 0:
            raise AudioWriteError(
                f"Sample rate must be positive, got {sample_rate}"
            )

        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info("Writing audio file to: %s...", dest_path)
            sf.write(str(dest_path), data, sample_rate)

            resolved_path = dest_path.resolve()
            logger.info("Successfully saved audio file at: %s", resolved_path)
            return resolved_path

        except Exception as err:
            logger.error("Failed to write audio file to %s: %s", dest_path, err)
            raise AudioWriteError(
                f"Could not save audio file to {dest_path}: {err}"
            ) from err
