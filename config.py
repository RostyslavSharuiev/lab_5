"""Configuration models for Audio Track Separation application."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AudioConfig:
    """Configuration settings for audio input file and separated output stems.

    Attributes:
        input_path: Path to the input audio file (MP3, WAV, etc.).
        output_dir: Directory where separated track stems will be saved.
        output_prefix: Prefix added to stem filenames (default: 'output_').
        output_format: Audio container format extension (default: 'wav').
    """

    input_path: Path = field(default_factory=lambda: Path("audio.mp3"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    output_prefix: str = "output_"
    output_format: str = "wav"

    def __post_init__(self) -> None:
        """Validate audio configuration parameters."""
        if not str(self.input_path).strip():
            raise ValueError("Input audio path cannot be empty")

        cleaned_format = self.output_format.lstrip(".")
        if not cleaned_format:
            raise ValueError("Output format extension cannot be empty")

        object.__setattr__(self, "output_format", cleaned_format)

    def get_stem_path(self, stem_name: str) -> Path:
        """Return target file path for a separated audio stem.

        Args:
            stem_name: Stem identifier (e.g., 'drums', 'bass', 'vocals', 'other').

        Returns:
            Path: Fully constructed output path for the stem file.
        """
        clean_name = stem_name.strip()
        if not clean_name:
            raise ValueError("Stem name cannot be empty")
        filename = f"{self.output_prefix}{clean_name}.{self.output_format}"
        return self.output_dir / filename


@dataclass(frozen=True)
class ModelConfig:
    """Configuration settings for the Demucs source separation model.

    Attributes:
        name: Pretrained Demucs model name (e.g., 'htdemucs', 'htdemucs_ft', 'mdx_extra').
        device: Target execution device ('auto', 'cpu', or 'cuda').
    """

    name: str = "htdemucs"
    device: str = "auto"

    def __post_init__(self) -> None:
        """Validate model configuration parameters."""
        if not self.name.strip():
            raise ValueError("Model name cannot be empty")

        allowed_devices = {"auto", "cpu", "cuda"}
        normalized_device = self.device.strip().lower()
        if normalized_device not in allowed_devices:
            raise ValueError(
                f"Device must be one of {allowed_devices}, got '{self.device}'"
            )

        object.__setattr__(self, "device", normalized_device)

    def get_resolved_device(self, cuda_available: bool | None = None) -> str:
        """Resolve 'auto' device to 'cuda' (if available) or 'cpu'.

        Args:
            cuda_available: Optional override for CUDA availability (useful in unit tests).

        Returns:
            str: 'cuda' or 'cpu'.
        """
        if self.device == "auto":
            if cuda_available is None:
                import torch

                cuda_available = torch.cuda.is_available()
            return "cuda" if cuda_available else "cpu"
        return self.device


@dataclass(frozen=True)
class SeparationConfig:
    """Configuration settings for Demucs inference execution.

    Attributes:
        split: Whether to split audio into chunks to limit memory consumption.
        overlap: Overlap fraction between consecutive chunks in range [0.0, 1.0).
        shifts: Number of random shifts for equivariant stabilization (>= 1).
    """

    split: bool = True
    overlap: float = 0.25
    shifts: int = 1

    def __post_init__(self) -> None:
        """Validate separation configuration parameters."""
        if not (0.0 <= self.overlap < 1.0):
            raise ValueError(
                f"Overlap must be in range [0.0, 1.0), got {self.overlap}"
            )

        if self.shifts < 1:
            raise ValueError(f"Shifts must be an integer >= 1, got {self.shifts}")


@dataclass(frozen=True)
class AppConfig:
    """Aggregate configuration holding all domain-specific settings."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
