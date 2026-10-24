"""Command Line Interface (CLI) module for parsing application arguments."""

import argparse
from pathlib import Path

from config import AppConfig, AudioConfig, ModelConfig, SeparationConfig


def parse_args(args: None | list[str] = None) -> AppConfig:
    """Parse command line arguments and return a validated AppConfig instance.

    Args:
        args: Optional list of command-line argument strings (useful for testing).

    Returns:
        AppConfig: Aggregate configuration object containing AudioConfig, ModelConfig, and SeparationConfig.
    """
    default_audio = AudioConfig()
    default_model = ModelConfig()
    default_separation = SeparationConfig()

    parser = argparse.ArgumentParser(
        prog="track-separator",
        description="Cross-platform command line utility to separate audio tracks using Demucs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Audio file settings
    audio_group = parser.add_argument_group("Audio Settings")
    audio_group.add_argument(
        "-i",
        "--input",
        type=Path,
        default=default_audio.input_path,
        dest="input_path",
        help="Path to input audio file to separate",
    )
    audio_group.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=default_audio.output_dir,
        dest="output_dir",
        help="Output directory path for saved stems",
    )
    audio_group.add_argument(
        "-p",
        "--prefix",
        type=str,
        default=default_audio.output_prefix,
        dest="output_prefix",
        help="Prefix added to generated stem filenames",
    )
    audio_group.add_argument(
        "-f",
        "--format",
        type=str,
        default=default_audio.output_format,
        dest="output_format",
        help="Audio format extension for output stems (e.g. wav, flac)",
    )

    # Model settings
    model_group = parser.add_argument_group("Model Settings")
    model_group.add_argument(
        "-m",
        "--model",
        type=str,
        default=default_model.name,
        dest="model_name",
        help="Pretrained Demucs model name (e.g. htdemucs, htdemucs_ft, mdx_extra)",
    )
    model_group.add_argument(
        "-d",
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default=default_model.device,
        dest="device",
        help="Execution device ('auto', 'cpu', or 'cuda')",
    )

    # Separation inference settings
    sep_group = parser.add_argument_group("Separation Settings")
    sep_group.add_argument(
        "--overlap",
        type=float,
        default=default_separation.overlap,
        dest="overlap",
        help="Overlap fraction between consecutive splits (0.0 to 1.0)",
    )
    sep_group.add_argument(
        "--shifts",
        type=int,
        default=default_separation.shifts,
        dest="shifts",
        help="Number of random shifts for equivariant stabilization (>= 1)",
    )
    sep_group.add_argument(
        "--no-split",
        action="store_false",
        dest="split",
        default=default_separation.split,
        help="Disable splitting audio into chunks (requires large GPU memory)",
    )

    parsed = parser.parse_args(args)

    audio_config = AudioConfig(
        input_path=parsed.input_path,
        output_dir=parsed.output_dir,
        output_prefix=parsed.output_prefix,
        output_format=parsed.output_format,
    )

    model_config = ModelConfig(
        name=parsed.model_name,
        device=parsed.device,
    )

    separation_config = SeparationConfig(
        split=parsed.split,
        overlap=parsed.overlap,
        shifts=parsed.shifts,
    )

    return AppConfig(
        audio=audio_config,
        model=model_config,
        separation=separation_config,
    )
