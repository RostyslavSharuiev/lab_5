"""Separation pipeline orchestrator coordinating reading, processing, separation, and persistence."""

import logging
from pathlib import Path

from config import AppConfig
from interfaces import (
    AudioReaderProtocol,
    AudioWriterProtocol,
    AudioPreprocessorProtocol,
    TrackSeparatorProtocol,
)

logger = logging.getLogger(__name__)


class SeparationPipeline:
    """End-to-end audio track separation pipeline adhering to DIP and SRP.

    Coordinates reading input audio, preprocessing tensor, running neural separation,
    and writing output stems to disk.
    """

    def __init__(
        self,
        config: AppConfig,
        reader: AudioReaderProtocol,
        writer: AudioWriterProtocol,
        preprocessor: AudioPreprocessorProtocol,
        separator: TrackSeparatorProtocol,
    ) -> None:
        """Initialize the pipeline with dependencies and configuration.

        Args:
            config: Aggregate application configuration.
            reader: Component implementing AudioReaderProtocol.
            writer: Component implementing AudioWriterProtocol.
            preprocessor: Component implementing AudioPreprocessorProtocol.
            separator: Component implementing TrackSeparatorProtocol.
        """
        self.config = config
        self.reader = reader
        self.writer = writer
        self.preprocessor = preprocessor
        self.separator = separator

    def run(self) -> dict[str, Path]:
        """Execute the complete separation pipeline.

        Returns:
            dict[str, Path]: Mapping from stem name to resolved path of saved audio file.
        """
        input_file = self.config.audio.input_path
        logger.info("Executing separation pipeline for input: %s", input_file)

        # 1. Read input audio file
        raw_audio, sample_rate = self.reader.read(input_file)

        # 2. Preprocess audio into normalized tensor
        wav_tensor = self.preprocessor.process(raw_audio)

        # 3. Perform source separation
        stems = self.separator.separate(wav_tensor)

        # 4. Save separated stems to disk
        saved_paths: dict[str, Path] = {}
        for stem_name, audio_data in stems.items():
            stem_path = self.config.audio.get_stem_path(stem_name)
            saved_file = self.writer.write(stem_path, audio_data, sample_rate)
            saved_paths[stem_name] = saved_file

        logger.info(
            "Pipeline successfully separated %d stems into '%s'",
            len(saved_paths),
            self.config.audio.output_dir,
        )
        return saved_paths
