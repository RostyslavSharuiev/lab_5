"""Main application entry point for the Audio Track Separator CLI."""

import logging
import sys
from enum import IntEnum

from audio_io import AudioIO, AudioIOError
from cli import parse_args
from config import AppConfig
from logger import setup_logging
from pipeline import SeparationPipeline
from preprocessor import AudioPreprocessor, PreprocessingError
from separator import DemucsSeparator, SeparatorError


class ExitCode(IntEnum):
    """Standardized application exit codes."""

    SUCCESS = 0
    FAILURE = 1
    INVALID_CONFIG = 2
    INTERRUPTED = 130


def main(argv: list[str] | None = None) -> int:
    """Run the audio track separation pipeline.

    Args:
        argv: Optional list of command-line arguments.

    Returns:
        int: Process exit code.
    """
    setup_logging()
    logger = logging.getLogger("main")

    try:
        app_config: AppConfig = parse_args(argv)
        logger.info("Initializing audio separator pipeline...")

        audio_io = AudioIO()
        preprocessor = AudioPreprocessor()
        separator = DemucsSeparator(
            model_config=app_config.model,
            separation_config=app_config.separation,
        )

        pipeline = SeparationPipeline(
            config=app_config,
            reader=audio_io,
            writer=audio_io,
            preprocessor=preprocessor,
            separator=separator,
        )

        logger.info("Starting audio separation for: %s", app_config.audio.input_path)
        saved_stems = pipeline.run()

        logger.info("Audio track separation completed successfully!")
        for stem_name, path in saved_stems.items():
            logger.info("Stem '%s' saved to: %s", stem_name, path)

        return ExitCode.SUCCESS

    except SystemExit as err:
        return ExitCode.SUCCESS if err.code == 0 else ExitCode.INVALID_CONFIG
    except KeyboardInterrupt:
        logger.warning("\nSeparation process aborted by user.")
        return ExitCode.INTERRUPTED
    except ValueError as err:
        logger.error("Configuration error: %s", err)
        return ExitCode.INVALID_CONFIG
    except (AudioIOError, PreprocessingError, SeparatorError) as err:
        logger.error("Application error: %s", err)
        return ExitCode.FAILURE
    except Exception as err:
        logger.critical("Unexpected failure: %s", err, exc_info=True)
        return ExitCode.FAILURE


if __name__ == "__main__":
    sys.exit(main())