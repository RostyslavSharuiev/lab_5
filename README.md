# 🎵 Demucs Audio Track Separator CLI

A production-grade, cross-platform Python command-line utility for source separation of music tracks into isolated instrument stems (drums, bass, vocals, other) using state-of-the-art pretrained [Demucs](https://github.com/facebookresearch/demucs) deep learning models.

Built with Python 3.12+, `demucs`, `PyTorch`, `soundfile`, and managed with `uv`.

---

## 🚀 Features

- 🎧 **High-Quality Source Separation**: Separates any track into 4 distinct stems (`drums`, `bass`, `other`, `vocals`) using the `htdemucs` neural architecture.
- ⚙️ **Configurable Pipeline**: Custom input/output paths, file prefixes, formats (`wav`, `flac`, etc.), overlap ratios, and inference device (`cpu`, `cuda`, `auto`).
- 📁 **Dedicated Output Management**: Stems are automatically organized in an isolated output directory (default: `output/`).
- 📝 **Structured Logging**: Context-aware log messages with timestamps, log levels, and module names.
- 🛡️ **Robust Error Handling**: Standardized process exit codes (`ExitCode`) for CLI scripts and CI/CD pipelines.
- 🧪 **Unit Tested**: Comprehensive test suite covering domain configs, CLI parsing, audio I/O, preprocessing, separator lifecycle, and pipeline orchestration using `pytest`.

---

## 📋 Prerequisites

- **Python**: `>= 3.12`
- **Package Manager**: [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`
- **FFmpeg / Libsndfile**: Pre-bundled with `soundfile` wheels on Windows, macOS, and Linux.

---

## ⚡ Quick Start

### 1. Install Dependencies

Using `uv`:
```bash
uv sync
```

---

## 💻 Usage

### Basic Track Separation (Default: `audio.mp3`, model `htdemucs`, device `auto`)

```bash
uv run python main.py
```
*Saves separated stems to `output/output_drums.wav`, `output/output_bass.wav`, `output/output_other.wav`, `output/output_vocals.wav`.*

### Custom Track Separation

Separate a custom audio file using CPU, custom output folder, and `flac` format:

```bash
uv run python main.py -i song.mp3 -o separated_tracks -p stem_ -f flac -d cpu --overlap 0.5
```

---

## 🛠️ CLI Arguments Reference

### Audio Settings
| Flag | Long Flag | Default | Description |
|---|---|---|---|
| `-i` | `--input` | `audio.mp3` | Path to input audio file to separate |
| `-o` | `--output-dir` | `output` | Target directory path for saved stems |
| `-p` | `--prefix` | `output_` | Prefix prepended to stem filenames |
| `-f` | `--format` | `wav` | Audio container format extension (`wav`, `flac`) |

### Model Settings
| Flag | Long Flag | Default | Description |
|---|---|---|---|
| `-m` | `--model` | `htdemucs` | Pretrained Demucs model name (`htdemucs`, `htdemucs_ft`, etc.) |
| `-d` | `--device` | `auto` | Execution device (`auto`, `cpu`, `cuda`) |

### Separation Settings
| Flag | Long Flag | Default | Description |
|---|---|---|---|
| | `--overlap` | `0.25` | Overlap fraction between consecutive audio chunks (0.0 to 1.0) |
| | `--shifts` | `1` | Number of random shifts for equivariant stabilization (>= 1) |
| | `--no-split` | `False` | Disable chunked splitting (requires large GPU memory) |

---

## 🧪 Running Tests

To run the unit test suite with `pytest`:

```bash
uv run pytest
```

---

## 📂 Project Structure

```text
lab_5/
├── audio_io.py       # Audio file reading & writing engine via soundfile (SRP)
├── cli.py            # CLI argument parser with grouped options
├── config.py         # Strongly-typed frozen dataclasses (Audio, Model, Separation, App)
├── conftest.py       # Pytest root path configuration
├── interfaces.py     # Domain protocols (ISP, DIP, OCP)
├── logger.py         # Application logging configuration
├── main.py           # Pipeline entry point & exit code management
├── pipeline.py       # Dependency-injected SeparationPipeline orchestrator
├── preprocessor.py   # Audio tensor preparation, stereo duplication & normalization
├── separator.py      # Demucs neural model lifecycle & inference wrapper
├── pyproject.toml    # Project dependencies & metadata
├── tests/            # Full unit test suite
│   ├── test_audio_io.py
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_main.py
│   ├── test_pipeline.py
│   ├── test_preprocessor.py
│   └── test_separator.py
└── output/           # Default directory for generated stem files
```

---

## 🚦 Exit Codes

| Code | Name | Description |
|---|---|---|
| `0` | `SUCCESS` | Track separation finished successfully |
| `1` | `FAILURE` | Model loading, inference, or audio I/O error |
| `2` | `INVALID_CONFIG` | Invalid CLI arguments or configuration parameter |
| `130` | `INTERRUPTED` | Process aborted by user (`Ctrl+C`) |
