"""Portable fingerprints and metadata shared by the pipelines."""

import hashlib
import importlib.metadata
from pathlib import Path


def file_fingerprint(path: Path) -> str:
    """Hash CSV/text content after normalizing Windows and Unix line endings."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def matches_input(path: Path, summary: dict) -> bool:
    normalized = Path(path).read_bytes().replace(b"\r\n", b"\n")
    expected = summary["input_sha256"]
    if summary.get("fingerprint_format") == "sha256-lf-v1":
        return file_fingerprint(path) == expected
    # Read historical Windows runs without changing their recorded provenance.
    return expected in {
        hashlib.sha256(normalized).hexdigest(),
        hashlib.sha256(normalized.replace(b"\n", b"\r\n")).hexdigest(),
    }


def software_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in (
        "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn",
        "joblib", "openpyxl", "threadpoolctl",
    )}
