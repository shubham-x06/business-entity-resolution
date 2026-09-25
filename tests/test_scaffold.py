"""
test_scaffold.py  –  Smoke tests verifying the repo scaffold is importable.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

# ── Repo root ───────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "code" / "business_entity_resolution" / "src"

# Add src/ to the import path so the package is importable.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ── Module import tests ────────────────────────────────────────────
MODULES = [
    "business_entity_resolution",
    "business_entity_resolution.normalize",
    "business_entity_resolution.blocking",
    "business_entity_resolution.features",
    "business_entity_resolution.model",
    "business_entity_resolution.grouping",
    "business_entity_resolution.pipeline",
    "business_entity_resolution.io_utils",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_importable(module_name: str) -> None:
    """Each pipeline module should be importable without errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None


# ── Config file existence tests ─────────────────────────────────────
CONFIG_DIR = REPO_ROOT / "code" / "business_entity_resolution" / "configs"
CONFIG_FILES = ["paths.yaml", "blocking.yaml", "model.yaml"]


@pytest.mark.parametrize("filename", CONFIG_FILES)
def test_config_exists(filename: str) -> None:
    """Each YAML config must exist."""
    assert (CONFIG_DIR / filename).is_file(), f"Missing config: {filename}"


# ── YAML parseable tests ───────────────────────────────────────────
def test_configs_parseable() -> None:
    """All YAML configs must be valid YAML."""
    import yaml

    for filename in CONFIG_FILES:
        path = CONFIG_DIR / filename
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert isinstance(data, dict), f"{filename} did not parse to a dict"


# ── Normalize smoke test ───────────────────────────────────────────
def test_normalize_business_name() -> None:
    """Basic normalisation should lowercase and strip punctuation."""
    from business_entity_resolution.normalize import normalize_business_name

    result = normalize_business_name("Acme Corporation, Inc.")
    assert "acme" in result
    assert result == result.lower()


def test_normalize_handles_none() -> None:
    """None / NaN inputs should return an empty string."""
    from business_entity_resolution.normalize import normalize_text

    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""


# ── io_utils smoke test ────────────────────────────────────────────
def test_load_config() -> None:
    """load_config should return a dict."""
    from business_entity_resolution.io_utils import load_config

    cfg = load_config(CONFIG_DIR / "paths.yaml")
    assert isinstance(cfg, dict)
    assert "data" in cfg


# ── CLI --help exit-code tests ─────────────────────────────────────
SCRIPTS_DIR = REPO_ROOT / "code" / "business_entity_resolution" / "scripts"


@pytest.mark.parametrize(
    "script",
    ["run_train.py", "run_infer.py"],
)
def test_script_help(script: str) -> None:
    """--help must exit 0."""
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{script} --help failed:\n{result.stderr}"


# ── Version string ─────────────────────────────────────────────────
def test_version() -> None:
    """Package should expose a __version__ attribute."""
    import business_entity_resolution

    assert hasattr(business_entity_resolution, "__version__")
    assert isinstance(business_entity_resolution.__version__, str)
