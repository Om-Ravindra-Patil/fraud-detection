from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = PROJECT_ROOT / "sql"


def load_config(path: Path | None = None) -> dict:
    """Load configs/config.yaml and resolve relative paths against the project root."""
    path = path or PROJECT_ROOT / "configs" / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: PROJECT_ROOT / v for k, v in cfg["paths"].items()}
    return cfg
