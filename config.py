"""
Laedt die Hyperparameter aus config.yaml.

Wird von train_mario.py und watch_agent.py benutzt, damit alle Skripte
dieselben Standardwerte verwenden, ohne Zahlen doppelt im Code zu pflegen.
"""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Laedt eine YAML-Konfigurationsdatei und gibt sie als dict zurueck.

    Ohne Angabe wird config.yaml im selben Ordner wie dieses Skript benutzt.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
