#!/usr/bin/env python3
"""Build the stylometry profile from the local corpus.

Usage:
    python train.py                 # uses config.yaml
    python train.py --config x.yaml
"""
from __future__ import annotations

import argparse
import json
import sys

from src.corpus_loader import load_documents
from src.model import StylometryModel
from src.store import load_config, save_model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    target = load_documents(cfg["corpus_dir"])
    impostor = load_documents(cfg.get("impostor_dir", "")) if cfg.get("impostor_dir") else []

    print(f"Loaded {len(target)} target document(s) from {cfg['corpus_dir']!r}")
    if impostor:
        print(f"Loaded {len(impostor)} impostor document(s) -> supervised mode")

    if len(target) < 2:
        print("ERROR: need at least 2 target documents. Add more emails to the "
              "corpus directory (one .txt/.eml per email, or a .mbox).",
              file=sys.stderr)
        return 1

    model = StylometryModel().fit(
        target, impostor or None, calib_z=float(cfg.get("calib_z", 1.0)))
    save_model(model, cfg["model_path"], cfg["profile_path"])

    print("\nProfile:")
    print(json.dumps(model.summary(), indent=2))
    print(f"\nSaved model -> {cfg['model_path']}")
    if model.mode == "one_class":
        print("\nNote: one-class mode. 'likelihood' is a calibrated confidence, "
              "not a true probability. Add impostor emails to impostors/ for a "
              "genuine probability + AUC.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
