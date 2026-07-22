#!/usr/bin/env python3
"""Score a single piece of text against the trained profile.

Usage:
    python score.py --file email.txt
    python score.py --text "some email body ..."
    cat email.txt | python score.py
"""
from __future__ import annotations

import argparse
import json
import sys

from src.corpus_loader import clean_email_text
from src.store import load_config, load_model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--file")
    ap.add_argument("--text")
    ap.add_argument("--raw", action="store_true",
                    help="skip reply/signature cleaning")
    args = ap.parse_args()

    if args.file:
        text = open(args.file, encoding="utf-8", errors="replace").read()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    if not args.raw:
        text = clean_email_text(text)

    cfg = load_config(args.config)
    model = load_model(cfg["model_path"])
    result = model.score(text)
    result["threshold"] = cfg["threshold"]
    result["same_author"] = result["likelihood"] >= cfg["threshold"]
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
