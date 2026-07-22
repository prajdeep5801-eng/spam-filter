#!/usr/bin/env python3
"""Poll Gmail, score unread inbox messages, and act on likely impersonations.

Usage:
    python run_filter.py --once      # single pass, then exit (good for cron)
    python run_filter.py             # loop forever, sleeping between polls
    python run_filter.py --once --dry-run   # score + print, take NO action

Safety:
  * Default action is "label" (config.yaml) -- nothing is moved to Spam unless
    you set action: "spam" explicitly.
  * --dry-run overrides everything: it only prints decisions.
  * Every scored message gets the processed_label so it is never reprocessed.
"""
from __future__ import annotations

import argparse
import time

from src.gmail_client import GmailClient
from src.store import load_config, load_model


def process_once(client: GmailClient, model, cfg: dict, dry_run: bool) -> int:
    g = cfg["gmail"]
    processed_id = client.ensure_label(g["processed_label"])
    flag_id = client.ensure_label(g["flag_label"])
    threshold = float(cfg["threshold"])
    action = g.get("action", "label")

    # Exclude already-processed messages from the query.
    query = f"{g['poll_query']} -label:{g['processed_label']}"
    ids = client.list_message_ids(query, g.get("max_messages_per_poll", 25))
    if not ids:
        print("[poll] no new messages")
        return 0

    acted = 0
    for msg_id in ids:
        msg = client.get_message(msg_id)
        if not msg["body"].strip():
            client.add_label(msg_id, processed_id)
            continue

        result = model.score(msg["body"])
        like = result["likelihood"]
        flagged = like >= threshold
        tag = "FLAG" if flagged else "ok  "
        print(f"[{tag}] like={like:.3f} sim={result['similarity']:.3f} "
              f"from={msg['from'][:45]!r} subj={msg['subject'][:40]!r}")

        if dry_run:
            continue

        if flagged:
            client.add_label(msg_id, flag_id)
            if action == "spam":
                client.move_to_spam(msg_id)
            acted += 1
        client.add_label(msg_id, processed_id)

    print(f"[poll] processed {len(ids)} message(s), flagged {acted}")
    return acted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--once", action="store_true", help="single pass then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="score and print only; take no Gmail action")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = load_model(cfg["model_path"])
    g = cfg["gmail"]
    client = GmailClient(g["credentials_path"], g["token_path"])

    print(f"Mode={model.mode} action={g.get('action')} "
          f"threshold={cfg['threshold']} dry_run={args.dry_run}")

    if args.once:
        process_once(client, model, cfg, args.dry_run)
        return 0

    interval = int(g.get("poll_interval_seconds", 120))
    print(f"Polling every {interval}s. Ctrl-C to stop.")
    while True:
        try:
            process_once(client, model, cfg, args.dry_run)
        except Exception as exc:  # keep the daemon alive across transient errors
            print(f"[error] {exc}")
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
