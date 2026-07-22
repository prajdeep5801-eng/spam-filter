"""Load and clean email text from local files (.txt / .eml / .mbox).

Each returned item is one "document" = the body text one person actually wrote,
with quoted replies and signatures stripped so the stylometry model sees the
author's own words rather than material they merely forwarded/quoted.
"""
from __future__ import annotations

import email
import mailbox
import re
from email import policy
from pathlib import Path

# Lines / blocks that mark the start of quoted material or a signature.
_REPLY_MARKERS = [
    re.compile(r"^On .*wrote:\s*$"),                 # "On Mon, ... John wrote:"
    re.compile(r"^-{2,}\s*Original Message\s*-{2,}", re.I),
    re.compile(r"^_{5,}\s*$"),
    re.compile(r"^From:\s.+", re.I),                  # forwarded header block
    re.compile(r"^Sent:\s.+", re.I),
    re.compile(r"^>?\s*Begin forwarded message:", re.I),
]
_SIGNATURE_MARKER = re.compile(r"^--\s*$")           # standard "-- " sig delimiter


def clean_email_text(raw: str) -> str:
    """Strip quoted replies, forwarded headers and signatures from a body."""
    lines = raw.replace("\r\n", "\n").split("\n")
    kept: list[str] = []
    for line in lines:
        stripped = line.rstrip()
        # Cut everything from the first reply/forward marker onward.
        if any(m.match(stripped) for m in _REPLY_MARKERS):
            break
        # Signature delimiter: drop the rest.
        if _SIGNATURE_MARKER.match(stripped):
            break
        # Drop quoted lines ("> ...").
        if stripped.lstrip().startswith(">"):
            continue
        kept.append(line)
    text = "\n".join(kept)
    # Collapse runs of blank lines and trailing whitespace.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _extract_body(msg: email.message.Message) -> str:
    """Return the best-effort plain-text body of an email.Message."""
    if msg.is_multipart():
        # Prefer the first text/plain part that is not an attachment.
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and \
                    "attachment" not in str(part.get("Content-Disposition", "")):
                try:
                    return part.get_content()
                except Exception:
                    payload = part.get_payload(decode=True) or b""
                    return payload.decode(part.get_content_charset() or "utf-8",
                                          errors="replace")
        return ""
    try:
        return msg.get_content()
    except Exception:
        payload = msg.get_payload(decode=True) or b""
        return payload.decode(msg.get_content_charset() or "utf-8", errors="replace")


def _load_eml(path: Path) -> list[str]:
    with path.open("rb") as fh:
        msg = email.message_from_binary_file(fh, policy=policy.default)
    return [_extract_body(msg)]


def _load_mbox(path: Path) -> list[str]:
    box = mailbox.mbox(str(path))
    bodies: list[str] = []
    for msg in box:
        bodies.append(_extract_body(msg))
    return bodies


def _load_txt(path: Path) -> list[str]:
    return [path.read_text(encoding="utf-8", errors="replace")]


def load_documents(directory: str | Path, *, min_chars: int = 40) -> list[str]:
    """Load all supported files under ``directory`` into cleaned documents.

    Files with the same extension are treated per the format:
      - .txt  -> one document per file
      - .eml  -> one document per file
      - .mbox -> one document per message

    Documents shorter than ``min_chars`` after cleaning are dropped, since very
    short texts carry too little style signal to be useful.
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    docs: list[str] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        try:
            if ext == ".txt":
                raw_docs = _load_txt(path)
            elif ext == ".eml":
                raw_docs = _load_eml(path)
            elif ext == ".mbox":
                raw_docs = _load_mbox(path)
            else:
                continue
        except Exception as exc:  # pragma: no cover - robustness for odd files
            print(f"[corpus] skipping {path}: {exc}")
            continue

        for raw in raw_docs:
            cleaned = clean_email_text(raw)
            if len(cleaned) >= min_chars:
                docs.append(cleaned)
    return docs
