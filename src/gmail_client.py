"""Thin Gmail API wrapper: OAuth, fetch message bodies, label / move to spam.

Auth uses the InstalledApp (desktop) OAuth flow. On first run a browser opens
for consent; the resulting token is cached to ``token.json`` and refreshed
automatically thereafter.

Scope: ``gmail.modify`` -- required to add/remove labels and to move a message
to Spam. It does NOT allow permanent deletion.
"""
from __future__ import annotations

import base64
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from .corpus_loader import clean_email_text

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


class GmailClient:
    def __init__(self, credentials_path: str = "credentials.json",
                 token_path: str = "token.json"):
        self.credentials_path = Path(credentials_path)
        self.token_path = Path(token_path)
        self.service = self._authenticate()
        self._label_cache: dict[str, str] = {}

    # ------------------------------------------------------------------ auth
    def _authenticate(self):
        creds = None
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(
                str(self.token_path), SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self.credentials_path.exists():
                    raise FileNotFoundError(
                        f"OAuth client file not found: {self.credentials_path}. "
                        "Download it from Google Cloud Console (see README).")
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), SCOPES)
                creds = flow.run_local_server(port=0)
            self.token_path.write_text(creds.to_json())
        return build("gmail", "v1", credentials=creds)

    # ---------------------------------------------------------------- labels
    def ensure_label(self, name: str) -> str:
        """Return the id of label ``name``, creating it if necessary."""
        if name in self._label_cache:
            return self._label_cache[name]
        existing = self.service.users().labels().list(userId="me").execute()
        for lbl in existing.get("labels", []):
            if lbl["name"].lower() == name.lower():
                self._label_cache[name] = lbl["id"]
                return lbl["id"]
        created = self.service.users().labels().create(
            userId="me",
            body={"name": name,
                  "labelListVisibility": "labelShow",
                  "messageListVisibility": "show"}).execute()
        self._label_cache[name] = created["id"]
        return created["id"]

    # -------------------------------------------------------------- messages
    def list_message_ids(self, query: str, max_results: int = 25) -> list[str]:
        resp = self.service.users().messages().list(
            userId="me", q=query, maxResults=max_results).execute()
        return [m["id"] for m in resp.get("messages", [])]

    def get_message(self, msg_id: str) -> dict:
        """Return {id, from, subject, body(cleaned), label_ids}."""
        msg = self.service.users().messages().get(
            userId="me", id=msg_id, format="full").execute()
        payload = msg.get("payload", {})
        headers = {h["name"].lower(): h["value"]
                   for h in payload.get("headers", [])}
        body = self._extract_body(payload)
        return {
            "id": msg_id,
            "from": headers.get("from", ""),
            "subject": headers.get("subject", ""),
            "body": clean_email_text(body),
            "label_ids": msg.get("labelIds", []),
        }

    @staticmethod
    def _decode(data: str) -> str:
        return base64.urlsafe_b64decode(data.encode("utf-8")).decode(
            "utf-8", errors="replace")

    def _extract_body(self, payload: dict) -> str:
        mime = payload.get("mimeType", "")
        body_data = payload.get("body", {}).get("data")
        if mime == "text/plain" and body_data:
            return self._decode(body_data)
        # Recurse into parts, preferring text/plain.
        parts = payload.get("parts", [])
        for part in parts:
            if part.get("mimeType") == "text/plain":
                data = part.get("body", {}).get("data")
                if data:
                    return self._decode(data)
        for part in parts:  # fall back to nested multiparts
            nested = self._extract_body(part)
            if nested:
                return nested
        # Last resort: strip nothing, return html-as-text if that's all there is.
        if mime == "text/html" and body_data:
            return self._decode(body_data)
        return ""

    # --------------------------------------------------------------- actions
    def modify(self, msg_id: str, add: list[str] | None = None,
               remove: list[str] | None = None) -> None:
        self.service.users().messages().modify(
            userId="me", id=msg_id,
            body={"addLabelIds": add or [], "removeLabelIds": remove or []}
        ).execute()

    def add_label(self, msg_id: str, label_id: str) -> None:
        self.modify(msg_id, add=[label_id])

    def move_to_spam(self, msg_id: str) -> None:
        # SPAM/INBOX are system labels usable directly by id.
        self.modify(msg_id, add=["SPAM"], remove=["INBOX"])
