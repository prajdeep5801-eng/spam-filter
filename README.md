# Stylometry Spam Filter for Gmail (courtsey : MAHARANI :) )

Flags (and optionally spam-files) incoming Gmail messages that appear to be
written by a **specific target person** — even when they send from an unfamiliar
address. It does this with **classic stylometry** (authorship attribution):
character n-grams + topic-independent writing-style statistics, compared against
a corpus of that person's known emails.

> Use case: someone you've blocked keeps emailing you from new addresses. Their
> *writing style* stays constant even when the address changes — this catches that.

---

## How it works

1. **Corpus** — you provide emails written by the target person (local
   `.txt` / `.eml` / `.mbox` files in `corpus/`).
2. **Train** — `train.py` builds a style profile: a character-n-gram + style
   centroid, calibrated with a leave-one-out pass over the corpus.
3. **Score** — for any new email, the model returns a `likelihood` (0–1) that
   the same person wrote it.
4. **Act** — `run_filter.py` polls your Gmail inbox, scores unread messages, and
   for anything above your `threshold` either applies a **label** (safe default)
   or moves it to **Spam**.

### Two scoring modes (chosen automatically)

| Mode | When | Output |
|------|------|--------|
| **one-class** | only `corpus/` provided | calibrated *confidence* (heuristic — no negatives to learn from) |
| **supervised** | you also fill `impostors/` with other people's emails | true *probability* + cross-validated AUC |

**Adding impostor emails is strongly recommended** — it turns the score into a
real probability and lets you measure accuracy. Put ~5–20 emails from other
people (yourself, newsletters, colleagues) in `impostors/`.

---

## Quick start (no Gmail needed — verify the model first)

The repo ships a demo corpus so you can see it work immediately:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train on the bundled example author + impostors
python train.py --config examples/config.yaml

# Score a held-out email in the target's style  -> high likelihood
python score.py --config examples/config.yaml --file examples/test_genuine.txt

# Score a formal impostor email                 -> low likelihood
python score.py --config examples/config.yaml --file examples/test_impostor.txt

# Run the tests
pytest tests/
```

---

## Using it on your own data

1. Put the **target person's** emails in `corpus/` (one `.txt` or `.eml` per
   email, or a single `.mbox`). More is better — aim for at least 5, ideally
   15+. In Gmail you can export a search to `.mbox` via
   [Google Takeout](https://takeout.google.com), or "Show original" → save a
   single message as `.eml`.
2. (Recommended) Put other people's emails in `impostors/`.
3. Train:
   ```bash
   python train.py            # uses config.yaml
   ```
   Inspect `data/profile.json`. In supervised mode, a `cv_auc` near 1.0 means the
   target's style is cleanly separable from the impostors.
4. Tune `threshold` in `config.yaml` by scoring a few known emails with
   `score.py` until genuine ones land above it and others below.

---

## Connecting to Gmail

### 1. Create Google OAuth credentials (one-time)

1. Go to the [Google Cloud Console](https://console.cloud.google.com/) and create
   a project (or pick an existing one).
2. **APIs & Services → Library →** search **Gmail API → Enable**.
3. **APIs & Services → OAuth consent screen:**
   - User type: **External** (for a normal @gmail.com account), then create.
   - Fill app name + your email for support/developer contact. Save.
   - **Test users → Add users →** add **your own Gmail address**. (In "Testing"
     status only listed test users can authorize — that's fine for personal use.)
4. **APIs & Services → Credentials → Create Credentials → OAuth client ID:**
   - Application type: **Desktop app**. Create.
   - **Download JSON**, rename it to **`credentials.json`**, and place it in this
     project's root folder (next to `run_filter.py`).

`credentials.json` and `token.json` are already git-ignored — never commit them.

### 2. First run (grant access)

```bash
python run_filter.py --once --dry-run
```

A browser window opens asking you to allow access. Approve it. A `token.json` is
saved so future runs are non-interactive. `--dry-run` scores your inbox and
prints decisions **without changing anything** — a safe first test.

> **Scope:** the app requests `gmail.modify`, which allows adding labels and
> moving messages to Spam. It **cannot permanently delete** anything.

> **Token expiry:** while the app is in "Testing" status, Google expires the
> refresh token after **7 days**. When that happens, delete `token.json` and run
> the command again to re-consent. (Publishing the app removes this but requires
> Google's verification for the restricted scope — not worth it for personal use.)

### 3. Go live

Edit `config.yaml`:

```yaml
threshold: 0.60          # raise to flag fewer, lower to flag more
gmail:
  action: "label"        # "label" = flag only (safe). "spam" = move to Spam.
  flag_label: "PossibleImpersonation"
```

Run a single real pass (applies the label / spam action):

```bash
python run_filter.py --once
```

Or run continuously (polls every `poll_interval_seconds`):

```bash
python run_filter.py
```

**Recommendation:** run with `action: "label"` for a week first. Review anything
tagged `PossibleImpersonation` in Gmail to check accuracy, then switch to
`action: "spam"` once you trust it.

---

## Running it automatically (macOS)

### Option A — cron (simple)

Score every 5 minutes:

```bash
crontab -e
```
```
*/5 * * * * cd /Users/rajdeeppaul/quant && .venv/bin/python run_filter.py --once >> filter.log 2>&1
```

### Option B — launchd (survives reboots, macOS-native)

Create `~/Library/LaunchAgents/com.user.stylometry-filter.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.user.stylometry-filter</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/rajdeeppaul/quant/.venv/bin/python</string>
    <string>/Users/rajdeeppaul/quant/run_filter.py</string>
    <string>--once</string>
  </array>
  <key>WorkingDirectory</key><string>/Users/rajdeeppaul/quant</string>
  <key>StartInterval</key><integer>300</integer>
  <key>StandardOutPath</key><string>/Users/rajdeeppaul/quant/filter.log</string>
  <key>StandardErrorPath</key><string>/Users/rajdeeppaul/quant/filter.log</string>
</dict>
</plist>
```
```bash
launchctl load ~/Library/LaunchAgents/com.user.stylometry-filter.plist
```

---

## Project layout

```
config.yaml          # all settings
train.py             # build the profile from corpus/
score.py             # score one email (CLI / stdin)
run_filter.py        # poll Gmail, score, label or spam
requirements.txt
corpus/              # <- target person's emails (you fill this)
impostors/           # <- other people's emails (optional, recommended)
data/                # trained model + profile.json
examples/            # bundled demo data + config
src/
  corpus_loader.py   # load & clean .txt/.eml/.mbox (strips quotes/signatures)
  features.py        # char n-grams + style-scalar features
  model.py           # one-class + supervised authorship model
  gmail_client.py    # Gmail OAuth + fetch/label/spam
  store.py           # config + model persistence
tests/
```

---

## Limitations & honest caveats

- **Stylometry needs enough text.** Very short emails ("ok, see you at 6") carry
  little style signal and score unreliably. Longer corpora and longer incoming
  emails work far better.
- **One-class `likelihood` is a calibrated confidence, not a probability.** With
  no negative examples there's no ground truth for "not the author." Add
  `impostors/` for a genuine, measurable probability.
- **A determined impersonator can mimic style**, and people's writing shifts by
  context (work vs. casual). Treat this as one signal, not proof. That's why the
  default action is a **label**, not deletion.
- **False positives are possible** — a friend who writes similarly could get
  flagged. Review labeled mail before enabling `action: "spam"`.
- This tool never deletes mail; Spam is recoverable for 30 days in Gmail.
```
