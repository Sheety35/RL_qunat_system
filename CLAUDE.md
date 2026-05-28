# Claude Code — Project Instructions
# E:\Shashank\stocks

## MANDATORY: Change Logging

**Every session MUST append a log entry to `CHANGES.log` in this directory.**

### When to log
- Append an entry when ANY file in this project is created, modified, or deleted.
- Log at the end of the session (or after each logical group of changes).
- Never skip logging, even for single-line edits.

### How to log
Append to `E:\Shashank\stocks\CHANGES.log` using this exact format:

```
--------------------------------------------------------------------------------
SESSION YYYY-MM-DD — <one-line summary of what the session accomplished>
--------------------------------------------------------------------------------

[YYYY-MM-DD] filename.py :: <change title> — <one or two sentences explaining
  what was changed and WHY (the bug it fixed or the reason for the change).
  Indent continuation lines by 2 spaces.>

[YYYY-MM-DD] filename.py :: <next change> — ...
```

Use today's date from the `currentDate` context variable.
Group all changes from the same session under one `SESSION` header.
If a previous session entry already exists for today, append under it (no new header).

### What to include in each entry
- The filename and a short title (after `::`)
- What was changed — be specific (old value → new value, old formula → new formula)
- Why it was changed — the bug it fixed, the symptom it addressed, or the reason

### What NOT to log
- Exploratory reads, searches, or questions (only log actual file changes)
- Changes to `CHANGES.log` itself or `CLAUDE.md`

---

## Project Context

**Goal:** SAC reinforcement learning agent trading NIFTY 50 + NIFTY BANK futures.

**Stack:**
- Python 3.11, stable-baselines3 SAC, gymnasium, pandas, pyarrow
- Always run Python via `./venv/Scripts/python.exe` (never system Python)
- Features in `features/` as parquet files (built by Phases 0–2)
- Models saved to `models/sac_multi/`

**Key files:**
- `sac_trainer.py` — gymnasium env + SAC training loop
- `guardrails.py` — hard risk rules (HG1–HG11), never bypass
- `feature_engineering.py` — Phase 2 feature pipeline
- `train.py` — entrypoint that chains all phases

**Train/Val/Test split (walk-forward, never mix):**
- Train:    2019-01-01 → 2022-12-31
- Validate: 2023-01-01 → 2023-12-31
- Test:     2024-01-01 → 2024-12-31  ← touch only once at the very end

**Hard rules (never override in code):**
- HG1: No entries before 09:55 or at/after 15:00
- HG2: Halt trading if daily loss > 3% of capital
- HG3: Single trade max loss capped at 2% of capital
- HG8: Force-exit all positions at 15:00
- Lots are fixed at 1 per instrument during training (₹50k capital)
