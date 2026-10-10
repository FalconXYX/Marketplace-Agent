# Marketplace Agent

Autonomous selling agent. Design: DESIGN.md. Python 3.12, uv, SQLite via SQLModel.

## Commands

| | |
|---|---|
| Tests | `uv run pytest` |
| CLI | `uv run agent --help` |
| Live tests (real services) | `MA_LIVE_TESTS=1 uv run pytest -m live` |

## Rules

1. **The model decides, code constrains.** Never encode selling strategy in code (no "drop price on day 7", no counter-offer formulas). Code only wakes the agent and rejects illegal actions. Guards return Reject with a reason the model can act on.
2. **Money is integer cents.** Field names end in `_cents`.
3. **Datetimes are timezone-aware UTC from an injected `Clock`.** Never call `datetime.now()` outside `core/clock.py`. Local time only for availability windows, human-facing text, and provider quota days (Gemini resets at midnight Pacific).
4. **`Commitment` is append-only.** Never update or delete a row; append RELEASED / UNDONE / COMPLETED events referencing the ACCEPTED row via `ref_id`.
5. **No network in tests.** Inject fakes (httpx `MockTransport`, fake clients, `ScriptedProvider`, `FakeMarketplace`). Anything touching a real service is `@pytest.mark.live` and skipped unless `MA_LIVE_TESTS=1`.
6. **Do not edit `pyproject.toml` or install packages.** If you need a dependency, say so in your final report.
7. **Contract files are frozen**: `core/models.py`, `core/clock.py`, `core/db.py`, `config.py`, `platforms/base.py`, `ai/base.py`, `tools/base.py`, `notify/base.py`, `agent/plan.py`. If one is insufficient, report it instead of changing it.
8. **Privacy.** Never put `pickup_address`, API keys, or passwords in prompts, logs, or exceptions. Strip EXIF from every photo on ingest.
9. **Writing style** for docs, docstrings, comments, and user-facing strings: concise, no em-dashes, tables over prose. Comment only what the code cannot say.

## Layout

| Path | Owns |
|---|---|
| `core/` | Models, clock, db, item services |
| `intake/` | Photo ingest, EXIF strip, resize |
| `ai/` | Providers, router, quota, pattern cache, drafting, prompts |
| `agent/` | Context assembly, reasoning loop, plan parsing |
| `tools/` | Tool implementations and registry |
| `guard/` | Invariants, commitment ledger, trust ladder |
| `memory/` | Scoped lessons |
| `platforms/` | `fake`, `kijiji`, `facebook` adapters |
| `runner/` | Ingestion, wake triggers, tick loop |
| `control/` | Human overrides (undo, approve, set price, mark sold) shared by CLI, web, Discord |
| `notify/` | Console, recording, Discord |
| `web/` | Photo intake form |
| `cli.py` | Typer entry point `agent` |
