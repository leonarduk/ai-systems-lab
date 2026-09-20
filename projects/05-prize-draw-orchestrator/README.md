# Prize Draw Orchestrator

An orchestrating client that schedules runs, calls a prize-draw MCP server's
tools (`search_draws`, `parse_entry_page`, `submit_entry`, `check_log` — see
[issue #22](https://github.com/leonarduk/ai-systems-lab/issues/22)), and uses
a configurable LLM backend to parse competition pages, filter them against
your criteria, reason about eligibility, and answer simple free-text
tie-breakers.

LLM-backend selection lives entirely in this client, not in the MCP server:
swapping between Ollama/DeepSeek/Claude requires no changes to the server's
scraping/entry tool implementations.

## Status: integrates against a documented stub interface

Issue #22 (the MCP server) had not been merged at the time this was written.
Rather than depend on unmerged code, this client is built against the tool
contract described in that issue:

| Tool | Arguments | Returns |
|---|---|---|
| `search_draws` | `{"criteria": {...}}` | `{"draws": [{"draw_id", "url", "title", ...}, ...]}` |
| `parse_entry_page` | `{"draw_id", "url"}` | `{"content": "<raw HTML/text>"}` |
| `submit_entry` | `{"draw_id", "fields", "confirm_personal_data", "dry_run"}` | `{"status": "submitted" \| "dry_run" \| ...}` |
| `check_log` | `{"query": {"draw_id"}}` or `{"record": {...}}` | `{"seen": bool, "entries": [...]}` or `{"ok": true}` |

`mcp_client.py` defines this as an `MCPToolClient` protocol with one generic
`call_tool(name, arguments)` method — the same shape as the official MCP
Python SDK's `ClientSession.call_tool`. `StdioMCPToolClient` is a real
implementation using that SDK (see `requirements.txt`); it has not been
exercised against a live server since none existed yet. Once issue #22's
server is available, set `MCP_SERVER_COMMAND` / `MCP_SERVER_ARGS` to launch
it and this client should work unmodified. Most tests mock the protocol via
`tests/fakes.py::FakeMCPToolClient`, so they never depend on a live server;
`tests/test_stdio_mcp_tool_client.py` additionally drives the real client
against a mock server subprocess.

Every call is bounded: `StdioMCPToolClient` takes a `connect_timeout`
(default 10s, for the MCP handshake) and a `call_timeout` (default 60s, for
the tool itself, which fetches and parses pages). Both raise `MCPToolError`
when they expire, as do handshake and startup failures. Without the connect
timeout a server that writes unparseable output and holds its pipe open
blocks the orchestrator forever — no error, no log line, the poll just stops.

## How it works

1. `search_draws` returns candidate draws matching your configured criteria
   (prize type/value, entry method, region/eligibility, closing date).
2. For each candidate, `check_log` is queried first to skip anything already
   seen/entered (duplicate avoidance).
3. `parse_entry_page` fetches the draw's raw page content, which is handed to
   the configured LLM to normalize into structured fields (prize, closing
   date, entry requirements, entry URL) and classify against your criteria.
4. The LLM also flags draws that require a purchase, have a complex/creative
   tie-breaker, or otherwise need a human — those are **never** auto-entered,
   only recorded under "needs review".
5. Eligible draws are submitted via `submit_entry`, respecting dry-run mode,
   and the result is recorded via `check_log`.
6. A summary (found / entered / needs review / duplicates skipped / errors)
   is logged after every run — see `RunSummary.as_text()` in `orchestrator.py`.

## Configuring the LLM backend

Set `LLM_PROVIDER` in your environment or `.env` file (see `.env.example`):

| Provider | `LLM_PROVIDER` | Config | Data leaves your machine? |
|---|---|---|---|
| Ollama (default) | `ollama` | `OLLAMA_HOST`, `OLLAMA_MODEL` | No |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL` | **Yes** |
| Claude | `claude` | `ANTHROPIC_API_KEY`, `CLAUDE_MODEL` | **Yes** |

**Ollama is the default and is always used unless `LLM_PROVIDER` is
explicitly set to something else.** DeepSeek and Claude are opt-in only.

**Before enabling DeepSeek or Claude**: understand that every competition
page's content, and any personal data embedded in it (e.g. a tie-breaker
question referencing your details), is sent as part of the prompt to that
provider's hosted API (`api.deepseek.com` or `api.anthropic.com`). Only
enable one of these after making an informed decision to send that data
off your machine — the CLI logs a warning on every run when a non-Ollama
provider is configured, as a reminder.

Store `DEEPSEEK_API_KEY` / `ANTHROPIC_API_KEY` in your local `.env` file
(gitignored, same pattern as `projects/04-gmail-inbox-labeler`) or your
shell/CI secrets manager — never commit them, and never print them to logs.

## Safety gates

- **`DRY_RUN=true` by default.** Eligible draws are logged as "would enter"
  but `submit_entry` is always called with `dry_run: true` until you
  explicitly set `DRY_RUN=false` or pass `--live` on the CLI.
- **`CONFIRM_PERSONAL_DATA=false` by default.** If a draw's entry
  requirements plausibly involve personal/financial data (address, phone,
  card/bank/payment details, date of birth, postcode, etc.), the draw is
  routed to "needs review" instead of being auto-entered, even if otherwise
  eligible — regardless of dry-run mode. Set `CONFIRM_PERSONAL_DATA=true`
  only after reviewing what will be sent.
- Draws requiring a purchase, or with a complex/creative free-text
  tie-breaker, are always routed to "needs review" for a human, never
  auto-entered.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env: point MCP_SERVER_COMMAND/MCP_SERVER_ARGS at the issue #22 server,
# choose an LLM_PROVIDER, and review the safety gates above.
```

Pull an Ollama model if using the default backend:

```bash
ollama pull llama3
ollama serve
```

## Usage

Recommended: review a single dry-run pass before scheduling anything:

```bash
python cli.py --once --dry-run
```

Run continuously on the configured interval (`RUN_INTERVAL_MINUTES`, still
in dry-run unless `DRY_RUN=false`/`--live` is set):

```bash
python cli.py
```

Enable full automation only once you've reviewed dry-run output and are
comfortable with the configured criteria and safety gates:

```bash
python cli.py --live
```

| Flag | Description |
|---|---|
| `--once` | Run a single pass and exit (default: loop forever on `RUN_INTERVAL_MINUTES`) |
| `--dry-run` | Force dry-run for this invocation |
| `--live` | Disable dry-run; actually submit eligible entries |
| `--verbose` | Enable debug logging |

## Criteria

Draw-matching criteria (prize type/value, entry method, region, max days to
closing) are loaded from the JSON file at `CRITERIA_CONFIG_PATH`. If unset,
built-in defaults are used — `criteria.example.json` is a starting point, not
a fallback file, so copy it, edit it, and set `CRITERIA_CONFIG_PATH` to point
at your copy for it to actually take effect.

## Tests

```bash
python -m pytest tests/ -v
```

All 4 MCP tool calls and all 3 LLM providers are mocked/faked in tests (see
`tests/fakes.py`) — no network access or live server is required.
