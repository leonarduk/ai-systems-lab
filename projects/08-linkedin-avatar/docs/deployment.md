# Deployment

How the LinkedIn Avatar is actually hosted, and how to reproduce or fix it without needing to
remember anything else about this project. See [`design.md`](./design.md) §9 for the reasoning
behind these choices — this doc is the "just tell me what to click" version.

Two things are live: the **app** (Render, does the actual chatting) and the **landing page**
(GitHub Pages, what LinkedIn actually links to).

---

## 1. The app — Render

Render's free web-service tier. One-time setup:

1. [dashboard.render.com](https://dashboard.render.com) → **New** → **Web Service** → connect the
   `leonarduk/ai-systems-lab` GitHub repo.
2. **Root directory**: `projects/08-linkedin-avatar`
3. **Build command**: `pip install -r requirements.txt`
4. **Start command**: `python app.py`
5. **Instance type**: Free
6. **Health check path**: `/health` (Render dashboard → Settings → Health Checks). The app
   exposes a lightweight `GET /health` endpoint that returns HTTP 200 with
   `{"status": "ok"}`. It has no external dependencies and no side effects, so it's safe for
   Render to poll frequently — see "Health check endpoint" below.
7. **Environment variables** — set every one of these in Render's dashboard (Settings →
   Environment), never in the repo:

   > ⚠️ **Secrets warning.** Every value marked *(secret)* below — in particular
   > `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `DEEPSEEK_API_KEY`, `PUSHOVER_USER` and
   > `PUSHOVER_TOKEN` — must **never** be committed to the repository, pasted into issues, or
   > included in logs. Set them only in Render's dashboard (or a local `.env` that is
   > `.gitignore`d). If a secret is ever committed, treat it as compromised and rotate it
   > immediately (see §1.5 Emergency shutdown).

   | Variable | Value | Notes |
   |---|---|---|
   | `DEEPSEEK_API_KEY` | *(secret)* | Required. From [platform.deepseek.com](https://platform.deepseek.com) |
   | `AVATAR_PROVIDER` | `deepseek` | Optional — this is already the default |
   | `AVATAR_MODEL` | `deepseek-v4-flash` | Optional — this is already the default. `deepseek-v4-pro` for higher quality at ~3× the cost |
   | `AVATAR_DAILY_BUDGET_USD` | `2.00` | Optional — raise this if the kill-switch trips on genuine traffic, not to "fix" an error |
   | `AVATAR_MAX_CONTEXT_TOKENS` | `40000` | Optional |
   | `AVATAR_MAX_INPUT_CHARS` | `1500` | Optional |
   | `AVATAR_SESSION_RATE_LIMIT` | `20/hour` | Optional |
   | `AVATAR_IP_RATE_LIMIT` | `40/day` | Optional |
   | `PUSHOVER_USER` | *(secret)* | Optional — without it, `record_contact`/`record_unknown_question` log instead of notifying |
   | `PUSHOVER_TOKEN` | *(secret)* | Optional, pairs with `PUSHOVER_USER` |
   | `TELEGRAM_BOT_TOKEN` | *(secret)* | Optional — from [@BotFather](https://t.me/BotFather); another channel for the same two tools, independent of Pushover. **Never commit this value.** |
   | `TELEGRAM_CHAT_ID` | *(secret)* | Optional, pairs with `TELEGRAM_BOT_TOKEN` — your numeric chat ID, e.g. from [@userinfobot](https://t.me/userinfobot) |
   | `GRADIO_SERVER_NAME` | `0.0.0.0` | **Required** — Render routes traffic to this, not `127.0.0.1` |
   | `GRADIO_SERVER_PORT` | `10000` | **Required** — must match the port Render expects |

   `ANTHROPIC_API_KEY` is only needed if `AVATAR_PROVIDER` is ever switched to `anthropic` (the
   documented fallback in design §7) — not part of the default setup.

8. Deploy. Render builds and starts automatically; every push to `main` that touches
   `projects/08-linkedin-avatar/` triggers a redeploy (this is Render's default behaviour for a
   connected GitHub repo — no extra configuration needed).

### Health check endpoint

The app exposes `GET /health` (implemented in `app.py`'s `build_health_app`). The health app and
the Gradio Blocks demo are combined with `gr.mount_gradio_app(health_app, demo, path="/", ...)` in
`build_app()`, then served with `uvicorn.run(...)` — not via `demo.launch(app_kwargs=...)`, which
silently drops a second FastAPI app passed that way. It returns:

```
HTTP/1.1 200 OK
Content-Type: application/json

{"status": "ok"}
```

Configure Render's health check path to `/health` (Settings → Health Checks). Success criteria:
HTTP 200 with the JSON body above. The endpoint is intentionally trivial — no logging, no
external calls, no dependency on DeepSeek/Pushover/Telegram — so it can't be the cause of a
false "unhealthy" flag during a cold start or a provider outage. If Render reports the service
unhealthy, the problem is elsewhere (see "Checking logs" below), not the health check itself.

### Checking logs

Render dashboard → the service → **Logs** tab. Live-tails stdout/stderr. This is where a crash
loop from a missing environment variable shows up — the app fails fast on `DEEPSEEK_API_KEY`
being unset (`avatar/llm.py`'s `_build_client` raises `KeyError` immediately), which reads as a
repeated restart in the Render UI. If that happens, it's almost always a missing or mistyped env
var — check the table above against what's actually set before looking anywhere else.

### Cold-start behaviour

A **cold start** is any time the process is not already running and has to be brought up from
scratch. It is triggered by:

- **Idle spin-down** — Render's free tier sleeps the service after ~15 minutes with no traffic.
- **Redeploy** — every push to `main` that touches `projects/08-linkedin-avatar/` rebuilds and
  restarts the service.
- **Manual restart / instance recycle** — e.g. after an env-var change, or if Render recycles the
  instance.

What happens during a cold start:

- **Build phase (redeploy only).** Render runs `pip install -r requirements.txt` before the
  process starts. Duration depends on Render's cache state and PyPI latency — **TBD, to be
  measured** on a clean build. Idle spin-down does *not* re-run the build; it only restarts the
  process.
- **Process start.** `python app.py` imports the Gradio app and the `avatar/` package. There is
  no model loading on our side (the LLM is a remote API call), so this is fast — but the exact
  wall-clock time is **TBD, to be measured**.
- **Warm-up window.** Until the Gradio server is listening on `0.0.0.0:10000`, requests to the
  Render URL will fail — typically a connection error, a 502, or a hanging request that
  eventually times out. This is expected, not an outage. The landing page mitigates it by firing
  a `fetch()` at the app the moment it loads (see below), so the instance is usually warm by the
  time a visitor clicks "Start chatting".
- **Health checks.** Render's own health check will fail during the warm-up window; the service
  is only marked healthy once the port is accepting connections. A short burst of failed checks
  during a cold start is normal.

Configuration that affects cold-start time:

- **Instance type.** Currently **Free**. Upgrading to a paid instance type removes the idle
  spin-down entirely (no more cold starts from idleness) and gives more CPU during the build.
- **Disk persistence.** Not used — the app is stateless and holds no local state across restarts,
  so there is nothing to rehydrate on cold start. If persistent disk is ever added, factor its
  mount/attach time into the cold-start budget.
- **Build cache.** Render caches `pip` downloads between builds; a cache miss (e.g. after a
  `requirements.txt` change) makes the build phase noticeably longer.

**Rule of thumb for operators:** if the service was idle for >15 minutes, expect a cold start on
the next request. Allow up to ~60s before treating it as a real failure. Only escalate (see
Emergency shutdown) if the service is *still* not responding after that, or if the logs show a
crash loop rather than a normal startup.

### Taking it down in a hurry

Render dashboard → the service → **Suspend** (or delete the service entirely for something more
permanent). This stops it serving traffic immediately — the fastest option if something is
actively wrong (a jailbreak going viral, a cost spike despite the kill-switch, anything requiring
an instant "make it stop"). Resuming later is one click; the app rebuilds from the same repo
state.

If the problem is a specific leaked or compromised secret rather than "take the whole thing
down", rotating that one key (below) is faster than suspending and doesn't interrupt the app.
For a suspected secret leak, an active abuse incident, or anything needing credentials revoked
as well as traffic stopped, use the full **Emergency shutdown** procedure below instead.

### Rotating a key

1. Generate a new key at the provider (DeepSeek, Pushover, Telegram's @BotFather).
2. Render dashboard → the service → Settings → Environment → update the variable's value.
3. Render redeploys automatically on an environment variable change.
4. Revoke the old key at the provider once the new deploy is confirmed live (see the smoke test
   below) — don't revoke first, or the current deploy starts failing before its replacement is
   confirmed working.

### Emergency shutdown

Use this when the service must stop **now** — a suspected secret leak, an active abuse incident,
an uncontrolled cost spike, or anything else where "make it stop" beats "fix it in place". The
steps are ordered so that traffic stops first, then credentials are neutralised, then you confirm
the shutdown actually took effect.

1. **Stop the Render web service.**
   - **Dashboard (preferred):** [dashboard.render.com](https://dashboard.render.com) → the
     `08-linkedin-avatar` service → **Suspend**. This halts traffic immediately and is reversible
     with one click. Use **Delete** only if the service should not come back — deleting also
     destroys the service's configuration (environment variables, build settings), so a later
     restore means recreating the service and re-entering every env var from scratch, not just
     un-suspending it.
   - **CLI (if you have it configured):** `render services suspend <service-id>` (or
     `render services delete <service-id>` for permanent removal). Confirm the service ID in the
     dashboard first — do not guess it.
   - Do **not** rely on the daily budget kill-switch for an emergency: it is a cost guard, not a
     shutdown control, and it does not stop the process or revoke credentials.

2. **Revoke compromised credentials.** Do this even if the service is suspended — a leaked token
   is usable from anywhere, not just from Render.
   - **Telegram:** open [@BotFather](https://t.me/BotFather) → `/mybots` → select the bot →
     **API Token** → **Revoke current token**. This invalidates `TELEGRAM_BOT_TOKEN` everywhere.
     Generate a replacement only when you are ready to redeploy.
   - **Pushover:** log in at [pushover.net](https://pushover.net) → your application → **Regenerate
     Application Token** (invalidates `PUSHOVER_TOKEN`). If `PUSHOVER_USER` may also be exposed,
     rotate the user key from the same page.
   - **DeepSeek:** log in at [platform.deepseek.com](https://platform.deepseek.com) → API keys →
     revoke the exposed key and issue a new one (`DEEPSEEK_API_KEY`).
   - **Anthropic** (only if `AVATAR_PROVIDER=anthropic`): revoke the key in the Anthropic console
     (`ANTHROPIC_API_KEY`).
   - Update the corresponding values in Render's Environment settings **only after** the new
     credentials are in hand, and only if the service is being brought back up.

3. **Verify the service is actually down.**
   - Render dashboard → the service → **Logs**: confirm the process has stopped (no new lines)
     and that the service status shows *Suspended* (or *Deleted*).
   - From a terminal, hit the service URL directly, e.g.
     `curl -i https://<service-name>.onrender.com/` — expect a connection failure or a Render
     "service unavailable" response, **not** a Gradio page. A 200 with the chat UI means it is
     still up; go back to step 1.
   - If Telegram or Pushover were configured, confirm the revoked tokens no longer work. For
     Telegram, check the **JSON body**, not just the HTTP status — the Bot API can report an
     invalid token with `"ok": false` even when the transport-level status looks like success:
     `curl -s https://api.telegram.org/bot<old-token>/getMe` should return a body containing
     `"ok":false` (e.g. `{"ok":false,"error_code":401,"description":"Unauthorized"}`). A body with
     `"ok":true` means the token is still live; go back to step 2.

4. **Document the incident.** Record what happened, when the service was suspended, which
   credentials were revoked and when, and who was notified. Link the incident note from the
   relevant issue or PR so the next person has the timeline. If a runbook exists elsewhere in the
   repo, link it here; otherwise this section is the runbook.

---

## 2. The landing page — GitHub Pages

`projects/08-linkedin-avatar/site/index.html` is a single self-contained static file — no build
step, nothing to deploy beyond publishing it. Two ways to serve it; pick whichever GitHub Pages
setup is already used elsewhere in this account, so there's only one publishing pattern to
remember:

- **Repo-root Pages** (`leonarduk.github.io/ai-systems-lab/...`) — if this repo's GitHub Pages is
  (or becomes) enabled for the whole repo from the `main` branch, the page is reachable at
  `https://leonarduk.github.io/ai-systems-lab/projects/08-linkedin-avatar/site/`. This is the URL
  already assumed in `site/index.html`'s Open Graph/Twitter meta tags — if a different Pages setup
  is used instead, update those tags to match (search that file for `leonarduk.github.io`).
- **A dedicated Pages source** (e.g. a `gh-pages` branch or a `docs/` folder) — more setup, but
  keeps the public landing page's URL independent of the repo's internal layout. Only worth it if
  the URL needs to survive a restructuring of `projects/08-linkedin-avatar/`.

Either way: GitHub repo → **Settings** → **Pages** → choose the source. No build step, no secrets,
nothing to rotate — it's a static file.

---

## 3. Verifying a deploy actually works

After any deploy (first time or after a change), a quick end-to-end smoke test — this is what
issue #130's "success looks like" checklist is asking for, and it's cheap enough to run after every
redeploy:

1. Open the Render app URL directly. It should load the chat UI (allow up to 60s if it was
   asleep — see Cold-start behaviour above).
2. Hit `https://<app-url>/health` and confirm it returns HTTP 200 with `{"status": "ok"}`. This
   is the same endpoint Render polls — if it fails, Render will mark the service unhealthy.
3. Ask it a real question — e.g. "Tell me about issue-worm." — and confirm it answers correctly,
   grounded in the actual knowledge files, not a generic non-answer.
4. Say something that should trigger `record_contact` (e.g. "I'd like to talk to him about a
   role, here's my email: test@example.com") and confirm a notification actually arrives on
   whichever channel(s) are configured (Pushover, Telegram, or both — see `tools._notify` in
   `avatar/tools.py`), or check the Render logs for the "not configured; logging instead" line for
   any channel that isn't set up yet.
5. Open the landing page URL, confirm it loads instantly, and click "Start chatting" through to a
   warm app.
6. Paste the landing page URL into a LinkedIn post's preview (or a tool like
   [opengraph.xyz](https://www.opengraph.xyz/)) and confirm the title, description and image
   render as a proper card, not a bare link.

If any of these fail, check Render's logs first (§1) — a missing environment variable is the most
common cause and shows up there immediately.
