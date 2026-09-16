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
   | `TELEGRAM_BOT_TOKEN` | *(secret)* | Optional — from [@BotFather](https://t.me/BotFather); another channel for the same two tools, independent of Pushover |
   | `TELEGRAM_CHAT_ID` | *(secret)* | Optional, pairs with `TELEGRAM_BOT_TOKEN` — your numeric chat ID, e.g. from [@userinfobot](https://t.me/userinfobot) |
   | `GRADIO_SERVER_NAME` | `0.0.0.0` | **Required** — Render routes traffic to this, not `127.0.0.1` |
   | `GRADIO_SERVER_PORT` | `10000` | **Required** — must match the port Render expects |

   `ANTHROPIC_API_KEY` is only needed if `AVATAR_PROVIDER` is ever switched to `anthropic` (the
   documented fallback in design §7) — not part of the default setup.

8. Deploy. Render builds and starts automatically; every push to `main` that touches
   `projects/08-linkedin-avatar/` triggers a redeploy (this is Render's default behaviour for a
   connected GitHub repo — no extra configuration needed).

### Health check endpoint

The app exposes `GET /health` (implemented in `app.py`'s `build_health_app`, mounted onto the
Gradio server via `app_kwargs={"app": ...}`). It returns:

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

Render's free tier sleeps after 15 minutes idle and takes 30–60s to wake on the next request.
Two mitigations are already built (design §9):

- The landing page (`site/index.html`) fires a `fetch()` at the app the moment it loads, so the
  instance is waking while the visitor reads — the "Start chatting" button usually lands on a warm
  app by the time it's clicked.
- A keep-warm GitHub Action (issue #131, not yet built) pings the app every ~14 minutes during
  waking hours to reduce how often it sleeps at all.

Directly hitting the Render URL cold (no landing page fetch first) still means a 30–60s wait on
the very first request. That's expected, not a bug.

### Taking it down in a hurry

Render dashboard → the service → **Suspend** (or delete the service entirely for something more
permanent). This stops it serving traffic immediately — the fastest option if something is
actively wrong (a jailbreak going viral, a cost spike despite the kill-switch, anything requiring
an instant "make it stop"). Resuming later is one click; the app rebuilds from the same repo
state.

If the problem is a specific leaked or compromised secret rather than "take the whole thing
down", rotating that one key (below) is faster than suspending and doesn't interrupt the app.

### Rotating a key

1. Generate a new key at the provider (DeepSeek, Pushover, Telegram's @BotFather).
2. Render dashboard → the service → Settings → Environment → update the variable's value.
3. Render redeploys automatically on an environment variable change.
4. Revoke the old key at the provider once the new deploy is confirmed live (see the smoke test
   below) — don't revoke first, or the current deploy starts failing before its replacement is
   confirmed working.

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
   asleep).
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
