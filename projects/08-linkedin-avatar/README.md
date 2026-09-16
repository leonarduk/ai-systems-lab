# LinkedIn Avatar

An AI twin that talks to visitors about my career and my GitHub projects — one link, dropped into
the Featured section of my LinkedIn profile.

**Status:** deployed. See [`docs/design.md`](./docs/design.md) and the issues in
[milestone 4 — LinkedIn Avatar M1](https://github.com/leonarduk/ai-systems-lab/milestone/4).

**Live URLs**:
- App: [ai-systems-lab-s8gy.onrender.com](https://ai-systems-lab-s8gy.onrender.com)
- Landing page: [leonarduk.github.io/ai-systems-lab/projects/08-linkedin-avatar/site/](https://leonarduk.github.io/ai-systems-lab/projects/08-linkedin-avatar/site/)

See [`docs/deployment.md`](./docs/deployment.md) for exactly how to deploy, rotate a key, check
logs, or take it down in a hurry.

## What it does

A visitor lands on a static page from LinkedIn, clicks through to a chat, and can ask about my
background ("what's his Java experience?"), about anything in my GitHub ("tell me about
issue-worm"), or ask to be put in touch — which sends me a push notification with their email. When
it doesn't know something, it says so and records the question rather than inventing an answer.

## How it works

Its knowledge is three plain-text files baked into a cached system prompt: a hand-written
`summary.txt` for voice, a `profile.md` derived from my LinkedIn PDF export with contact details
redacted, and a `github.json` snapshot of my public repos refreshed nightly by a GitHub Action — so
a new repo anywhere in this org shows up in what the twin knows within a day, no redeploy needed.
There are three tools — record a contact, record an unanswerable question, look up one project in
detail — and no database.

Because it is a public endpoint with an API key behind it, the guardrails are part of the design and
not an afterthought: input length cap, per-session and per-IP rate limits, and a daily spend
kill-switch.

**Tech stack:** Python, Gradio, DeepSeek API (OpenAI-compatible, automatic prompt caching + tool use),
GitHub REST API, Pushover/Telegram, Render, GitHub Pages.

## Knowledge file formats

`knowledge/summary.txt` is a single plain-text paragraph, first person, under ~200 words — no
front matter, no headings. It's the only file that sets personality and sits in every request's
cached prefix, so it's kept short and hand-edited.

`knowledge/projects.md` is plain Markdown: one `## repo-name` heading per repo (matching its GitHub
name exactly), followed by 1–3 sentences of first-person framing. The snapshot builder
(`build/build_github_snapshot.py`) parses it into per-repo notes and prefers a `projects.md` entry
over the repo's own README excerpt wherever one exists — a repo with no heading here just falls back
to its README. Headings are matched case-sensitively against the repo name; keep them in the same
order as the root README where practical, but order isn't semantically meaningful to the parser.

## Smoke test

`scripts/smoke_test.py` verifies a deployed instance end to end: required env vars are present,
the root URL returns 200, the chat endpoint answers a real question, and the contact-capture flow
works. Standard library only — no extra dependencies.

```bash
# From projects/08-linkedin-avatar/
python scripts/smoke_test.py --base-url https://ai-systems-lab-s8gy.onrender.com --dry-run
```

`--dry-run` validates the contact-capture request without sending a real Pushover notification.
Drop the flag (with `PUSHOVER_USER`/`PUSHOVER_TOKEN` set) to verify delivery end to end. Exits
non-zero with a clear message on the first failed check. Also runs daily via
`.github/workflows/smoke-test.yml`.

## Evals

`evals/run_evals.py` runs `evals/cases.yaml`'s ~15 behavioural cases against the real DeepSeek API
and prints a pass/fail table plus an estimated cost. It's a manual regression suite for prompt
changes — run it after any edit to `summary.txt`, `projects.md` or the rules block in
`avatar/context.py`, before deploying:

```bash
python evals/run_evals.py
python evals/run_evals.py --model deepseek-v4-pro   # compare a different model
```

Every assertion targets *behaviour* (a tool call, a required or forbidden substring/pattern in the
reply), never exact wording, so a harmless rewording of an answer doesn't fail the suite. Every
notification channel (Pushover, Telegram) is stubbed for the whole run — no case can ever send a
real notification, regardless of what's in the environment. Deliberately **not** wired into CI: it
costs real money and needs `DEEPSEEK_API_KEY`, so it's run by hand, not on every push.

## Prior art

Built on the shape of [ed-donner/agents](https://github.com/ed-donner/agents) `1_foundations/twin`
from his agents course. The design doc records what changed and why — GitHub knowledge, no personal
PDF in a public repo, cost and abuse controls, and a landing page that solves the free-tier cold
start.
