"""Gradio ChatInterface for Steve's AI twin.

Thin by design: the system prompt is built once at import time (never per
request, or DeepSeek's prompt cache never gets a stable prefix — see
docs/design.md §4), and each turn is just guardrails → the DeepSeek
tool-use loop → guardrails again for spend accounting.
"""

import logging
import os

import gradio as gr
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from avatar import context, guardrails, llm, styles

load_dotenv()
logging.basicConfig(level=logging.INFO)

SYSTEM_PROMPT = context.build_system_prompt()


def _client_ip(request):
    if request is None or request.client is None:
        return "unknown"
    return request.client.host


def _session_id(request):
    if request is None or not request.session_hash:
        return "unknown"
    return request.session_hash


def _to_conversation(history, message):
    conversation = [{"role": m["role"], "content": m["content"]} for m in history]
    conversation.append({"role": "user", "content": message})
    return conversation


def chat(message, history, request: gr.Request):
    session_id = _session_id(request)
    ip = _client_ip(request)

    allowed, refusal = guardrails.check_request(session_id, ip, message)
    if not allowed:
        return refusal

    conversation = _to_conversation(history, message)
    reply, usage = llm.send_message(conversation, SYSTEM_PROMPT)
    guardrails.record_usage(usage)
    return reply


def build_demo():
    with gr.Blocks(title=styles.TITLE) as demo:
        gr.Markdown(
            f"# {styles.TITLE}\n\n{styles.DESCRIPTION}", elem_id="avatar-header"
        )
        gr.ChatInterface(
            chat, examples=styles.EXAMPLE_QUESTIONS, run_examples_on_click=False
        )
        gr.Markdown(styles.FOOTER_MARKDOWN, elem_id="avatar-footer")
    return demo


def build_health_app():
    """Minimal FastAPI app exposing GET /health for Render's health check.

    Deliberately side-effect free: no logging, no external calls, no
    dependency on DeepSeek/Pushover/Telegram. Just proves the process is up.
    """
    health_app = FastAPI()

    @health_app.get("/health")
    def health():
        return JSONResponse({"status": "ok"})

    return health_app


def build_app():
    """Combine the chat UI and the health endpoint into one FastAPI app.

    demo.launch()'s app_kwargs is for keyword arguments to Gradio's own
    FastAPI constructor (e.g. docs_url) — passing a second app instance
    under the "app" key there is silently accepted and silently does
    nothing; /health returns 404 (confirmed by launching it and hitting
    both routes — see issue #194's review discussion). gr.mount_gradio_app
    is the actual documented way to serve a custom FastAPI app's routes
    alongside a Gradio Blocks demo.
    """
    return gr.mount_gradio_app(
        build_health_app(), build_demo(), path="/", css=styles.CSS
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        build_app(),
        host=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
