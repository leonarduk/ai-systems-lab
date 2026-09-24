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
    # NOTE: scripts/smoke_test.py cross-checks the env vars this module reads.
    # Keep REQUIRED_ENV_VARS / OPTIONAL_ENV_VARS in that script in sync with
    # any new os.environ.get(...) calls added here.
    with gr.Blocks(title=styles.TITLE) as demo:
        gr.Markdown(
            f"# {styles.TITLE}\n\n{styles.DESCRIPTION}", elem_id="avatar-header"
        )
        gr.ChatInterface(
            chat, examples=styles.EXAMPLE_QUESTIONS, run_examples_on_click=False
        )
        gr.Markdown(styles.FOOTER_MARKDOWN, elem_id="avatar-footer")
    return demo


if __name__ == "__main__":
    port = os.environ.get("GRADIO_SERVER_PORT")
    build_demo().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME"),
        server_port=int(port) if port else None,
        css=styles.CSS,
    )
