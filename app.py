"""Streamlit chat UI for the Simple ReAct Assistant.

Run with:  uv run streamlit run app.py

Conversation history is a list of LangChain messages in `st.session_state.messages`,
which Streamlit keeps separately for each browser session. That list is the single
source of truth: it is rendered on every rerun and passed to the (stateless) agent on
every turn. A turn is committed only once it finishes, so reruns never duplicate it.
"""

import logging
import time

import streamlit as st
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from src.agent import (
    ToolActivity,
    build_agent,
    describe_error,
    final_answer,
    stream_agent,
    tool_activity,
)
from src.config import ConfigError, configure_logging, load_settings

logger = logging.getLogger("app")

EXAMPLE_QUESTIONS = [
    "What is a Python decorator?",
    "What are the latest developments in AI?",
    "What is 18% of 12,500?",
    "What is the current population of Japan, and what is 2% of it?",
]
TOOL_ICONS = {"web_search": "🔎", "calculator": "🧮"}
MAX_PROMPT_CHARS = 2000
MAX_RESULT_PREVIEW_CHARS = 800


@st.cache_resource(show_spinner=False)
def get_agent(model_name: str):
    # Shared by all sessions. Safe because the agent keeps no conversation state.
    return build_agent(model_name)


def queue_prompt(prompt: str) -> None:
    st.session_state.queued_prompt = prompt


def clear_chat() -> None:
    # Resets both what the user sees and what the agent receives next turn.
    st.session_state.messages = []
    st.session_state.pop("queued_prompt", None)


def to_markdown(text: str) -> str:
    """Escape `$` outside code blocks so amounts like $5 aren't rendered as LaTeX."""
    parts = text.split("```")
    return "```".join(p if i % 2 else p.replace("$", r"\$") for i, p in enumerate(parts))


def split_turns(messages: list[AnyMessage]) -> list[tuple[HumanMessage, list[AnyMessage]]]:
    """Group history into (user message, assistant/tool messages that answered it)."""
    turns: list[tuple[HumanMessage, list[AnyMessage]]] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            turns.append((message, []))
        elif turns:
            turns[-1][1].append(message)
    return turns


def describe_call(item: ToolActivity) -> str:
    """A one-line summary of a tool's input, e.g. the search query or the expression."""
    value = item.arguments.get("query") or item.arguments.get("expression") or ""
    value = " ".join(str(value).split()).replace("`", "'")
    return value if len(value) <= 120 else value[:119] + "…"


def render_assistant_turn(turn_messages: list[AnyMessage]) -> None:
    activity = tool_activity(turn_messages)
    if activity:
        labels = dict.fromkeys(f"{TOOL_ICONS.get(a.name, '🛠️')} {a.label}" for a in activity)
        st.caption(" · ".join(labels))
        with st.expander("Tool details"):
            for item in activity:
                st.markdown(f"**{item.name}** · `{describe_call(item)}`")
                result = item.result or "(no result)"
                if len(result) > MAX_RESULT_PREVIEW_CHARS:
                    result = result[:MAX_RESULT_PREVIEW_CHARS] + "…"
                # Rendered as plain text: web content is untrusted.
                st.code(result, language=None, wrap_lines=True)
    answer = final_answer(turn_messages)
    st.markdown(to_markdown(answer) if answer else "_No answer was returned._")


def run_turn(prompt: str, model_name: str) -> None:
    """Run the agent on a new prompt, showing tool activity live, then commit the turn."""
    user_message = HumanMessage(prompt)
    with st.chat_message("user"):
        st.markdown(to_markdown(prompt))

    with st.chat_message("assistant"):
        new_messages: list[AnyMessage] = []
        started = time.monotonic()
        with st.status("Thinking…") as status:
            try:
                history = [*st.session_state.messages, user_message]
                for message in stream_agent(get_agent(model_name), history):
                    new_messages.append(message)
                    if isinstance(message, AIMessage) and message.tool_calls:
                        for item in tool_activity([message]):
                            status.update(label=f"{item.label}…")
                            icon = TOOL_ICONS.get(item.name, "🛠️")
                            status.markdown(f"{icon} {item.label}: `{describe_call(item)}`")
            # UI boundary: log the full error server-side, show only a safe summary.
            except Exception as exc:
                logger.exception("Agent run failed with %s", type(exc).__name__)
                status.update(label="Something went wrong", state="error")
                # Drop the partial turn (it may hold unanswered tool calls) and record the error.
                new_messages = [AIMessage(f"⚠️ {describe_error(exc)}")]
            else:
                used = len(tool_activity(new_messages))
                logger.info("Turn finished in %.1fs using %d tool call(s)", time.monotonic() - started, used)
                if not final_answer(new_messages):
                    logger.warning("Model finished the turn without any answer text")
                if used:
                    label = f"Done · used {used} tool{'' if used == 1 else 's'}"
                else:
                    label = "Done · answered directly"
                status.update(label=label, state="complete")

    st.session_state.messages = [*st.session_state.messages, user_message, *new_messages]
    st.rerun()  # Re-render from history so completed turns always look the same.


def main() -> None:
    st.set_page_config(page_title="Simple ReAct Assistant", page_icon="🤖")
    st.title("Simple ReAct Assistant")
    st.markdown(
        "Ask me anything. I answer from my own knowledge when I can, **search the web** "
        "for current information (with sources), and use a **calculator** for math. "
        "I decide which to use, and I can combine them."
    )

    try:
        settings = load_settings()
    except ConfigError as exc:
        configure_logging()
        logger.warning("Configuration problem: %s", exc)
        st.error(f"**Configuration needed.** {exc}")
        st.stop()
    configure_logging(settings.log_level)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("Try an example")
        for question in EXAMPLE_QUESTIONS:
            st.button(question, on_click=queue_prompt, args=(question,), width="stretch")
        st.divider()
        st.button("Clear chat", on_click=clear_chat, type="primary", width="stretch")
        st.caption(f"Model: `{settings.openai_model}` · Chat history lasts for this browser session only.")

    for user_message, turn_messages in split_turns(st.session_state.messages):
        with st.chat_message("user"):
            st.markdown(to_markdown(str(user_message.text)))
        with st.chat_message("assistant"):
            render_assistant_turn(turn_messages)

    prompt = st.chat_input("Ask a question…", max_chars=MAX_PROMPT_CHARS)
    prompt = prompt or st.session_state.pop("queued_prompt", None)
    if prompt and prompt.strip():
        run_turn(prompt.strip(), settings.openai_model)


# `streamlit run app.py` executes this file as the `__main__` module on every rerun,
# so the guard still runs the app; a plain `import app` no longer renders the UI.
if __name__ == "__main__":
    main()
