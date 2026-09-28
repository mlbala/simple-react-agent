"""Streamlit UI tests using Streamlit's headless AppTest runner and a scripted fake model."""

from pathlib import Path

import pytest
import streamlit as st
from conftest import scripted_model, tool_call_message
from langchain_core.messages import AIMessage, HumanMessage
from streamlit.testing.v1 import AppTest

import src.agent
import src.config

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
APP_TIMEOUT_SECONDS = 30


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    # Never read a developer's real `.env` file during tests.
    monkeypatch.setattr(src.config, "load_dotenv", lambda **kwargs: False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def use_fake_model(monkeypatch, responses):
    """Make the app build its agent around a scripted model instead of OpenAI."""
    model = scripted_model(responses)
    real_build_agent = src.agent.build_agent
    monkeypatch.setattr(src.agent, "build_agent", lambda model_name: real_build_agent(model))
    return model


def start_app() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT_SECONDS)
    app.run()
    assert not app.exception, app.exception
    return app


def assistant_markdown(app: AppTest) -> str:
    return "\n".join(block.value for block in app.markdown)


def test_app_starts_with_title_intro_and_examples(monkeypatch):
    app = start_app()

    assert app.title[0].value == "Simple ReAct Assistant"
    assert "search the web" in app.markdown[0].value
    labels = [button.label for button in app.sidebar.button]
    assert "Clear chat" in labels
    assert "What is 18% of 12,500?" in labels
    assert len(app.chat_input) == 1


def test_missing_config_shows_actionable_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.delenv("TAVILY_API_KEY")

    app = start_app()

    assert len(app.error) == 1
    message = app.error[0].value
    assert "OPENAI_API_KEY" in message and "TAVILY_API_KEY" in message
    assert ".env.example" in message
    assert len(app.chat_input) == 0  # The app stops before showing the chat.


def test_chat_turn_shows_tool_activity_and_answer(monkeypatch):
    use_fake_model(
        monkeypatch,
        [
            tool_call_message("calculator", {"expression": "0.18 * 12500"}),
            AIMessage("18% of 12,500 is **2,250**."),
        ],
    )
    app = start_app()

    app.chat_input[0].set_value("What is 18% of 12,500?").run()

    assert not app.exception
    assert "18% of 12,500 is **2,250**." in assistant_markdown(app)
    assert any("Using calculator" in caption.value for caption in app.caption)
    assert app.expander[0].label == "Tool details"
    assert "0.18 * 12500 = 2250" in app.code[0].value
    assert len(app.session_state["messages"]) == 4  # user, tool request, tool result, answer


def test_follow_up_uses_history_and_clear_chat_resets_it(monkeypatch):
    model = use_fake_model(
        monkeypatch,
        [AIMessage("A decorator wraps a function."), AIMessage("Here is an example.")],
    )
    app = start_app()

    app.chat_input[0].set_value("What is a Python decorator?").run()
    app.chat_input[0].set_value("Show me an example").run()

    # The second model call received the first question and answer as context.
    second_call = [m.text for m in model.received[1] if isinstance(m, (HumanMessage, AIMessage))]
    assert second_call == [
        "What is a Python decorator?",
        "A decorator wraps a function.",
        "Show me an example",
    ]
    assert len(app.chat_message) == 4

    next(b for b in app.sidebar.button if b.label == "Clear chat").click().run()

    assert app.session_state["messages"] == []
    assert len(app.chat_message) == 0


def test_example_button_submits_question(monkeypatch):
    use_fake_model(monkeypatch, [AIMessage("A decorator wraps a function.")])
    app = start_app()

    next(b for b in app.sidebar.button if b.label == "What is a Python decorator?").click().run()

    assert app.session_state["messages"][0].text == "What is a Python decorator?"
    assert "A decorator wraps a function." in assistant_markdown(app)


def test_agent_failure_shows_safe_message_without_details(monkeypatch):
    def broken_build_agent(model_name):
        raise RuntimeError("boom: secret-token-123")

    monkeypatch.setattr(src.agent, "build_agent", broken_build_agent)
    app = start_app()

    app.chat_input[0].set_value("Hello").run()

    assert not app.exception
    text = assistant_markdown(app)
    assert "Something went wrong" in text
    assert "secret-token-123" not in text and "Traceback" not in text
    # The failed turn is recorded without partial tool messages.
    assert [type(m).__name__ for m in app.session_state["messages"]] == ["HumanMessage", "AIMessage"]


def test_reruns_do_not_duplicate_messages(monkeypatch):
    use_fake_model(monkeypatch, [AIMessage("A decorator wraps a function.")])
    app = start_app()
    app.chat_input[0].set_value("What is a Python decorator?").run()

    app.run()
    app.run()

    assert len(app.session_state["messages"]) == 2
    assert len(app.chat_message) == 2


def test_sessions_do_not_share_history(monkeypatch):
    model = use_fake_model(monkeypatch, [AIMessage("First answer."), AIMessage("Second answer.")])
    first_session = start_app()
    first_session.chat_input[0].set_value("Question from session one").run()

    second_session = start_app()
    second_session.chat_input[0].set_value("Question from session two").run()

    assert len(second_session.session_state["messages"]) == 2
    assert [m.text for m in model.received[1] if isinstance(m, HumanMessage)] == ["Question from session two"]
