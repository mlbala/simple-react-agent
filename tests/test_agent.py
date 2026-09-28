"""Agent tests with a scripted fake model: tool routing mechanics, limits, and error messages.

The fake model stands in for OpenAI, so these tests check how the agent executes the
model's decisions, not whether a real model makes good decisions (that needs live checks).
"""

from datetime import date

import httpx
import openai
import pytest
from conftest import scripted_model, tool_call_message
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src import agent as agent_module
from src.agent import (
    MAX_MODEL_CALLS,
    build_agent,
    create_model,
    describe_error,
    final_answer,
    stream_agent,
    tool_activity,
)
from src.prompts import build_system_prompt
from src.tools import search
from src.tools.search import SearchError

OPENAI_URL = "https://api.openai.com/v1/responses"
SEARCH_RESULTS = [{"title": "Japan population", "url": "https://example.com/japan", "snippet": "123,000,000"}]


def run_turn(responses, history):
    model = scripted_model(responses)
    new_messages = list(stream_agent(build_agent(model), history))
    return model, new_messages


def test_answers_directly_without_tools():
    _, messages = run_turn(
        [AIMessage("A decorator wraps a function.")], [HumanMessage("What is a decorator?")]
    )

    assert final_answer(messages) == "A decorator wraps a function."
    assert tool_activity(messages) == []
    assert not any(isinstance(m, ToolMessage) for m in messages)


def test_system_prompt_is_sent_with_todays_date():
    model, _ = run_turn([AIMessage("Hi!")], [HumanMessage("Hello")])

    system_message = model.received[0][0]
    assert isinstance(system_message, SystemMessage)
    assert date.today().strftime("%B %d, %Y") in system_message.text


def test_calculator_call_is_executed_and_result_returned_to_model():
    model, messages = run_turn(
        [tool_call_message("calculator", {"expression": "0.18 * 12500"}), AIMessage("It is 2,250.")],
        [HumanMessage("What is 18% of 12,500?")],
    )

    [activity] = tool_activity(messages)
    assert (activity.name, activity.label, activity.result) == (
        "calculator",
        "Using calculator",
        "0.18 * 12500 = 2250",
    )
    assert model.received[1][-1].text == "0.18 * 12500 = 2250"  # Model saw the tool result.
    assert final_answer(messages) == "It is 2,250."


def test_search_then_calculate(monkeypatch):
    monkeypatch.setattr(search, "search_web", lambda query: SEARCH_RESULTS)

    _, messages = run_turn(
        [
            tool_call_message("web_search", {"query": "Japan population"}, "call_1"),
            tool_call_message("calculator", {"expression": "123000000 * 0.02"}, "call_2"),
            AIMessage("About 2,460,000 people [Japan population](https://example.com/japan)."),
        ],
        [HumanMessage("What is 2% of Japan's population?")],
    )

    activity = tool_activity(messages)
    assert [a.label for a in activity] == ["Searching the web", "Using calculator"]
    assert "https://example.com/japan" in activity[0].result
    assert activity[1].result == "123000000 * 0.02 = 2460000"
    assert "https://example.com/japan" in final_answer(messages)


def test_follow_up_receives_previous_turns():
    history = [
        HumanMessage("What is a Python decorator?"),
        AIMessage("A function that wraps another function."),
        HumanMessage("Show me an example."),
    ]

    model, _ = run_turn([AIMessage("@my_decorator ...")], history)

    received_texts = [m.text for m in model.received[0][1:]]  # Skip the system prompt.
    assert received_texts == [m.text for m in history]


def test_tool_failure_is_reported_to_model(monkeypatch):
    def failing_search(query):
        raise SearchError("Tavily rate limit or usage quota reached. Try again later.")

    monkeypatch.setattr(search, "search_web", failing_search)

    model, messages = run_turn(
        [
            tool_call_message("web_search", {"query": "latest AI news"}),
            AIMessage("Sorry, web search is unavailable right now."),
        ],
        [HumanMessage("What is the latest AI news?")],
    )

    tool_result = model.received[1][-1]
    assert isinstance(tool_result, ToolMessage)
    assert tool_result.text.startswith("Error: web search failed.")
    assert final_answer(messages) == "Sorry, web search is unavailable right now."


def test_runaway_tool_loop_is_stopped():
    def endless_tool_calls():
        call_number = 0
        while True:
            call_number += 1
            yield tool_call_message("calculator", {"expression": "1 + 1"}, f"call_{call_number}")

    model = scripted_model(endless_tool_calls())

    with pytest.raises(ModelCallLimitExceededError) as excinfo:
        list(stream_agent(build_agent(model), [HumanMessage("Loop forever")]))

    assert len(model.received) == MAX_MODEL_CALLS
    assert "too many steps" in describe_error(excinfo.value)


def fake_openai_error(error_class, status):
    response = httpx.Response(status, request=httpx.Request("POST", OPENAI_URL))
    return error_class("secret detail sk-test-123", response=response, body=None)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (fake_openai_error(openai.AuthenticationError, 401), "OPENAI_API_KEY"),
        (fake_openai_error(openai.NotFoundError, 404), "OPENAI_MODEL"),
        (fake_openai_error(openai.RateLimitError, 429), "rate limit"),
        (openai.APIConnectionError(request=httpx.Request("POST", OPENAI_URL)), "Could not reach"),
        (RuntimeError("secret detail sk-test-123"), "Check the app logs"),
    ],
)
def test_errors_map_to_safe_actionable_messages(error, expected):
    message = describe_error(error)

    assert expected in message
    assert "sk-test-123" not in message


def test_openai_model_uses_responses_api(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    model = create_model("gpt-6-luna")

    assert model.model_name == "gpt-6-luna"
    assert model.use_responses_api is True
    assert {tool.name for tool in agent_module.TOOLS} == {"web_search", "calculator"}


def test_system_prompt_covers_required_behaviour():
    prompt = build_system_prompt(date(2026, 1, 2)).lower()

    for phrase in [
        "january 02, 2026",
        "answer directly",
        "web_search",
        "calculator",
        "combine tools",
        "clarifying question",
        "never invent tool results",
        "untrusted",
        "cite",
        "concise",
    ]:
        assert phrase in prompt, phrase
