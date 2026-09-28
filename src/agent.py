"""The ReAct agent: an OpenAI chat model that decides when to call tools, built with LangGraph.

`create_agent` (from `langchain.agents`) compiles a LangGraph graph that loops:
model -> (optional) tool calls -> model ... until the model answers without calling a tool.
The agent itself is stateless; the caller passes the conversation history on every turn.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date

import openai
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRequest,
    ToolCallLimitMiddleware,
    dynamic_prompt,
)
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from src.prompts import build_system_prompt
from src.tools import calculator, web_search

logger = logging.getLogger(__name__)

TOOLS = [web_search, calculator]
TOOL_LABELS = {"web_search": "Searching the web", "calculator": "Using calculator"}

MAX_MODEL_CALLS = 6  # Model calls per user message (each tool round trip costs one).
MAX_TOOL_CALLS = 8  # Tool executions per user message; extra calls are refused.
# Hard backstop on graph steps. Each model/tool round trip takes ~5 steps because
# middleware hooks count as steps, so this must stay well above 5 * MAX_MODEL_CALLS.
RECURSION_LIMIT = 50
MODEL_TIMEOUT_SECONDS = 60
MODEL_MAX_RETRIES = 2


@dataclass(frozen=True)
class ToolActivity:
    """One tool call and its result, for display in the UI."""

    name: str
    label: str
    arguments: dict
    result: str | None


@dynamic_prompt
def _system_prompt(request: ModelRequest) -> str:
    # Built per model call so the date stays correct in a long-running app.
    return build_system_prompt(today=date.today())


def create_model(model_name: str) -> ChatOpenAI:
    # The Responses API supports tool calling with OpenAI's reasoning models at any effort level.
    logger.info("Creating OpenAI model %s (Responses API, timeout %ss)", model_name, MODEL_TIMEOUT_SECONDS)
    return ChatOpenAI(
        model=model_name,
        use_responses_api=True,
        timeout=MODEL_TIMEOUT_SECONDS,
        max_retries=MODEL_MAX_RETRIES,
    )


def build_agent(model: str | BaseChatModel) -> CompiledStateGraph:
    """Build the agent from a model name (OpenAI) or any tool-calling chat model."""
    if isinstance(model, str):
        model = create_model(model)
    return create_agent(
        model=model,
        tools=TOOLS,
        middleware=[
            _system_prompt,
            ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="error"),
            ToolCallLimitMiddleware(run_limit=MAX_TOOL_CALLS),
        ],
    )


def stream_agent(agent: CompiledStateGraph, history: list[AnyMessage]) -> Iterator[AnyMessage]:
    """Run one turn and yield each new message as soon as the agent produces it.

    `history` must end with the user's new message. Yielded messages are AIMessages
    (tool requests or the final answer) and ToolMessages (tool results).
    """
    updates = agent.stream(
        {"messages": history},
        config={"recursion_limit": RECURSION_LIMIT},
        stream_mode="updates",
    )
    for update in updates:
        for node_name, node_update in update.items():
            if not isinstance(node_update, dict):
                continue
            for message in node_update.get("messages", []):
                if isinstance(message, AIMessage):
                    for call in message.tool_calls:
                        logger.info("Agent requested tool: %s", call["name"])
                elif isinstance(message, ToolMessage):
                    logger.info("Tool finished: %s (%d chars)", message.name, len(message.text))
                logger.debug("Message from node %s: %s", node_name, type(message).__name__)
                yield message


def final_answer(turn_messages: list[AnyMessage]) -> str:
    """Return the text of the last AI message that is an answer, not a tool request."""
    for message in reversed(turn_messages):
        if isinstance(message, AIMessage) and not message.tool_calls and message.text.strip():
            return str(message.text)
    return ""


def tool_activity(turn_messages: list[AnyMessage]) -> list[ToolActivity]:
    """Pair every tool call in a turn with its result (None while still running)."""
    results = {m.tool_call_id: m for m in turn_messages if isinstance(m, ToolMessage)}
    activity = []
    for message in turn_messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            result = results.get(call["id"])
            activity.append(
                ToolActivity(
                    name=call["name"],
                    label=TOOL_LABELS.get(call["name"], f"Using {call['name']}"),
                    arguments=call["args"],
                    result=str(result.text) if result is not None else None,
                )
            )
    return activity


def describe_error(exc: Exception) -> str:
    """Map an exception to a short, actionable message that is safe to show in the UI."""
    if isinstance(exc, openai.AuthenticationError):
        return "OpenAI rejected the API key. Check OPENAI_API_KEY in your `.env` file."
    if isinstance(exc, openai.PermissionDeniedError):
        return "Your OpenAI project cannot use this model. Check OPENAI_MODEL and your model access."
    if isinstance(exc, openai.NotFoundError):
        return "The configured OpenAI model was not found. Check OPENAI_MODEL in your `.env` file."
    if isinstance(exc, openai.RateLimitError):
        return "OpenAI rate limit or quota reached. Wait a moment and try again, or check your billing."
    if isinstance(exc, openai.APIConnectionError):  # Includes timeouts.
        return "Could not reach OpenAI (network error or timeout). Check your connection and try again."
    if isinstance(exc, openai.BadRequestError):
        return "OpenAI rejected the request. Make sure OPENAI_MODEL supports tool calling."
    if isinstance(exc, openai.InternalServerError):
        return "OpenAI had a temporary server error. Please try again."
    if isinstance(exc, (ModelCallLimitExceededError, GraphRecursionError)):
        return "The assistant took too many steps and was stopped. Try a simpler or more specific question."
    return "Something went wrong while generating a response. Check the app logs for details."
