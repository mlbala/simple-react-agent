"""Shared test helpers. Nothing here calls a real API."""

from collections.abc import Iterable

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import Field


class FakeToolModel(GenericFakeChatModel):
    """A scripted chat model: returns the given AIMessages in order and records its inputs."""

    received: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.received.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def scripted_model(responses: Iterable[AIMessage]) -> FakeToolModel:
    return FakeToolModel(messages=iter(responses))


def tool_call_message(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    """An AIMessage in which the model asks to call one tool."""
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])
