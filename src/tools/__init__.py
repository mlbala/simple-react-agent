"""Tools the agent can choose to call. Model knowledge needs no tool."""

from src.tools.calculator import calculator
from src.tools.search import web_search

__all__ = ["calculator", "web_search"]
