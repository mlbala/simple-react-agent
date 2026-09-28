"""Tests for the Tavily search tool, using a fake client (no network calls)."""

import pytest
import requests
from tavily.errors import (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    UsageLimitExceededError,
)
from tavily.errors import TimeoutError as TavilyTimeoutError

from src.tools import search
from src.tools.search import (
    MAX_ATTEMPTS,
    MAX_OUTPUT_CHARS,
    MAX_RESULTS,
    MAX_SNIPPET_CHARS,
    SearchError,
    format_results,
    search_web,
    web_search,
)


class FakeTavilyClient:
    """Returns (or raises) the scripted outcomes in order and records each call."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def tavily_response(count=2, content="Useful snippet."):
    return {
        "results": [
            {"title": f"Result {i}", "url": f"https://example.com/{i}", "content": content}
            for i in range(1, count + 1)
        ]
    }


def http_error(status: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(response=response)


def no_sleep(seconds):
    pass


def test_returns_titles_urls_and_snippets_with_bounded_request():
    client = FakeTavilyClient(tavily_response(count=2))

    results = search_web("latest AI news", client=client)

    assert results == [
        {"title": "Result 1", "url": "https://example.com/1", "snippet": "Useful snippet."},
        {"title": "Result 2", "url": "https://example.com/2", "snippet": "Useful snippet."},
    ]
    call = client.calls[0]
    assert call["max_results"] == MAX_RESULTS
    assert call["timeout"] > 0


def test_limits_result_count_snippet_size_and_skips_bad_urls():
    response = tavily_response(count=MAX_RESULTS + 3, content="word " * 1000)
    response["results"].insert(0, {"title": "No URL", "url": "", "content": "x"})
    response["results"].insert(1, {"title": "Script", "url": "javascript:alert(1)", "content": "x"})

    results = search_web("query", client=FakeTavilyClient(response))

    assert len(results) == MAX_RESULTS
    assert all(r["url"].startswith("https://") for r in results)
    assert all(len(r["snippet"]) <= MAX_SNIPPET_CHARS for r in results)


def test_formatted_output_is_capped_and_marks_content_untrusted():
    results = [
        {"title": f"T{i}", "url": f"https://example.com/{i}", "snippet": "s" * MAX_SNIPPET_CHARS}
        for i in range(20)
    ]

    text = format_results("query", results)

    assert len(text) <= MAX_OUTPUT_CHARS
    assert "Untrusted" in text
    assert "https://example.com/0" in text


@pytest.mark.parametrize("response", [{}, {"results": None}, {"results": "oops"}, None])
def test_malformed_response_is_treated_as_no_results(response):
    assert search_web("query", client=FakeTavilyClient(response)) == []


def test_empty_results_return_clear_message(monkeypatch):
    monkeypatch.setattr(search, "search_web", lambda query: [])

    assert web_search.invoke({"query": "nothing"}).startswith("No web results found")


def test_empty_query_is_rejected():
    with pytest.raises(SearchError, match="empty"):
        search_web("   ", client=FakeTavilyClient())


def test_missing_api_key_is_reported_without_calling_tavily(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(SearchError, match="TAVILY_API_KEY"):
        search_web("query")


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (InvalidAPIKeyError("bad key"), "rejected the API key"),
        (UsageLimitExceededError("slow down"), "rate limit"),
        (ForbiddenError("no credits"), "denied"),
        (BadRequestError("bad query"), "invalid"),
        (http_error(404), "HTTP 404"),
        (requests.JSONDecodeError("Expecting value", "<html>", 0), "unreadable"),
        (requests.exceptions.ChunkedEncodingError("connection broken"), "incomplete"),
    ],
)
def test_permanent_errors_are_not_retried(error, message):
    client = FakeTavilyClient(error, tavily_response())

    with pytest.raises(SearchError, match=message):
        search_web("query", client=client, sleep=no_sleep)
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "error",
    [TavilyTimeoutError(15), requests.ConnectionError("network down"), http_error(503)],
)
def test_transient_errors_are_retried_then_succeed(error):
    delays = []
    client = FakeTavilyClient(error, tavily_response(count=1))

    results = search_web("query", client=client, sleep=delays.append)

    assert len(results) == 1
    assert len(client.calls) == 2
    assert delays == [search.RETRY_BACKOFF_SECONDS]


def test_gives_up_after_bounded_attempts():
    client = FakeTavilyClient(*[TavilyTimeoutError(15)] * (MAX_ATTEMPTS + 2))

    with pytest.raises(SearchError, match="Could not reach Tavily"):
        search_web("query", client=client, sleep=no_sleep)
    assert len(client.calls) == MAX_ATTEMPTS


def test_tool_reports_failures_honestly(monkeypatch):
    def failing_search(query):
        raise SearchError("Tavily rejected the API key. Check TAVILY_API_KEY.")

    monkeypatch.setattr(search, "search_web", failing_search)

    output = web_search.invoke({"query": "latest news"})

    assert output.startswith("Error: web search failed.")
    assert "Do not invent results" in output
