"""Tests for the safe calculator."""

import time
from decimal import Decimal

import pytest

from src.tools.calculator import (
    MAX_EXPRESSION_LENGTH,
    CalculatorError,
    calculator,
    evaluate_expression,
    format_number,
)


def calc(expression: str) -> str:
    return format_number(evaluate_expression(expression))


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3", "5"),
        ("10 - 4", "6"),
        ("6 * 7", "42"),
        ("20 / 4", "5"),
        ("7 / 2", "3.5"),
        ("2 ** 10", "1024"),
        ("-5 + 3", "-2"),
        ("+4", "4"),
        ("1_000 * 3", "3000"),
    ],
)
def test_basic_arithmetic(expression, expected):
    assert calc(expression) == expected


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", "14"),
        ("(2 + 3) * 4", "20"),
        ("10 - 2 - 3", "5"),
        ("100 / 10 / 2", "5"),
        ("2 ** 3 ** 2", "512"),  # Exponentiation is right-associative.
        ("-3 ** 2", "-9"),  # Unary minus binds looser than **.
        ("(-3) ** 2", "9"),
        ("((1 + 2) * (3 + 4)) / 7", "3"),
    ],
)
def test_operator_precedence_and_parentheses(expression, expected):
    assert calc(expression) == expected


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("0.1 + 0.2", "0.3"),  # Exact, unlike binary floats.
        ("1.5 * 2.25", "3.375"),
        ("10 / 4", "2.5"),
        ("19.99 * 3", "59.97"),
        ("1 / 3", "0.3333333333333333333333333333"),
        ("2 ** 0.5", "1.414213562373095048801688724"),
        ("1e-3 * 5", "0.005"),
    ],
)
def test_decimal_calculations(expression, expected):
    assert calc(expression) == expected


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("0.18 * 12500", "2250"),  # 18% of 12,500
        ("12500 * 18 / 100", "2250"),
        ("80 * 1.15", "92"),  # Increase 80 by 15%
        ("200 * (1 - 0.25)", "150"),  # 25% discount
        ("(50 - 40) / 40 * 100", "25"),  # Percent change from 40 to 50
        ("45 / 60 * 100", "75"),  # 45 is what percent of 60
    ],
)
def test_percentage_arithmetic(expression, expected):
    assert calc(expression) == expected


@pytest.mark.parametrize("expression", ["1 / 0", "0 / 0", "5 / (3 - 3)", "0 ** -1", "2.5 / 0.0"])
def test_division_by_zero(expression):
    with pytest.raises(CalculatorError, match="Division by zero"):
        evaluate_expression(expression)


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("2 +", "Invalid expression syntax"),
        ("(2 + 3", "Invalid expression syntax"),
        ("2 3", "Invalid expression syntax"),
        ("12,500 * 0.18", "thousands separators"),
        ("18 % 5", "percentages"),
        ("2 ^ 8", "Use \\*\\*"),
        ("5 // 2", "Unsupported syntax"),
        ("1 < 2", "Unsupported syntax"),
        ("0 ** 0", "undefined"),
        ("(-8) ** 0.5", "fractional power"),
        ("'a' * 3", "Only numbers"),
        ("True + 1", "Only numbers"),
        ("None", "Only numbers"),
        ("2j", "Only numbers"),
    ],
)
def test_invalid_expressions(expression, message):
    with pytest.raises(CalculatorError, match=message):
        evaluate_expression(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hacked')",
        "abs(-1)",
        "pow(2, 10)",
        "(1).real",
        "x + 1",
        "__builtins__",
        "open('/etc/passwd')",
        "[1, 2, 3]",
        "{'a': 1}",
        "lambda: 1",
        "1 if True else 2",
        "(x := 5)",
        "[x for x in range(10)]",
        "f'{1}'",
        "import os",
        "exec('1')",
    ],
)
def test_rejects_unsafe_input(expression):
    with pytest.raises(CalculatorError):
        evaluate_expression(expression)


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("9 ** 9 ** 9", "Exponent is too large"),
        ("2 ** 100000", "Exponent is too large"),
        ("10 ** 1000", "too large"),
        ("1e999 * 10", "too large"),
        ("(10 ** 999) * (10 ** 999)", "too large"),
        ("1 + " * 60 + "1", "too long"),
        ("-" * 50 + "1", "nested too deeply"),
    ],
)
def test_rejects_excessively_expensive_expressions(expression, message):
    started = time.perf_counter()
    with pytest.raises(CalculatorError, match=message):
        evaluate_expression(expression)
    assert time.perf_counter() - started < 1  # Rejected quickly, not after heavy work.


def test_rejects_expression_over_length_limit():
    with pytest.raises(CalculatorError, match="too long"):
        evaluate_expression("1" * (MAX_EXPRESSION_LENGTH + 1))


def test_returns_exact_decimal():
    assert evaluate_expression("0.1 + 0.2") == Decimal("0.3")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Decimal("2250.00"), "2250"),
        (Decimal("-0"), "0"),
        (Decimal("0.500"), "0.5"),
        (Decimal("1E+30"), "1e+30"),
    ],
)
def test_format_number(value, expected):
    assert format_number(value) == expected


def test_tool_returns_result_or_error_text():
    assert calculator.invoke({"expression": "0.18 * 12500"}) == "0.18 * 12500 = 2250"
    assert calculator.invoke({"expression": "1 / 0"}) == "Error: Division by zero."
    assert calculator.invoke({"expression": "abs(-1)"}).startswith("Error: ")
