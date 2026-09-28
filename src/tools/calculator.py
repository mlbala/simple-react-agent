"""A safe arithmetic calculator tool.

Expressions are parsed with Python's `ast` module and evaluated by walking an
allow-list of node types. Nothing is ever passed to `eval()` or `exec()`.
Numbers are `decimal.Decimal`, so `0.1 + 0.2` is exactly `0.3`.
"""

import ast
import logging
from decimal import Context, Decimal, DivisionByZero, InvalidOperation, Overflow, localcontext

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

MAX_EXPRESSION_LENGTH = 200
MAX_NESTING_DEPTH = 20
MAX_EXPONENT = 1000  # Largest allowed |exponent| in `a ** b`.
MAX_MAGNITUDE_DIGITS = 1000  # Results and operands must stay below 10**1000.

# 28 significant digits; Overflow/InvalidOperation/DivisionByZero raise instead of returning inf/NaN.
_DECIMAL_CONTEXT = Context(
    prec=28,
    Emax=MAX_MAGNITUDE_DIGITS - 1,
    Emin=-MAX_MAGNITUDE_DIGITS,
    traps=[Overflow, InvalidOperation, DivisionByZero],
)

_BINARY_OPERATORS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
}
_UNARY_OPERATORS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


class CalculatorError(ValueError):
    """Raised for invalid, unsafe, or too-expensive expressions."""


def evaluate_expression(expression: str) -> Decimal:
    """Safely evaluate an arithmetic expression and return the exact `Decimal` result."""
    if not isinstance(expression, str) or not expression.strip():
        raise CalculatorError("Expression is empty.")
    expression = expression.strip()
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError(f"Expression is too long (max {MAX_EXPRESSION_LENGTH} characters).")
    if "," in expression:
        raise CalculatorError("Remove thousands separators (write 12500, not 12,500).")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError("Invalid expression syntax.") from exc

    with localcontext(_DECIMAL_CONTEXT):
        try:
            result = _evaluate(tree.body, depth=0)
        except ZeroDivisionError as exc:  # decimal.DivisionByZero subclasses ZeroDivisionError.
            raise CalculatorError("Division by zero.") from exc
        except Overflow as exc:
            raise CalculatorError("Result is too large.") from exc
        except InvalidOperation as exc:
            raise CalculatorError("The result is undefined or not a real number.") from exc
    if not result.is_finite():
        raise CalculatorError("The result is undefined or not a real number.")
    return result


def format_number(value: Decimal) -> str:
    """Format a Decimal without trailing zeros, e.g. 2250.00 -> '2250'."""
    value = value.normalize()
    if value.is_zero():
        return "0"
    if -10 <= value.adjusted() < 28:
        return format(value, "f")
    return format(value, "e")


def _evaluate(node: ast.AST, depth: int) -> Decimal:
    if depth > MAX_NESTING_DEPTH:
        raise CalculatorError(f"Expression is nested too deeply (max depth {MAX_NESTING_DEPTH}).")

    if isinstance(node, ast.Constant):
        return _to_decimal(node.value)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand, depth + 1))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left, depth + 1)
        right = _evaluate(node.right, depth + 1)
        if isinstance(node.op, ast.Div) and right.is_zero():
            raise CalculatorError("Division by zero.")
        if isinstance(node.op, ast.Pow):
            _check_power(left, right)
        return _BINARY_OPERATORS[type(node.op)](left, right)

    raise CalculatorError(_unsupported_message(node))


def _to_decimal(value: object) -> Decimal:
    # bool is a subclass of int, so reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalculatorError("Only numbers are allowed.")
    if isinstance(value, int):
        if len(str(abs(value))) > MAX_MAGNITUDE_DIGITS:
            raise CalculatorError("Number is too large.")
        return Decimal(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise CalculatorError("Number is too large.")
    return Decimal(repr(value))  # repr keeps the literal as written (0.1 stays 0.1).


def _check_power(base: Decimal, exponent: Decimal) -> None:
    if base.is_zero() and exponent < 0:
        raise CalculatorError("Division by zero.")  # 0 ** -n == 1 / 0 ** n
    if abs(exponent) > MAX_EXPONENT:
        raise CalculatorError(f"Exponent is too large (max {MAX_EXPONENT}).")
    if base < 0 and exponent != exponent.to_integral_value():
        raise CalculatorError("A negative number cannot be raised to a fractional power.")


def _unsupported_message(node: ast.AST) -> str:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return "The % operator is not supported. For percentages, divide by 100 (18% of 250 -> 0.18 * 250)."
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitXor):
        return "Use ** for exponentiation (2 ** 8), not ^."
    if isinstance(node, ast.Name):
        return f"Unknown name {node.id!r}: variables and constants are not supported."
    if isinstance(node, ast.Call):
        return "Function calls are not allowed."
    if isinstance(node, ast.Attribute):
        return "Attribute access is not allowed."
    return f"Unsupported syntax: {type(node).__name__}."


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly. Use this for ANY arithmetic instead of mental math.

    Supports numbers (including decimals), + - * / ** and parentheses.
    Rewrite percentages and word problems as plain arithmetic first, for example:
    - "18% of 12,500" -> "0.18 * 12500"
    - "increase 80 by 15%" -> "80 * 1.15"
    - "percent change from 40 to 50" -> "(50 - 40) / 40 * 100"
    Do not use thousands separators, variables, functions, or the % sign.
    """
    try:
        result = format_number(evaluate_expression(expression))
    except CalculatorError as exc:
        logger.info("Calculator rejected an expression")
        return f"Error: {exc}"
    logger.info("Calculator evaluated an expression (%d chars)", len(expression))
    return f"{expression.strip()} = {result}"
