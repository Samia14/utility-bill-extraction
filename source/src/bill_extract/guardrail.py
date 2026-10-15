"""Decide whether a customer question stays inside the bill-question scope."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from bill_extract.errors import ModelResponseError

# The only job this assistant is allowed to do.
SCOPE_TASK = (
    "Answer customer questions using the bill. "
    "Some answers require calculations or estimates."
)

REFUSAL_MESSAGE = (
    "I can only answer questions about this electricity bill. "
    "I can explain the amounts and dates on the bill, calculate from those figures, "
    "or give an estimate based on this bill."
)

EMPTY_QUESTION_MESSAGE = "Please ask a question about this electricity bill."

LONG_QUESTION_MESSAGE = (
    "That question is too long. Please ask a shorter question about this electricity bill."
)

CHECK_FAILED_MESSAGE = "I could not check that question. Please try again."

MAX_QUESTION_LENGTH = 2000


class ScopeCheck(BaseModel):
    """Result of the scope check. The reason is for logs, not for the customer."""

    in_scope: bool
    reason: str = Field(default="")


def prepare_question(question: str) -> str:
    """Collapse whitespace. Raise nothing; callers inspect the text."""
    return " ".join(question.split())


def customer_reply_for_block(question: str) -> str | None:
    """Return a customer-facing sentence when the question must not be answered.

    None means the question is long enough and short enough to send to the scope check.
    """
    text = prepare_question(question)
    if not text:
        return EMPTY_QUESTION_MESSAGE
    if len(text) > MAX_QUESTION_LENGTH:
        return LONG_QUESTION_MESSAGE
    return None


def decide_scope(question: str, classify: Callable[[str], ScopeCheck]) -> ScopeCheck:
    """Run the scope check. An unreadable decision stops the answer."""
    text = prepare_question(question)
    blocked = customer_reply_for_block(text)
    if blocked is not None:
        return ScopeCheck(in_scope=False, reason=blocked)
    try:
        decision = classify(text)
    except ModelResponseError:
        raise
    except Exception as exc:
        raise ModelResponseError(CHECK_FAILED_MESSAGE) from exc
    if isinstance(decision, ScopeCheck):
        return decision
    if isinstance(decision, dict):
        try:
            return ScopeCheck.model_validate(decision)
        except Exception as exc:
            raise ModelResponseError(CHECK_FAILED_MESSAGE) from exc
    raise ModelResponseError(CHECK_FAILED_MESSAGE)


def scope_check_from_text(content: object) -> ScopeCheck:
    """Read a model reply as a scope decision. Fail closed when it is not one."""
    if isinstance(content, ScopeCheck):
        return content
    if isinstance(content, dict):
        try:
            return ScopeCheck.model_validate(content)
        except Exception as exc:
            raise ModelResponseError(CHECK_FAILED_MESSAGE) from exc
    text = content if isinstance(content, str) else ""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ModelResponseError(CHECK_FAILED_MESSAGE)
    try:
        return ScopeCheck.model_validate_json(text[start : end + 1])
    except Exception as exc:
        raise ModelResponseError(CHECK_FAILED_MESSAGE) from exc
