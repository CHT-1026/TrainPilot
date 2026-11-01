"""File contracts: generation receives only BlindQuestion, never a gold record."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class BlindQuestion(Record):
    qid: str
    question: str = Field(min_length=1)


class Prediction(Record):
    qid: str
    status: Literal['completed', 'api_error', 'incomplete']
    text: str
    error: str | None
    finish_reason: str | None
    request_sha256: str
    latency_seconds: float
    usage: dict | None
    response_model: str | None = None
    response_id: str | None = None
    reasoning_chars: int = 0
    error_detail: str | None = None
    provenance: dict | None = None


class GradingRule(Record):
    qid: str
    accepted_answers: list[str] = Field(min_length=1)
    author: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ReviewDecision(Record):
    item_id: str
    correct: bool
    reason: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)


class ReferenceErratum(Record):
    qid: str
    kind: Literal['reference_correction', 'invalid_question']
    corrected_answer: str
    reason: str = Field(min_length=1)
    sources: list[dict]
