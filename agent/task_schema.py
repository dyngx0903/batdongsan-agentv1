from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TaskMessage:
    role: str
    content: str
    timestamp: str | None = None


@dataclass
class TaskInput:
    session_id: str
    messages: List[TaskMessage]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessStep:
    step: str
    status: str
    tool: str
    summary: str
    session_id: str | None = None
    selected_tool: str | None = None
    latency_ms: int | None = None
    retrieval_mode: str | None = None
    fallback_reason: str | None = None
    error_type: str | None = None


@dataclass
class LLMInputSafety:
    pii_risk: str = "low"
    hallucination_risk: str = "low"


@dataclass
class LLMQueryUnderstanding:
    intent: str
    confidence: float
    slots: Dict[str, Any] = field(default_factory=dict)
    listing_refs: Dict[str, str] = field(default_factory=dict)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    missing_slots: List[str] = field(default_factory=list)
    clarification_question: str = ""
    safety: LLMInputSafety = field(default_factory=LLMInputSafety)


SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{2,64}$")
_ALLOWED_INPUT_ROLES = {"user", "assistant", "model", "system"}
_CANONICAL_ROLE_MAP = {
    "user": "user",
    "assistant": "assistant",
    "model": "assistant",
    "system": "system",
}
_ALLOWED_INTENTS = {
    "search_listings",
    "explain_listing",
    "similar_listings",
    "compare_listings",
    "suggest_area",
    "analytics_listings",
    "respond_to_user",
}
_ALLOWED_RISK_LEVELS = {"low", "medium", "high"}


def normalize_message_role(role: str) -> str:
    normalized = str(role or "").strip().lower()
    if normalized not in _ALLOWED_INPUT_ROLES:
        raise ValueError("Invalid role: only user, assistant, model, system are allowed")
    return _CANONICAL_ROLE_MAP[normalized]


def parse_task_input(payload: Dict[str, Any]) -> TaskInput:
    if not isinstance(payload, dict):
        raise ValueError("Task payload must be a dictionary")

    session_id = str(payload.get("sessionId") or "").strip()
    if not session_id:
        raise ValueError("Missing required field: sessionId")
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError("Invalid sessionId format: use 2-64 chars [A-Za-z0-9_-]")

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("Missing required field: messages")

    messages: List[TaskMessage] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            raise ValueError("Each message must be a dictionary")
        role = normalize_message_role(item.get("role"))
        content = str(item.get("content") or "").strip()
        if not role or not content:
            raise ValueError("Each message requires role and content")
        messages.append(
            TaskMessage(
                role=role,
                content=content,
                timestamp=item.get("timestamp"),
            )
        )

    metadata = payload.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")

    return TaskInput(
        session_id=session_id,
        messages=messages,
        metadata=metadata,
    )


def parse_llm_query_understanding(payload: Dict[str, Any]) -> LLMQueryUnderstanding:
    if not isinstance(payload, dict):
        raise ValueError("LLM understanding payload must be a dictionary")

    intent = str(payload.get("intent") or "").strip().lower()
    if intent not in _ALLOWED_INTENTS:
        raise ValueError("Invalid intent in LLM understanding payload")

    raw_confidence = payload.get("confidence", 0.0)
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        raise ValueError("Invalid confidence in LLM understanding payload")
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be in range [0.0, 1.0]")

    slots = payload.get("slots") if isinstance(payload.get("slots"), dict) else {}
    listing_refs = payload.get("listing_refs") if isinstance(payload.get("listing_refs"), dict) else {}
    user_profile = payload.get("user_profile") if isinstance(payload.get("user_profile"), dict) else {}

    normalized_listing_refs: Dict[str, str] = {}
    for key in ("listing_ref", "listing_ref_a", "listing_ref_b"):
        value = listing_refs.get(key)
        if value is None:
            continue
        text_value = str(value).strip()
        if text_value:
            normalized_listing_refs[key] = text_value

    raw_missing = payload.get("missing_slots")
    if raw_missing is None:
        missing_slots: List[str] = []
    elif isinstance(raw_missing, list):
        missing_slots = [str(item).strip() for item in raw_missing if str(item).strip()]
    else:
        raise ValueError("missing_slots must be a list when provided")

    clarification_question = str(payload.get("clarification_question") or "").strip()

    safety_payload = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    pii_risk = str(safety_payload.get("pii_risk") or "low").strip().lower()
    hallucination_risk = str(safety_payload.get("hallucination_risk") or "low").strip().lower()
    if pii_risk not in _ALLOWED_RISK_LEVELS:
        pii_risk = "low"
    if hallucination_risk not in _ALLOWED_RISK_LEVELS:
        hallucination_risk = "low"

    return LLMQueryUnderstanding(
        intent=intent,
        confidence=confidence,
        slots=slots,
        listing_refs=normalized_listing_refs,
        user_profile=user_profile,
        missing_slots=missing_slots,
        clarification_question=clarification_question,
        safety=LLMInputSafety(
            pii_risk=pii_risk,
            hallucination_risk=hallucination_risk,
        ),
    )
