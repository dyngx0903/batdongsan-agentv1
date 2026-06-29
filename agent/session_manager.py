from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .task_schema import normalize_message_role


@dataclass
class SessionState:
    session_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}

    def get_or_create(self, session_id: str) -> SessionState:
        key = str(session_id).strip()
        if key not in self._sessions:
            self._sessions[key] = SessionState(session_id=key)
        return self._sessions[key]

    def append_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> SessionState:
        state = self.get_or_create(session_id)
        normalized_messages: List[Dict[str, Any]] = []
        for item in messages:
            role = normalize_message_role(item.get("role"))
            normalized_item = dict(item)
            normalized_item["role"] = role
            normalized_messages.append(normalized_item)
        state.messages.extend(normalized_messages)
        return state

    def merge_metadata(self, session_id: str, metadata: Dict[str, Any]) -> SessionState:
        state = self.get_or_create(session_id)
        state.metadata.update(metadata or {})
        return state
