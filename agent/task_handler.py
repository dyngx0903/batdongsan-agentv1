from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


class TaskHandler:
    """Normalize runtime task I/O without coupling to transport layer."""

    def validate_input_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        errors: List[str] = []
        if not isinstance(task, dict):
            return {"valid": False, "errors": ["Task payload must be a dictionary"]}

        raw_messages = task.get("messages", task.get("message"))
        if not isinstance(raw_messages, list) or not raw_messages:
            errors.append("Field 'messages' (or alias 'message') must be a non-empty list")
        else:
            for idx, item in enumerate(raw_messages):
                if not isinstance(item, dict):
                    errors.append(f"Message {idx} must be a dictionary")
                    continue
                if "role" not in item:
                    errors.append(f"Message {idx} missing 'role'")
                if "content" not in item:
                    errors.append(f"Message {idx} missing 'content'")

        session_id = task.get("sessionId")
        if session_id is not None and not isinstance(session_id, str):
            errors.append("Field 'sessionId' must be a string when provided")

        metadata = task.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            errors.append("Field 'metadata' must be a dictionary when provided")

        return {"valid": len(errors) == 0, "errors": errors}

    def normalize_input_task(self, task: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        messages = task.get("messages", task.get("message", []))
        return {
            "sessionId": session_id,
            "messages": messages,
            "metadata": task.get("metadata") or {},
        }

    def extract_user_query(self, messages: List[Dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if str(msg.get("role") or "").strip().lower() == "user":
                return str(msg.get("content") or "").strip()
        return " ".join(str(msg.get("content") or "").strip() for msg in messages).strip()

    @staticmethod
    def build_process_step(
        *,
        step: str,
        status: str,
        tool: str,
        summary: str,
        session_id: Optional[str] = None,
        latency_ms: Optional[int] = None,
        fallback_reason: Optional[str] = None,
        retrieval_mode: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "step": step,
            "status": status,
            "tool": tool,
            "summary": summary,
        }
        if session_id:
            out["session_id"] = session_id
        if latency_ms is not None:
            out["latency_ms"] = int(latency_ms)
        if fallback_reason:
            out["fallback_reason"] = fallback_reason
        if retrieval_mode:
            out["retrieval_mode"] = retrieval_mode
        if error_type:
            out["error_type"] = error_type
        return out

    @staticmethod
    def now_iso() -> str:
        return datetime.utcnow().isoformat() + "Z"
