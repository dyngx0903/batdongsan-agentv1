from __future__ import annotations

from typing import Any, Dict, Optional

from agent import AgentRuntime


class BatdongsanAIChatbot:
    """Thin chat wrapper over AgentRuntime for UI/API callers."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.runtime = AgentRuntime(config_path=config_path)

    def chat(self, user_message: str, session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "sessionId": session_id,
            "messages": [{"role": "user", "content": user_message}],
            "metadata": metadata or {},
        }
        return self.runtime.execute(payload)

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return self.runtime.execute(task)
