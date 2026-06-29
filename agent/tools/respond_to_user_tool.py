from __future__ import annotations

from typing import Any, Dict

from agent.common import ExecutionMetrics

def run_respond_to_user(advisor: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    metrics = ExecutionMetrics()
    metrics.add_step("extract_response")
    
    try:
        final_response = str(args.get("final_response") or args.get("message") or "").strip()
        if not final_response:
            final_response = "Không có nội dung phản hồi phù hợp cho yêu cầu hiện tại."

        metrics.add_step("format_response")
        return {
            "tool": "respond_to_user",
            "status": "ok",
            "final_response": final_response,
            "message": final_response,
            "execution": metrics.finalize(),
        }
    except Exception as exc:
        metrics.set_error(type(exc).__name__)
        return {
            "tool": "respond_to_user",
            "status": "tool_error",
            "final_response": f"Error preparing response: {exc}",
            "message": f"Error preparing response: {exc}",
            "execution": metrics.finalize(),
        }
