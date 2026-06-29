from __future__ import annotations

from typing import Any, Dict, List

from agent.common import get_logger, ExecutionMetrics
from agent.data_access import AgentListingDataAccess


logger = get_logger("tool_suggest_area")


def _coerce_int(value: Any, default: int, minimum: int = 1, maximum: int = 20) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = default
    out = max(minimum, out)
    return min(maximum, out)


def _normalize_missing_fields(fields: Any) -> List[str]:
    out: List[str] = []
    for raw in fields or []:
        value = str(raw or "").strip()
        if not value:
            continue
        if value == "work_location":
            value = "commuting_destination"
        out.append(value)
    # Preserve order while deduplicating.
    return list(dict.fromkeys(out))


def _format_area_recommendation(area: Dict[str, Any], rank: int = 1, total_top: int = 3) -> str:
    """
    CHANGED (2026-06-27): Format single area with medal emoji, trade-offs (pros/cons), and reasoning.
    
    Args:
        area: Area dict with score, matching_reasons, non_matching_reasons, etc.
        rank: Position in ranking (1=🥇, 2=🥈, 3=🥉)
        total_top: Total areas to show (default 3)
    
    Returns:
        Formatted markdown string with area recommendation
    """
    medals = ["🥇", "🥈", "🥉"]
    medal = medals[min(rank - 1, 2)]
    area_name = str(area.get("district") or area.get("area") or "Unknown").strip()
    score = float(area.get("score") or 0.0)
    
    # Build pros (✅ matching_reasons)
    matching = list(area.get("matching_reasons") or [])
    non_matching = list(area.get("non_matching_reasons") or [])
    
    lines = [f"{medal} **{area_name}** — Score {score:.2f}"]
    
    # Pros
    if matching:
        for reason in matching[:3]:
            if reason and str(reason).strip():
                lines.append(f"  ✅ {reason.strip()}")
    
    # Cons/Trade-offs (⚠️)
    if non_matching:
        lines.append("  **Lưu ý:**")
        for reason in non_matching[:2]:
            if reason and str(reason).strip():
                # Remove emoji if already present
                reason_clean = str(reason).replace("⚠️", "").strip()
                lines.append(f"  ⚠️ {reason_clean}")
    
    return "\n".join(lines)


def _generate_final_recommendation(recommendations: List[Dict[str, Any]]) -> str:
    """
    CHANGED (2026-06-27): Generate final recommendation section with suggested area based on priorities.
    
    Shows different suggestions based on what matters most to user:
    - If tight budget: cheaper option
    - If center priority: most central
    - If balanced: highest score
    """
    if not recommendations or len(recommendations) == 0:
        return ""
    
    top_area = recommendations[0]
    area_name = str(top_area.get("district") or top_area.get("area") or "Unknown").strip()
    
    recommendation = f"""
---
⭐ **Khuyến nghị**

Nếu chỉ chọn một khu vực để xem listing trước, mình đề xuất **{area_name}** vì:
• Vị trí và chất lượng sống cân bằng tốt
• Khả năng tìm được bất động sản phù hợp ngân sách
• Tiềm năng tăng giá lâu dài

**Cách tiếp theo**: Chọn một khu vực để mình lọc listing phù hợp.
    """.strip()
    
    return recommendation


def _format_top_areas_narrative(recommendations: List[Dict[str, Any]]) -> str:
    """
    CHANGED (2026-06-27): Format top 3 areas in readable narrative with medals and trade-offs.
    
    Shows 🥇🥈🥉 with pros/cons instead of dumping raw data.
    """
    if not recommendations:
        return "Không có khu vực phù hợp với tiêu chí tìm kiếm."
    
    lines = []
    for idx, area in enumerate(recommendations[:3], start=1):
        formatted = _format_area_recommendation(area, rank=idx, total_top=3)
        lines.append(formatted)
        if idx < min(3, len(recommendations)):
            lines.append("-" * 60)
    
    return "\n".join(lines)


def run_suggest_area(advisor: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    metrics = ExecutionMetrics()
    
    try:
        metrics.add_step("parse_args")
        query = str(args.get("query") or "").strip()
        top_k = _coerce_int(args.get("top_k"), default=5)
        logger.info("normalize_input query_len=%d", len(query))

        metrics.add_step("suggest_areas")
        dal = AgentListingDataAccess(config_path=args.get("config_path"))
        suggested = dal.suggest_area(query=query, top_k=top_k)
        
        metrics.add_step("format_response")
        raw = {
            "query": query,
            "user_query": query,
            "need_clarification": suggested.need_clarification,
            "missing_fields": suggested.missing_fields,
            "area_rankings": getattr(suggested, "area_recommendations", []) or [],
            "area_recommendations": suggested.area_recommendations,
            "next_clarification_prompt": getattr(suggested, "next_clarification_prompt", None),
            "suggested_next_tool": getattr(suggested, "suggested_next_tool", None),
            "next_user_action": getattr(suggested, "next_user_action", None),
            "summary": suggested.summary,
            "message": suggested.summary,
        }

        missing_fields = _normalize_missing_fields(raw.get("missing_fields") or [])
        need_clarification = bool(raw.get("need_clarification"))
        status = "need_clarification" if need_clarification else "ok"

        logger.info(
            "branch_decision need_clarification=%s missing_fields=%s",
            need_clarification,
            missing_fields,
        )

        # CHANGED (2026-06-27): Generate formatted message with trade-offs and recommendation
        recommendations = raw.get("area_recommendations") or []
        if status == "ok" and recommendations:
            formatted_areas = _format_top_areas_narrative(recommendations)
            final_recommendation = _generate_final_recommendation(recommendations)
            formatted_message = f"{formatted_areas}\n\n{final_recommendation}"
        else:
            formatted_message = raw.get("next_clarification_prompt") or raw.get("message") or "Area suggestion completed"

        stable = {
            "tool": "suggest_area",
            "status": status,
            "query": query,
            "user_query": query,
            "need_clarification": need_clarification,
            "clarification_required": need_clarification,
            "missing_fields": missing_fields,
            "area_rankings": raw.get("area_rankings") or [],
            "area_recommendations": raw.get("area_recommendations") or [],
            "next_clarification_prompt": raw.get("next_clarification_prompt"),
            "suggested_next_tool": raw.get("suggested_next_tool"),
            "next_user_action": raw.get("next_user_action"),
            "message": formatted_message,
            "execution": metrics.finalize(),
        }

        return {
            **raw,
            **stable,
        }
    except Exception as exc:
        metrics.set_error(type(exc).__name__)
        logger.exception("suggest_area failed error=%s", exc)
        return {
            "tool": "suggest_area",
            "status": "tool_error",
            "query": str(args.get("query") or "").strip(),
            "user_query": str(args.get("query") or "").strip(),
            "need_clarification": False,
            "missing_fields": [],
            "area_rankings": [],
            "area_recommendations": [],
            "next_clarification_prompt": None,
            "suggested_next_tool": None,
            "next_user_action": None,
            "message": f"Area suggestion failed: {exc}",
            "execution": metrics.finalize(),
        }
