from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOP_OPTIONS = 10
DEFAULT_MAX_OPTION_TEXT = 280
DEFAULT_MAX_SUMMARY_TEXT = 12000
DEFAULT_MAX_BULLET_TEXT = 320


class RuntimeResponseFormatter:
    """Standardizes runtime response shape while preserving legacy keys."""

    def __init__(
        self,
        *,
        max_top_options: int = DEFAULT_MAX_TOP_OPTIONS,
        max_option_text: int = DEFAULT_MAX_OPTION_TEXT,
        max_summary_text: int = DEFAULT_MAX_SUMMARY_TEXT,
        max_bullet_text: int = DEFAULT_MAX_BULLET_TEXT,
    ) -> None:
        self.max_top_options = max_top_options
        self.max_option_text = max_option_text
        self.max_summary_text = max_summary_text
        self.max_bullet_text = max_bullet_text

    @staticmethod
    def _limit_text(value: Any, max_len: int) -> str:
        text = str(value or "").strip()
        if len(text) <= max_len:
            return text
        if max_len <= 3:
            return text[:max_len]
        return text[: max_len - 3].rstrip() + "..."

    def compact_option(self, item: Any) -> Dict[str, Any]:
        if not isinstance(item, dict):
            return {"value": self._limit_text(item, self.max_option_text)}

        keep_keys = {
            "detail_view",
            "source",
            "listing_id",
            "title",
            "url",
            "listing_ref",
            "price_text",
            "area_text",
            "price_value_vnd",
            "area_m2",
            "district",
            "city",
            "ward",
            "street",
            "property_type",
            "transaction_type",
            "project",
            "bedrooms",
            "bathrooms",
            "floors",
            "frontage_width_m",
            "road_access_width_m",
            "legal_status",
            "direction",
            "structure",
            "interior",
            "access",
            "location_quality",
            "neighborhood_quality",
            "view",
            "suitable_for",
            "price_analysis",
            "location_analysis",
            "size_analysis",
            "legal_analysis",
            "utilities",
            "enrichment_matches",
            "amenities_area",
            "amenities_building",
            "nearby_landmarks",
            "nearby_transport",
            "nearby_roads",
            "search_document",
            "summary",
            "winner",
            "winner_ref",
            "winner_reason",
            "trade_offs",
            "tradeoffs",
            "key_reasons",
            "listing_a_ref",
            "listing_b_ref",
            "fit_score",
            "budget_note",
            "message",
            "found",
            "similarity_summary",
        }

        compact: Dict[str, Any] = {}
        for key in keep_keys:
            if key in item and item[key] is not None:
                value = item[key]
                compact[key] = self._limit_text(value, self.max_option_text) if isinstance(value, str) else value

        if not compact:
            compact = {"value": self._limit_text(item, self.max_option_text)}
        return compact

    def build_final_response(
        self,
        *,
        summary: str,
        top_options: List[Dict[str, Any]],
        reasons: List[str],
        cautions: List[str],
        next_step: str,
        next_questions: List[str] | None = None,
    ) -> Dict[str, Any]:
        compact_options = [self.compact_option(item) for item in (top_options or [])[: self.max_top_options]]
        clean_reasons = [self._limit_text(item, self.max_bullet_text) for item in (reasons or [])]
        clean_cautions = [self._limit_text(item, self.max_bullet_text) for item in (cautions or [])]
        clean_questions = [self._limit_text(item, self.max_bullet_text) for item in (next_questions or [])]

        summary_text = self._limit_text(summary, self.max_summary_text)
        next_step_text = self._limit_text(next_step, self.max_bullet_text)

        # Keep legacy keys for backward compatibility while standardizing contract.
        return {
            "summary": summary_text,
            "top_options": compact_options,
            "reasons": clean_reasons,
            "cautions": clean_cautions,
            "next_step": next_step_text,
            "answer": summary_text,
            "recommendations": compact_options,
            "next_questions": clean_questions,
            "citations": [],
        }

    def build_output(
        self,
        *,
        state: str,
        branch: str,
        process_sequence: List[Dict[str, Any]],
        session_id: str | None,
        use_case: str | None,
        matched_signals: List[str],
        retrieval_stats: Dict[str, Any],
        confidence: float,
        fallback_mode: str,
        final_response: Dict[str, Any],
        planner_mode: str = "rule_only",
        llm_used_for_routing: bool = False,
        llm_used_for_response: bool = False,
        llm_model: str = "",
        selected_tool: str | None = None,
    ) -> Dict[str, Any]:
        logger.debug("formatter_branch state=%s branch=%s use_case=%s", state, branch, use_case)
        return {
            "state": state,
            "branch": branch,
            "process_sequence": process_sequence,
            "final_response": final_response,
            "metadata": {
                "sessionId": session_id,
                "schemaVersion": "agent_runtime.v1",
                "branch": branch,
                "use_case": use_case,
                "matched_signals": matched_signals,
                "retrieval_stats": retrieval_stats,
                "confidence": confidence,
                "planner_mode": planner_mode,
                "llm_used_for_routing": llm_used_for_routing,
                "llm_used_for_response": llm_used_for_response,
                "llm_model": llm_model,
                "selected_tool": selected_tool,
                "safety": {
                    "sql_free_form_used": False,
                    "fallback_mode": fallback_mode,
                },
            },
        }
