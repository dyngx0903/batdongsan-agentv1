from __future__ import annotations

from typing import Any, Dict

from agent.common import get_logger, ExecutionMetrics
from agent.data_access import AgentListingDataAccess

from ._ref_utils import normalize_single_listing_ref


logger = get_logger("tool_explain_listing")


def run_explain_listing(advisor: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    metrics = ExecutionMetrics()
    
    try:
        metrics.add_step("validate_and_parse_ref")
        source, listing_id, listing_ref = normalize_single_listing_ref(args)
        user_query = str(args.get("user_query") or "General search").strip() or "General search"

        logger.info(
            "normalize_input source=%s listing_id=%s listing_ref=%s",
            source,
            listing_id,
            listing_ref,
        )

        if not source or not listing_id:
            metrics.set_error("invalid_input")
            return {
                "tool": "explain_listing",
                "status": "invalid_input",
                "found": False,
                "listing_ref": listing_ref,
                "message": "Need source and listing_id or listing_ref for explain_listing",
                "execution": metrics.finalize(),
            }

        metrics.add_step("load_listing")
        dal = AgentListingDataAccess(config_path=args.get("config_path"))
        explain = dal.explain_listing(
            source=source,
            listing_id=listing_id,
            user_query=user_query,
        )
        
        metrics.add_step("analyze_and_score")
        raw = dict(explain.listing) if explain.found else {"found": False, "message": explain.message}
        if explain.found:
            raw["found"] = True

        structured_analysis = getattr(explain, "analysis", None) or raw.get("analysis_structured") or {}

        found = bool(explain.found)
        status = "ok" if found else "not_found"

        if not isinstance(raw, dict):
            raw = {
                "found": False,
                "message": "Unexpected explain_listing response",
            }
            status = "internal_error"
            metrics.set_error("response_type_error")

        logger.info("branch_decision found=%s status=%s", found, status)

        metrics.add_step("format_response")
        stable = {
            "tool": "explain_listing",
            "status": status,
            "found": found,
            "listing_ref": f"{source}/{listing_id}",
            "listing": {
                "source": source,
                "listing_id": listing_id,
                "title": raw.get("title"),
                "url": raw.get("url"),
            },
            "analysis": {
                "matched_hard_filters": raw.get("matched_hard_filters") or [],
                "matched_soft_preferences": raw.get("matched_soft_preferences") or [],
                "enrichment_matches": raw.get("enrichment_matches") or [],
                "similarity_summary": raw.get("similarity_summary"),
                "final_score": raw.get("final_score"),
                "base_score": raw.get("base_score"),
                "intent_bonus": raw.get("intent_bonus"),
                "use_case": raw.get("use_case"),
                "price_analysis": structured_analysis.get("price_analysis") or raw.get("price_analysis") or {},
                "location_analysis": structured_analysis.get("location_analysis") or raw.get("location_analysis") or {},
                "size_analysis": structured_analysis.get("size_analysis") or raw.get("size_analysis") or {},
                "legal_analysis": structured_analysis.get("legal_analysis") or raw.get("legal_analysis") or {},
                "utilities": structured_analysis.get("utilities") or raw.get("utilities") or [],
                "fit_score_breakdown": structured_analysis.get("fit_score_breakdown") or raw.get("fit_score_breakdown") or {},
            },
            "message": raw.get("message") or ("Listing explained" if found else "Listing not found"),
            "execution": metrics.finalize(),
        }

        # Keep backward-compatible top-level keys used by existing formatter/runtime.
        return {
            **raw,
            **stable,
            "source": source,
            "listing_id": listing_id,
        }
    except Exception as exc:
        metrics.set_error(type(exc).__name__)
        logger.exception("explain_listing failed error=%s", exc)
        return {
            "tool": "explain_listing",
            "status": "tool_error",
            "found": False,
            "message": f"Explain failed: {exc}",
            "execution": metrics.finalize(),
        }
