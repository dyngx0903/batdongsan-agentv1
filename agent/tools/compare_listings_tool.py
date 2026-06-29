from __future__ import annotations

from typing import Any, Dict

from agent.common import ExecutionMetrics, get_logger
from agent.data_access import AgentListingDataAccess

from ._ref_utils import normalize_compare_listing_refs


logger = get_logger("tool_compare_listings")


def _derive_compare_status(found: bool, winner: Any, winner_ref: Any) -> str:
    if not found:
        return "not_found"
    if winner == "tie":
        return "tie"
    if winner in {"A", "B"} and winner_ref:
        return "winner"
    return "incomplete"


def _normalize_user_profile(args: Dict[str, Any]) -> Dict[str, Any] | None:
    raw = args.get("user_context")
    if not isinstance(raw, dict):
        raw = args.get("user_profile")
    if not isinstance(raw, dict):
        return None

    profile: Dict[str, Any] = {
        "budget_vnd": raw.get("budget_vnd") or raw.get("budget") or raw.get("max_budget_vnd"),
        "bedrooms_needed": raw.get("bedrooms_needed") or raw.get("min_bedrooms"),
        "location_preference": raw.get("location_preference") or raw.get("district"),
        "commuting_destination": raw.get("commuting_destination") or raw.get("work_location"),
        "priority": raw.get("priority") if isinstance(raw.get("priority"), list) else None,
    }
    compact = {k: v for k, v in profile.items() if v is not None}
    return compact or None


def run_compare_listings(advisor: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    metrics = ExecutionMetrics()

    try:
        metrics.add_step("parse_refs")
        normalized = normalize_compare_listing_refs(args)
        source_a = normalized.get("source_a")
        listing_id_a = normalized.get("listing_id_a")
        source_b = normalized.get("source_b")
        listing_id_b = normalized.get("listing_id_b")
        user_query = str(args.get("user_query") or "").strip() or None
        user_profile = _normalize_user_profile(args)

        logger.info(
            "normalize_input a=%s/%s b=%s/%s has_user_profile=%s",
            source_a,
            listing_id_a,
            source_b,
            listing_id_b,
            bool(user_profile),
        )

        if not (source_a and listing_id_a and source_b and listing_id_b):
            metrics.set_error("invalid_input")
            return {
                "tool": "compare_listings",
                "status": "invalid_input",
                "found": False,
                "compare_status": "invalid_input",
                "message": "Need source_a/listing_id_a and source_b/listing_id_b or listing_ref_a/listing_ref_b",
                "missing_fields": [
                    key
                    for key, value in {
                        "source_a": source_a,
                        "listing_id_a": listing_id_a,
                        "source_b": source_b,
                        "listing_id_b": listing_id_b,
                    }.items()
                    if not value
                ],
                "execution": metrics.finalize(),
            }

        metrics.add_step("compare")
        dal = AgentListingDataAccess(config_path=args.get("config_path"))
        compare_kwargs: Dict[str, Any] = {
            "source_a": source_a,
            "listing_id_a": listing_id_a,
            "source_b": source_b,
            "listing_id_b": listing_id_b,
            "user_query": user_query,
        }
        if user_profile is not None:
            compare_kwargs["user_profile"] = user_profile

        compare = dal.compare_listings(**compare_kwargs)

        metrics.add_step("format_response")
        raw = {
            "found": compare.found,
            "listing_a": compare.listing_a,
            "listing_b": compare.listing_b,
            "recommendation": compare.recommendation,
            "message": compare.message,
        }

        recommendation = raw.get("recommendation") or {}
        winner = recommendation.get("winner")
        winner_ref = recommendation.get("winner_ref")
        found = bool(raw.get("found"))

        compare_status = _derive_compare_status(found, winner, winner_ref)
        status = "ok" if compare_status in {"winner", "tie"} else compare_status

        logger.info(
            "branch_decision found=%s winner=%s winner_ref=%s compare_status=%s",
            found,
            winner,
            winner_ref,
            compare_status,
        )

        stable = {
            "tool": "compare_listings",
            "status": status,
            "compare_status": compare_status,
            "found": found,
            "listing_ref_a": f"{source_a}/{listing_id_a}",
            "listing_ref_b": f"{source_b}/{listing_id_b}",
            "winner": winner,
            "winner_ref": winner_ref,
            "user_profile_used": bool(recommendation.get("user_profile_used")),
            "message": raw.get("message") or recommendation.get("summary"),
            "execution": metrics.finalize(),
        }

        return {
            **raw,
            **stable,
        }
    except Exception as exc:
        metrics.set_error(type(exc).__name__)
        logger.exception("compare_listings failed error=%s", exc)
        return {
            "tool": "compare_listings",
            "status": "tool_error",
            "found": False,
            "compare_status": "tool_error",
            "message": f"Compare failed: {exc}",
            "execution": metrics.finalize(),
        }
