from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

from agent.common import get_logger, ExecutionMetrics
from agent.data_access import AgentListingDataAccess

from ._ref_utils import normalize_single_listing_ref


logger = get_logger("tool_similar_listings")


def _normalize_item(item: Any) -> Dict[str, Any]:
    if is_dataclass(item):
        return asdict(item)
    if isinstance(item, dict):
        return dict(item)
    return {"value": str(item)}


def run_similar_listings(advisor: Any, args: Dict[str, Any]) -> Dict[str, Any]:





    metrics = ExecutionMetrics()
    
    try:
        metrics.add_step("parse_ref")
        source, listing_id, listing_ref = normalize_single_listing_ref(args)
        top_k = int(args.get("top_k", 10))
        context_query = args.get("context_query") or args.get("user_query") or args.get("query")

        logger.info(
            "normalize_input source=%s listing_id=%s listing_ref=%s top_k=%s has_context=%s",
            source,
            listing_id,
            listing_ref,
            top_k,
            bool(context_query),
        )

        if not source or not listing_id:
            metrics.set_error("invalid_input")
            return {
                "tool": "similar_listings",
                "status": "invalid_input",
                "found": False,
                "listing_ref": listing_ref,
                "items": [],
                "count": 0,
                "message": "Need source and listing_id or listing_ref for similar_listings",
                "execution": metrics.finalize(),
            }

        metrics.add_step("find_similar")
        dal = AgentListingDataAccess(config_path=args.get("config_path"))
        records = dal.similar_listings(
            source=source,
            listing_id=listing_id,
            context_query=context_query,
            top_k=top_k,
        )
        
        metrics.add_step("format_response")
        items = [_normalize_item(item) for item in (records or [])]
        found = len(items) > 0
        status = "ok" if found else "not_found"

        logger.info("branch_decision found=%s count=%d", found, len(items))

        return {
            "tool": "similar_listings",
            "status": status,
            "found": found,
            "source": source,
            "listing_id": listing_id,
            "listing_ref": f"{source}/{listing_id}",
            "context_query": context_query,
            "top_k": top_k,
            "count": len(items),
            "items": items,
            "message": "Similar listings found" if found else "No similar listings found",
            "execution": metrics.finalize(),
        }
    except Exception as exc:
        metrics.set_error(type(exc).__name__)
        logger.exception("similar_listings failed error=%s", exc)
        return {
            "tool": "similar_listings",
            "status": "tool_error",
            "found": False,
            "items": [],
            "count": 0,
            "message": f"Similar search failed: {exc}",
            "execution": metrics.finalize(),
        }
