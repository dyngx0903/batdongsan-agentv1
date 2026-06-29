from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional

from agent.common import parse_user_query
from agent.retrieval.retrieval_service import QueryFilters
from agent.data_access import AgentListingDataAccess


@dataclass
class SearchRetrievalKnobs:
    prefilter_limit: int = 300
    semantic_candidates_limit: Optional[int] = None
    lexical_candidates_limit: Optional[int] = None
    semantic_weight: float = 0.7
    lexical_weight: float = 0.3
    dual_match_bonus: float = 0.0


@dataclass
class SearchListingsInput:
    query: str
    top_k: int = 10
    parsed_filters: Optional[Dict[str, Any]] = None
    retrieval_knobs: Optional[Dict[str, Any]] = None


@dataclass
class SearchListingCard:
    source: Optional[str] = None
    listing_id: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    transaction_type: Optional[str] = None
    property_type: Optional[str] = None
    project: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    ward: Optional[str] = None
    street: Optional[str] = None
    price_text: Optional[str] = None
    area_text: Optional[str] = None
    price_value_vnd: Optional[int] = None
    area_m2: Optional[float] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    floors: Optional[int] = None
    frontage_width_m: Optional[float] = None
    road_access_width_m: Optional[float] = None
    legal_status: Optional[str] = None
    direction: Optional[str] = None
    suitable_for: Optional[str] = None
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    final_score: float = 0.0
    score: float = 0.0
    matched_by: str = "lexical"


@dataclass
class SearchListingsOutput:
    items: List[Dict[str, Any]]
    retrieval_stats: Dict[str, Any]
    applied_filters: Dict[str, Any]
    fallback_reason: Optional[str]
    retrieval_mode: str


def _safe_int(value: Any, default: int, minimum: Optional[int] = None) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = int(default)
    if minimum is not None:
        out = max(int(minimum), out)
    return out


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_query_filters(parsed_filters: Optional[Dict[str, Any]]) -> Optional[QueryFilters]:
    if not parsed_filters or not isinstance(parsed_filters, dict):
        return None

    allowed = {f.name for f in fields(QueryFilters)}
    sanitized: Dict[str, Any] = {}
    for key in allowed:
        if key in parsed_filters:
            sanitized[key] = parsed_filters[key]

    if not sanitized:
        return None
    return QueryFilters(**sanitized)


def _derive_parsed_filters_from_query(query: str) -> Dict[str, Any]:
    parsed = parse_user_query(query)
    hard = parsed.hard_filters
    out: Dict[str, Any] = {}
    for name in [f.name for f in fields(QueryFilters)]:
        value = getattr(hard, name, None)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        out[name] = value
    return out


def _coerce_retrieval_knobs(raw_knobs: Optional[Dict[str, Any]]) -> SearchRetrievalKnobs:
    if not raw_knobs or not isinstance(raw_knobs, dict):
        return SearchRetrievalKnobs()

    default = SearchRetrievalKnobs()
    return SearchRetrievalKnobs(
        prefilter_limit=_safe_int(raw_knobs.get("prefilter_limit"), default.prefilter_limit, minimum=1),
        semantic_candidates_limit=(
            None
            if raw_knobs.get("semantic_candidates_limit") is None
            else _safe_int(raw_knobs.get("semantic_candidates_limit"), default.prefilter_limit, minimum=1)
        ),
        lexical_candidates_limit=(
            None
            if raw_knobs.get("lexical_candidates_limit") is None
            else _safe_int(raw_knobs.get("lexical_candidates_limit"), default.prefilter_limit, minimum=1)
        ),
        semantic_weight=_safe_float(raw_knobs.get("semantic_weight"), default.semantic_weight),
        lexical_weight=_safe_float(raw_knobs.get("lexical_weight"), default.lexical_weight),
        dual_match_bonus=_safe_float(raw_knobs.get("dual_match_bonus"), default.dual_match_bonus),
    )


def _build_card(row: Dict[str, Any]) -> Dict[str, Any]:
    card = SearchListingCard(
        source=row.get("source"),
        listing_id=row.get("listing_id"),
        title=row.get("title"),
        url=row.get("url"),
        transaction_type=row.get("transaction_type"),
        property_type=row.get("property_type"),
        project=row.get("project"),
        city=row.get("city"),
        district=row.get("district"),
        ward=row.get("ward"),
        street=row.get("street"),
        price_text=row.get("price_text"),
        area_text=row.get("area_text"),
        price_value_vnd=row.get("price_value_vnd"),
        area_m2=row.get("area_m2"),
        bedrooms=row.get("bedrooms"),
        bathrooms=row.get("bathrooms"),
        floors=row.get("floors"),
        frontage_width_m=row.get("frontage_width_m"),
        road_access_width_m=row.get("road_access_width_m"),
        legal_status=row.get("legal_status"),
        direction=row.get("direction"),
        suitable_for=row.get("suitable_for"),
        semantic_score=float(row.get("semantic_score") or 0.0),
        lexical_score=float(row.get("lexical_score") or 0.0),
        final_score=float(row.get("final_score", row.get("score", 0.0)) or 0.0),
        score=float(row.get("score", row.get("final_score", 0.0)) or 0.0),
        matched_by=str(row.get("matched_by") or "lexical"),
    )
    return asdict(card)


def search_listings(
    tool_input: SearchListingsInput,
    search_engine: Optional[Any] = None,
    dal: Optional[AgentListingDataAccess] = None,
) -> SearchListingsOutput:
    """
    Deterministic wrapper for listing retrieval.

    This wrapper delegates to AgentListingDataAccess which handles the retrieval
    through ListingHybridSearch, keeping SQL generation centralized in DAL.
    """
    if dal is None:
        dal = AgentListingDataAccess()
    
    query = str(tool_input.query or "").strip()
    top_k = _safe_int(tool_input.top_k, 5, minimum=1)
    
    # For now, we use the DAL's search_listings which handles hybrid search
    # The parsed_filters and retrieval_knobs are kept for future extension
    retrieved = dal.search_listings(query=query, top_k=top_k, parsed_filters=tool_input.parsed_filters)
    
    # Transform the DAL response to our output format
    items = [_build_card(row) for row in retrieved.items]
    
    return SearchListingsOutput(
        items=items,
        retrieval_stats=retrieved.retrieval_stats,
        applied_filters={},  # DAL doesn't expose applied_filters yet
        fallback_reason=retrieved.retrieval_stats.get("fallback_reason"),
        retrieval_mode=retrieved.retrieval_stats.get("retrieval_mode", "lexical"),
    )


def run_search_listings(advisor: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool wrapper for search listings.
    
    Accepts query in args, delegates to DAL for retrieval.
    Returns structured output conforming to tool response contract.
    """
    from agent.common import get_logger, ExecutionMetrics
    
    logger = get_logger("tool_search_listings")
    metrics = ExecutionMetrics()
    
    try:
        metrics.add_step("parse_args")
        query = str(args.get("query") or "").strip()
        top_k = _safe_int(args.get("top_k", 10), 10, minimum=1)
        
        if not query:
            metrics.add_step("validate")
            metrics.set_error("empty_query")
            return {
                "tool": "search_listings",
                "status": "invalid_input",
                "found": False,
                "count": 0,
                "items": [],
                "message": "Empty query",
                "retrieval_stats": {"retrieval_mode": "none"},
                "fallback_mode": "empty_query",
                "execution": metrics.finalize(),
            }
        
        metrics.add_step("validate")
        metrics.add_step("initialize_dal")
        dal = AgentListingDataAccess(config_path=args.get("config_path"))
        parsed_filters = args.get("parsed_filters") if isinstance(args.get("parsed_filters"), dict) else None
        if not parsed_filters:
            parsed_filters = _derive_parsed_filters_from_query(query)
        effective_filters = dict(parsed_filters or {})
        
        metrics.add_step("hybrid_search")
        try:
            retrieved = dal.search_listings(query=query, top_k=top_k, parsed_filters=effective_filters)
        except TypeError as exc:
            # Backward compatibility for test stubs / legacy DAL signatures.
            if "parsed_filters" not in str(exc):
                raise
            retrieved = dal.search_listings(query=query, top_k=top_k)

        fallback_mode = str(retrieved.retrieval_stats.get("fallback_reason") or "none")
        if not retrieved.items and effective_filters:
            relaxation_candidates: List[tuple[str, Dict[str, Any]]] = []
            if effective_filters.get("district"):
                relaxed = dict(effective_filters)
                relaxed.pop("district", None)
                relaxation_candidates.append(("relaxed_district", relaxed))
            if effective_filters.get("property_type"):
                relaxed = dict(effective_filters)
                relaxed.pop("property_type", None)
                relaxation_candidates.append(("relaxed_property_type", relaxed))
            if effective_filters.get("district") and effective_filters.get("property_type"):
                relaxed = dict(effective_filters)
                relaxed.pop("district", None)
                relaxed.pop("property_type", None)
                relaxation_candidates.append(("relaxed_district_property_type", relaxed))

            for candidate_mode, candidate_filters in relaxation_candidates:
                metrics.add_step(f"retry_{candidate_mode}")
                try:
                    relaxed_retrieved = dal.search_listings(
                        query=query,
                        top_k=top_k,
                        parsed_filters=candidate_filters,
                    )
                except TypeError as exc:
                    if "parsed_filters" not in str(exc):
                        raise
                    relaxed_retrieved = dal.search_listings(query=query, top_k=top_k)
                if relaxed_retrieved.items:
                    retrieved = relaxed_retrieved
                    effective_filters = candidate_filters
                    fallback_mode = candidate_mode
                    break
        
        metrics.add_step("build_cards")
        found = len(retrieved.items) > 0
        status = "ok" if found else "not_found"
        items = [_build_card(row) for row in retrieved.items]
        
        metrics.add_step("format_response")
        logger.info("search_listings query=%s found=%s count=%d", query, found, len(items))
        
        return {
            "tool": "search_listings",
            "status": status,
            "found": found,
            "count": len(items),
            "items": items,
            "summary": (
                f"Found {len(items)} nearby-match listings"
                if found and fallback_mode.startswith("relaxed_")
                else (f"Found {len(items)} listings" if found else "No listings found")
            ),
            "message": (
                f"Found {len(items)} nearby-match listings"
                if found and fallback_mode.startswith("relaxed_")
                else (f"Found {len(items)} listings" if found else "No listings found")
            ),
            "retrieval_stats": retrieved.retrieval_stats,
            "applied_filters": effective_filters,
            "fallback_mode": fallback_mode,
            "next_step": (
                "Khong tim thay trong dung khu vuc yeu cau. Ban co the xem cac lua chon lan can hoac dieu chinh quan muc tieu."
                if found and fallback_mode.startswith("relaxed_")
                else "Ban co the xem cac lua chon hien tai hoac bo sung them tieu chi de loc sat hon."
            ),
            "use_case": "house_search",
            "matched_signals": retrieved.retrieval_stats.get("matched_signals", []),
            "execution": metrics.finalize(),
        }
    except Exception as exc:
        metrics.set_error(type(exc).__name__)
        logger.exception("search_listings failed query=%s error=%s", query if 'query' in locals() else "?", exc)
        return {
            "tool": "search_listings",
            "status": "tool_error",
            "found": False,
            "count": 0,
            "items": [],
            "message": f"Search failed: {exc}",
            "retrieval_stats": {},
            "fallback_mode": "tool_error",
            "execution": metrics.finalize(),
        }
