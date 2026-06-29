from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class RetrievalStats:
    retrieval_mode: str
    fallback_reason: str
    requested_top_k: int
    returned_count: int
    applied_filters: Dict[str, Any]
    matched_signals: List[str]
    contract_source: str | None = None
    prefilter_limit: int | None = None
    semantic_candidates_limit: int | None = None
    lexical_candidates_limit: int | None = None
    semantic_weight: float | None = None
    lexical_weight: float | None = None
    dual_match_bonus: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalOutput:
    items: List[Dict[str, Any]]
    retrieval_stats: RetrievalStats
