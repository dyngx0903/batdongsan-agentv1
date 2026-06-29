from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from utils.alias_registry import get_aliases
from utils.vn_normalizer import normalize_text

from agent.common import get_logger, load_config, parse_user_query
from .db_gateway import RetrievalDbGateway
from .search_contract import RetrievalOutput, RetrievalStats


@dataclass
class QueryFilters:
    transaction_type: Optional[str] = None
    property_type: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    ward: Optional[str] = None
    street: Optional[str] = None
    project: Optional[str] = None
    min_price_vnd: Optional[int] = None
    max_price_vnd: Optional[int] = None
    min_area_m2: Optional[float] = None
    max_area_m2: Optional[float] = None
    min_bedrooms: Optional[int] = None
    min_bathrooms: Optional[int] = None
    legal_status: Optional[str] = None
    direction: Optional[str] = None
    min_floors: Optional[int] = None
    min_frontage_width_m: Optional[float] = None
    min_road_access_width_m: Optional[float] = None


class ListingHybridSearch:
    TABLE_NAME = "listings"

    # Alias maps for property_type expansion (based on crawled db values).
    PROPERTY_TYPE_ALIASES = {
        "Chung cư": ["Chung cư", "Chung cư mini / Căn hộ dịch vụ", "Căn hộ dịch vụ"],
        "Nhà riêng": ["Nhà riêng", "Villa", "Biet thu"],
        "Nhà phố": ["Nhà phố", "Shophouse / Nhà phố thương mại", "Shop House"],
        "Đất": ["Đất", "Đất nền dự án", "Đất nền"],
    }

    def __init__(self, config_path: Optional[str] = None, db_gateway: Optional[RetrievalDbGateway] = None):
        self.logger = get_logger("retrieval")
        self.config_path = self._resolve_config_path(config_path)
        self.db_config = self._load_db_config(self.config_path)
        self.db_gateway = db_gateway or RetrievalDbGateway(config_path=config_path)
        self._unaccent_available: Optional[bool] = None

    def _resolve_config_path(self, config_path: Optional[str]) -> str:
        if config_path:
            return str(config_path)

        config_dir = Path(__file__).resolve().parents[2] / "CONFIG"
        return str(config_dir / "global.yaml")

    def _load_db_config(self, config_path: str) -> Dict[str, Any]:
        cfg = load_config(config_path)
        db = cfg.get("db", {}) if isinstance(cfg, dict) else {}
        required = ["host", "port", "user", "password", "dbname"]
        missing = [k for k in required if k not in db]
        if missing:
            raise ValueError(f"Missing DB config keys in {config_path}: {missing}")
        return db

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(
            host=self.db_config["host"],
            port=self.db_config["port"],
            user=self.db_config["user"],
            password=self.db_config["password"],
            dbname=self.db_config["dbname"],
            options="-c client_encoding=UTF8",
        )

    def parse_query_filters(self, query: str) -> QueryFilters:
        parsed = parse_user_query(query)
        hard = parsed.hard_filters
        return QueryFilters(
            transaction_type=hard.transaction_type,
            property_type=hard.property_type,
            city=hard.city,
            district=hard.district,
            ward=hard.ward,
            street=hard.street,
            project=hard.project,
            min_price_vnd=hard.min_price_vnd,
            max_price_vnd=hard.max_price_vnd,
            min_area_m2=hard.min_area_m2,
            min_bedrooms=hard.min_bedrooms,
            min_bathrooms=hard.min_bathrooms,
            legal_status=hard.legal_status,
            direction=hard.direction,
            min_floors=hard.min_floors,
            min_frontage_width_m=hard.min_frontage_width_m,
            min_road_access_width_m=hard.min_road_access_width_m,
        )

    def _expand_property_type_aliases(self, property_type: Optional[str]) -> List[str]:
        """Expand property_type to all known aliases in database."""
        if not property_type:
            return []

        aliases_from_registry = get_aliases("property_type", property_type)
        if aliases_from_registry:
            merged = [property_type]
            for alias in aliases_from_registry:
                if alias not in merged:
                    merged.append(alias)
            return merged

        # Check if property_type matches a key in our alias map.
        for key, aliases in self.PROPERTY_TYPE_ALIASES.items():
            if property_type.lower() in [a.lower() for a in aliases]:
                return aliases
        # If not found, return as-is (fallback for unmapped types).
        return [property_type]

    def _expand_transaction_type_aliases(self, transaction_type: Optional[str]) -> List[str]:
        if not transaction_type:
            return []
        aliases = [transaction_type]
        for alias in get_aliases("transaction_type", transaction_type):
            if alias not in aliases:
                aliases.append(alias)
        if normalize_text(transaction_type) == "thue" and "cho thuê" not in aliases:
            aliases.append("cho thuê")
        return aliases

    def _expand_city_aliases(self, city: Optional[str]) -> List[str]:
        if not city:
            return []
        aliases = [city]
        for alias in get_aliases("city", city):
            if alias not in aliases:
                aliases.append(alias)
        return aliases

    def _supports_unaccent(self) -> bool:
        if self._unaccent_available is not None:
            return self._unaccent_available
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'unaccent')")
                    row = cur.fetchone()
                    self._unaccent_available = bool(row and row[0])
        except Exception as exc:
            self.logger.warning("Could not determine unaccent extension availability: %s", exc)
            self._unaccent_available = False
        return bool(self._unaccent_available)

    def _build_text_match_clause(self, column: str, values: List[str], params: List[Any]) -> Optional[str]:
        unique_values: List[str] = []
        for value in values:
            value_text = str(value or "").strip()
            if value_text and value_text not in unique_values:
                unique_values.append(value_text)

        if not unique_values:
            return None

        clauses: List[str] = []
        if self._supports_unaccent():
            for value in unique_values:
                clauses.append(f"unaccent(lower(COALESCE({column}, ''))) LIKE unaccent(lower(%s))")
                params.append(f"%{value}%")
            return "(" + " OR ".join(clauses) + ")"

        seen_fallback_patterns: set[str] = set()
        for value in unique_values:
            candidates = [str(value).lower(), normalize_text(value)]
            for candidate in candidates:
                pattern = f"%{candidate}%"
                if pattern in seen_fallback_patterns:
                    continue
                seen_fallback_patterns.add(pattern)
                clauses.append(f"lower(COALESCE({column}, '')) LIKE %s")
                params.append(pattern)
        return "(" + " OR ".join(clauses) + ")" if clauses else None

    def _expand_district_aliases(self, district: Optional[str]) -> List[str]:
        """Expand district shorthand to canonical forms (e.g., Q2 -> Quận 2)."""
        if not district:
            return []

        aliases_from_registry = get_aliases("district", district)
        if aliases_from_registry:
            merged = [district]
            for alias in aliases_from_registry:
                if alias not in merged:
                    merged.append(alias)
            return merged

        # Simple mapping for common shorthands.
        expansions = {
            "Quận 1": ["Quận 1", "Q1", "Q 1"],
            "Quận 2": ["Quận 2", "Q2", "Q 2"],
            "Quận 3": ["Quận 3", "Q3", "Q 3"],
            "Quận 4": ["Quận 4", "Q4", "Q 4"],
            "Quận 5": ["Quận 5", "Q5", "Q 5"],
            "Quận 6": ["Quận 6", "Q6", "Q 6"],
            "Quận 7": ["Quận 7", "Q7", "Q 7"],
            "Quận 8": ["Quận 8", "Q8", "Q 8"],
            "Quận 9": ["Quận 9", "Q9", "Q 9"],
            "Quận 10": ["Quận 10", "Q10", "Q 10"],
            "Quận 11": ["Quận 11", "Q11", "Q 11"],
            "Quận 12": ["Quận 12", "Q12", "Q 12"],
            "Thủ Đức": ["Thủ Đức", "Thu Duc"],
        }
        for key, aliases in expansions.items():
            if district.lower() in [a.lower() for a in aliases]:
                return aliases
        # If not found, return as-is.
        return [district]

    def _expand_ward_aliases(self, ward: Optional[str]) -> List[str]:
        """Expand ward shorthand forms (e.g., Phường 5 <-> P5)."""
        if not ward:
            return []
        aliases_from_registry = get_aliases("ward", ward)
        if aliases_from_registry:
            merged = [ward]
            for alias in aliases_from_registry:
                if alias not in merged:
                    merged.append(alias)
            return merged
        match = re.search(r"(\d+)", ward)
        if not match:
            return [ward]
        number = match.group(1)
        return [f"Phường {number}", f"P.{number}", f"P{number}"]

    def _project_terms(self, project: Optional[str]) -> List[str]:
        if not project:
            return []
        tokens = [tok.strip() for tok in re.split(r"\s+", project) if len(tok.strip()) >= 4]
        dedup: List[str] = []
        for tok in tokens:
            lower = tok.lower()
            if lower not in [x.lower() for x in dedup]:
                dedup.append(tok)
        return dedup[:4]

    def _tokenize(self, text: str) -> List[str]:
        return [tok for tok in re.findall(r"\w+", (text or "").lower()) if len(tok) > 1]

    def _lexical_score(self, query: str, document: str) -> float:
        q_tokens = set(self._tokenize(query))
        d_tokens = set(self._tokenize(document))
        if not q_tokens or not d_tokens:
            return 0.0
        overlap = q_tokens.intersection(d_tokens)
        return float(len(overlap) / max(1, len(q_tokens)))

    def _to_vector_literal(self, embedding: Optional[List[float]]) -> Optional[str]:
        if not embedding:
            return None
        try:
            return "[" + ",".join(str(float(x)) for x in embedding) + "]"
        except (TypeError, ValueError):
            return None

    def build_where_clause(self, filters: QueryFilters) -> Tuple[str, List[Any]]:
        where = ["search_document IS NOT NULL", "BTRIM(search_document) <> ''"]
        params: List[Any] = []

        if filters.transaction_type:
            aliases = self._expand_transaction_type_aliases(filters.transaction_type)
            clause = self._build_text_match_clause("transaction_type", aliases, params)
            if clause:
                where.append(clause)
        if filters.property_type:
            aliases = self._expand_property_type_aliases(filters.property_type)
            clause = self._build_text_match_clause("property_type", aliases, params)
            if clause:
                where.append(clause)
        if filters.city:
            aliases = self._expand_city_aliases(filters.city)
            clause = self._build_text_match_clause("city", aliases, params)
            if clause:
                where.append(clause)
        if filters.district:
            aliases = self._expand_district_aliases(filters.district)
            clause = self._build_text_match_clause("district", aliases, params)
            if clause:
                where.append(clause)
        if filters.ward:
            aliases = self._expand_ward_aliases(filters.ward)
            clause = self._build_text_match_clause("ward", aliases, params)
            if clause:
                where.append(clause)
        if filters.street:
            street_like = f"%{filters.street.strip()}%"
            where.append("(street ILIKE %s OR search_document ILIKE %s)")
            params.extend([street_like, street_like])
        if filters.project:
            terms = self._project_terms(filters.project)
            base_like = f"%{filters.project.strip()}%"
            project_clauses = [
                "project ILIKE %s",
                "title ILIKE %s",
                "search_document ILIKE %s",
            ]
            params.extend([base_like, base_like, base_like])
            for term in terms:
                project_clauses.append("project ILIKE %s")
                project_clauses.append("title ILIKE %s")
                project_clauses.append("search_document ILIKE %s")
                term_like = f"%{term}%"
                params.extend([term_like, term_like, term_like])
            where.append("(" + " OR ".join(project_clauses) + ")")
        if filters.min_price_vnd is not None:
            where.append("price_value_vnd >= %s")
            params.append(filters.min_price_vnd)
        if filters.max_price_vnd is not None:
            # price_value_vnd is guaranteed not null by db schema; safe to relax filter.
            where.append("price_value_vnd <= %s")
            params.append(filters.max_price_vnd)
        if filters.min_area_m2 is not None:
            # area_m2 is guaranteed not null by db schema; safe to relax filter.
            where.append("area_m2 >= %s")
            params.append(filters.min_area_m2)
        if filters.max_area_m2 is not None:
            # area_m2 is guaranteed not null by db schema; safe to relax filter.
            where.append("area_m2 <= %s")
            params.append(filters.max_area_m2)
        if filters.min_bedrooms is not None:
            # bedrooms can be null; keep the check to avoid matching listings without bedroom count.
            where.append("bedrooms IS NOT NULL AND bedrooms >= %s")
            params.append(filters.min_bedrooms)
        if filters.min_bathrooms is not None:
            where.append("bathrooms IS NOT NULL AND bathrooms >= %s")
            params.append(filters.min_bathrooms)
        if filters.legal_status:
            where.append("legal_status ILIKE %s")
            params.append(f"%{filters.legal_status}%")
        if filters.direction:
            # direction is stored as text; use substring match to support JSON-like string payload.
            where.append("direction ILIKE %s")
            params.append(f"%{filters.direction}%")
        if filters.min_floors is not None:
            where.append("floors IS NOT NULL AND floors >= %s")
            params.append(filters.min_floors)
        if filters.min_frontage_width_m is not None:
            where.append("frontage_width_m IS NOT NULL AND frontage_width_m >= %s")
            params.append(filters.min_frontage_width_m)
        if filters.min_road_access_width_m is not None:
            where.append("road_access_width_m IS NOT NULL AND road_access_width_m >= %s")
            params.append(filters.min_road_access_width_m)

        return " AND ".join(where), params

    def _build_where_clause(self, filters: QueryFilters) -> Tuple[str, List[Any]]:
        return self.build_where_clause(filters)

    def get_query_embedding(self, query: str) -> Optional[str]:
        if not self.db_gateway:
            return None
        try:
            embedding = self.db_gateway.get_query_embedding(query)
            return self._to_vector_literal(embedding) if embedding else None
        except Exception as e:
            self.logger.warning("Failed to get query embedding: %s", e)
            return None

    @staticmethod
    def _bounded_score(value: Any, lower: float = 0.0, upper: float = 1.0) -> float:
        try:
            numeric = float(value or 0.0)
        except (TypeError, ValueError):
            numeric = 0.0
        if numeric < lower:
            return lower
        if numeric > upper:
            return upper
        return numeric

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Safely convert any value to float, returning None if conversion fails."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_score_for_fusion(value: Any) -> float:
        """
        Bounded normalization for heterogeneous score scales.
        - Keep [0, 1] unchanged.
        - Compress values > 1 into (0, 1) via x / (1 + x).
        """
        try:
            numeric = float(value or 0.0)
        except (TypeError, ValueError):
            numeric = 0.0
        if numeric <= 0.0:
            return 0.0
        if numeric <= 1.0:
            return numeric
        return numeric / (1.0 + numeric)

    @staticmethod
    def _normalize_weights(semantic_weight: float, lexical_weight: float) -> Tuple[float, float]:
        sem = max(0.0, float(semantic_weight or 0.0))
        lex = max(0.0, float(lexical_weight or 0.0))
        total = sem + lex
        if total <= 0.0:
            return 0.5, 0.5
        return sem / total, lex / total

    def fetch_candidates(
        self,
        query: str,
        where_sql: str,
        params: List[Any],
        query_vec: Optional[str],
        prefilter_limit: int,
        semantic_candidates_limit: int,
        lexical_candidates_limit: int,
    ) -> List[Dict[str, Any]]:
        lexical_rows = self._mark_rows_matched_by(
            self.fetch_lexical_candidates(
                query=query,
                where_sql=where_sql,
                params=params,
                lexical_candidates_limit=lexical_candidates_limit,
            ),
            matched_by="lexical",
        )

        if query_vec:
            try:
                semantic_rows = self._mark_rows_matched_by(
                    self.fetch_semantic_candidates(
                        query=query,
                        where_sql=where_sql,
                        params=params,
                        query_vec=query_vec,
                        semantic_candidates_limit=semantic_candidates_limit,
                    ),
                    matched_by="semantic",
                )
                return self.merge_candidates(semantic_rows, lexical_rows)
            except Exception as exc:
                # Semantic retrieval failed; gracefully degrade to lexical-only.
                self.logger.warning(
                    "Semantic retrieval failed; using lexical-only fallback. query=%r error=%s",
                    query,
                    exc,
                )
                if lexical_rows:
                    return lexical_rows
                return self._mark_rows_matched_by(
                    self.fetch_fallback_candidates(where_sql, params, prefilter_limit),
                    matched_by="lexical",
                )

        if lexical_rows:
            return lexical_rows
        return self._mark_rows_matched_by(
            self.fetch_fallback_candidates(where_sql, params, prefilter_limit),
            matched_by="lexical",
        )

    def _mark_rows_matched_by(
        self,
        rows: List[Dict[str, Any]],
        matched_by: str,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            record["matched_by"] = matched_by
            out.append(record)
        return out

    def fetch_semantic_candidates(
        self,
        query: str,
        where_sql: str,
        params: List[Any],
        query_vec: str,
        semantic_candidates_limit: int,
    ) -> List[Dict[str, Any]]:
        sql = f"""
            SELECT
                source,
                listing_id,
                url,
                title,
                transaction_type,
                property_type,
                project,
                city,
                district,
                ward,
                street,
                price_text,
                area_text,
                price_value_vnd,
                area_m2,
                legal_status,
                bedrooms,
                bathrooms,
                floors,
                frontage_width_m,
                road_access_width_m,
                direction,
                search_document,
                1 - (search_document_embedding <=> %s::vector) AS semantic_score,
                ts_rank_cd(
                    to_tsvector('simple', COALESCE(search_document, '')),
                    websearch_to_tsquery('simple', %s)
                ) AS lexical_score
            FROM {self.TABLE_NAME}
            WHERE {where_sql}
              AND search_document_embedding IS NOT NULL
            ORDER BY search_document_embedding <=> %s::vector ASC
            LIMIT %s
        """

        rows: List[Dict[str, Any]] = []
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql,
                        tuple(
                            [query_vec, query]
                            + params
                            + [
                                query_vec,
                                max(1, int(semantic_candidates_limit)),
                            ]
                        ),
                    )
                    columns = [d[0] for d in cur.description] if cur.description else []
                    for raw_row in cur.fetchall():
                        rows.append(dict(zip(columns, raw_row)))
        except psycopg.Error:
            # Defensive fallback when tsquery parsing fails.
            fallback_sql = f"""
                SELECT
                    source,
                    listing_id,
                    url,
                    title,
                    transaction_type,
                    property_type,
                    project,
                    city,
                    district,
                    ward,
                    street,
                    price_text,
                    area_text,
                    price_value_vnd,
                    area_m2,
                    legal_status,
                    bedrooms,
                    bathrooms,
                    floors,
                    frontage_width_m,
                    road_access_width_m,
                    direction,
                    search_document,
                    1 - (search_document_embedding <=> %s::vector) AS semantic_score,
                    0.0 AS lexical_score
                FROM {self.TABLE_NAME}
                WHERE {where_sql}
                  AND search_document_embedding IS NOT NULL
                ORDER BY search_document_embedding <=> %s::vector ASC
                LIMIT %s
            """
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        fallback_sql,
                        tuple(
                            [query_vec]
                            + params
                            + [
                                query_vec,
                                max(1, int(semantic_candidates_limit)),
                            ]
                        ),
                    )
                    columns = [d[0] for d in cur.description] if cur.description else []
                    for raw_row in cur.fetchall():
                        rows.append(dict(zip(columns, raw_row)))
        return rows

    def fetch_lexical_candidates(
        self,
        query: str,
        where_sql: str,
        params: List[Any],
        lexical_candidates_limit: int,
    ) -> List[Dict[str, Any]]:
        if not (query or "").strip():
            return []

        def _run_lexical_query(tsquery_fn: str) -> List[Dict[str, Any]]:
            sql = f"""
                SELECT
                    source,
                    listing_id,
                    url,
                    title,
                    transaction_type,
                    property_type,
                    project,
                    city,
                    district,
                    ward,
                    street,
                    price_text,
                    area_text,
                    price_value_vnd,
                    area_m2,
                    legal_status,
                    bedrooms,
                    bathrooms,
                    floors,
                    frontage_width_m,
                    road_access_width_m,
                    direction,
                    search_document,
                    0.0 AS semantic_score,
                    ts_rank_cd(
                        to_tsvector('simple', COALESCE(search_document, '')),
                        {tsquery_fn}('simple', %s)
                    ) AS lexical_score
                FROM {self.TABLE_NAME}
                WHERE {where_sql}
                  AND to_tsvector('simple', COALESCE(search_document, '')) @@ {tsquery_fn}('simple', %s)
                ORDER BY lexical_score DESC, updated_at DESC NULLS LAST
                LIMIT %s
            """
            out: List[Dict[str, Any]] = []
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql,
                        tuple(
                            [query]
                            + params
                            + [
                                query,
                                max(1, int(lexical_candidates_limit)),
                            ]
                        ),
                    )
                    columns = [d[0] for d in cur.description] if cur.description else []
                    for raw_row in cur.fetchall():
                        out.append(dict(zip(columns, raw_row)))
            return out

        try:
            return _run_lexical_query("websearch_to_tsquery")
        except psycopg.Error:
            try:
                return _run_lexical_query("plainto_tsquery")
            except psycopg.Error:
                return self.fetch_fallback_candidates(
                    where_sql=where_sql,
                    params=params,
                    prefilter_limit=lexical_candidates_limit,
                )

    def merge_candidates(
        self,
        semantic_rows: List[Dict[str, Any]],
        lexical_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: Dict[Tuple[Any, Any], Dict[str, Any]] = {}

        for record in semantic_rows + lexical_rows:
            key = (record.get("source"), record.get("listing_id"))
            if key not in merged:
                merged[key] = dict(record)
                continue

            existing = merged[key]
            existing["semantic_score"] = max(
                float(existing.get("semantic_score") or 0.0),
                float(record.get("semantic_score") or 0.0),
            )
            existing["lexical_score"] = max(
                float(existing.get("lexical_score") or 0.0),
                float(record.get("lexical_score") or 0.0),
            )
            current_match = str(existing.get("matched_by") or "")
            incoming_match = str(record.get("matched_by") or "")
            if current_match != incoming_match and incoming_match:
                existing["matched_by"] = "both"
            if not existing.get("search_document") and record.get("search_document"):
                existing["search_document"] = record.get("search_document")
            if not existing.get("url") and record.get("url"):
                existing["url"] = record.get("url")

        for record in merged.values():
            if record.get("matched_by") not in {"semantic", "lexical", "both"}:
                record["matched_by"] = "lexical"

        return list(merged.values())

    def fetch_fallback_candidates(
        self,
        where_sql: str,
        params: List[Any],
        prefilter_limit: int,
    ) -> List[Dict[str, Any]]:
        semantic_select = "0.0 AS semantic_score"

        sql = f"""
            SELECT
                source,
                listing_id,
                url,
                title,
                transaction_type,
                property_type,
                project,
                city,
                district,
                ward,
                street,
                price_text,
                area_text,
                price_value_vnd,
                area_m2,
                legal_status,
                bedrooms,
                bathrooms,
                floors,
                frontage_width_m,
                road_access_width_m,
                direction,
                search_document,
                {semantic_select},
                0.0 AS lexical_score
            FROM {self.TABLE_NAME}
            WHERE {where_sql}
            ORDER BY updated_at DESC NULLS LAST
            LIMIT %s
        """

        rows: List[Dict[str, Any]] = []
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params + [max(1, int(prefilter_limit))]))
                columns = [d[0] for d in cur.description] if cur.description else []
                for raw_row in cur.fetchall():
                    rows.append(dict(zip(columns, raw_row)))
        return rows

    def compute_scores(
        self,
        rows: List[Dict[str, Any]],
        query_vec: Optional[str],
        retrieval_mode: str,
        semantic_weight: float,
        lexical_weight: float,
        dual_match_bonus: float = 0.0,
    ) -> List[Dict[str, Any]]:
        scored_rows: List[Dict[str, Any]] = []
        sem_weight, lex_weight = self._normalize_weights(semantic_weight, lexical_weight)
        for record in rows:
            sem_score = self._normalize_score_for_fusion(record.get("semantic_score"))
            lex_score = self._normalize_score_for_fusion(record.get("lexical_score"))
            matched_by = str(record.get("matched_by") or "lexical")

            if query_vec is None:
                final_score = lex_score
            else:
                final_score = (sem_weight * sem_score) + (lex_weight * lex_score)
                if matched_by == "both":
                    final_score += float(dual_match_bonus or 0.0)
                final_score = self._bounded_score(final_score)

            record["semantic_score"] = sem_score
            record["lexical_score"] = lex_score
            record["matched_by"] = matched_by
            record["final_score"] = final_score
            record["score"] = final_score
            record["retrieval_mode"] = retrieval_mode
            scored_rows.append(record)
        return scored_rows

    def format_results(self, rows: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        rows.sort(key=lambda x: x.get("final_score", x.get("score", 0.0)), reverse=True)
        return rows[: max(1, int(top_k))]

    def search(
        self,
        query: str,
        top_k: int = 10,
        prefilter_limit: int = 300,
        semantic_candidates_limit: Optional[int] = None,
        lexical_candidates_limit: Optional[int] = None,
        filters: Optional[QueryFilters] = None,
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3,
        dual_match_bonus: float = 0.0,
    ) -> List[Dict[str, Any]]:
        rows, _meta = self.search_with_meta(
            query=query,
            top_k=top_k,
            prefilter_limit=prefilter_limit,
            semantic_candidates_limit=semantic_candidates_limit,
            lexical_candidates_limit=lexical_candidates_limit,
            filters=filters,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
            dual_match_bonus=dual_match_bonus,
        )
        return rows

    def search_with_meta(
        self,
        query: str,
        top_k: int = 10,
        prefilter_limit: int = 300,
        semantic_candidates_limit: Optional[int] = None,
        lexical_candidates_limit: Optional[int] = None,
        filters: Optional[QueryFilters] = None,
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3,
        dual_match_bonus: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        active_filters = filters or self.parse_query_filters(query)
        where_sql, params = self.build_where_clause(active_filters)
        query_vec: Optional[str] = None
        retrieval_mode = "lexical_only"
        fallback_reason: Optional[str] = None
        try:
            query_vec = self.get_query_embedding(query)
        except Exception as exc:
            self.logger.warning(
                "Query embedding failed; using lexical-only fallback. query=%r error=%s",
                query,
                exc,
            )
            query_vec = None
            fallback_reason = "embedding_error"

        if query_vec:
            retrieval_mode = "hybrid"
        else:
            if fallback_reason is None:
                fallback_reason = "embedding_unavailable"
            self.logger.info(
                "Query embedding unavailable; using lexical-only retrieval. query=%r reason=%s",
                query,
                fallback_reason,
            )
        semantic_limit = (
            max(1, int(semantic_candidates_limit))
            if semantic_candidates_limit is not None
            else max(1, int(prefilter_limit))
        )
        lexical_limit = (
            max(1, int(lexical_candidates_limit))
            if lexical_candidates_limit is not None
            else max(1, int(prefilter_limit))
        )
        rows = self.fetch_candidates(
            query=query,
            where_sql=where_sql,
            params=params,
            query_vec=query_vec,
            prefilter_limit=max(1, int(prefilter_limit)),
            semantic_candidates_limit=semantic_limit,
            lexical_candidates_limit=lexical_limit,
        )
        scored_rows = self.compute_scores(
            rows,
            query_vec,
            retrieval_mode,
            semantic_weight,
            lexical_weight,
            dual_match_bonus=dual_match_bonus,
        )
        if retrieval_mode == "lexical_only":
            for row in scored_rows:
                row["fallback_reason"] = fallback_reason
        formatted_rows = self.format_results(scored_rows, top_k)
        return formatted_rows, {
            "retrieval_mode": retrieval_mode,
            "fallback_reason": fallback_reason,
            "applied_filters": active_filters,
            "top_k": max(1, int(top_k)),
            "prefilter_limit": max(1, int(prefilter_limit)),
            "semantic_candidates_limit": semantic_limit,
            "lexical_candidates_limit": lexical_limit,
            "semantic_weight": float(semantic_weight),
            "lexical_weight": float(lexical_weight),
            "dual_match_bonus": float(dual_match_bonus),
            "returned_count": len(formatted_rows),
        }

    def search_listings(
        self,
        query: str,
        top_k: int = 10,
        prefilter_limit: int = 300,
        semantic_candidates_limit: Optional[int] = None,
        lexical_candidates_limit: Optional[int] = None,
        filters: Optional[QueryFilters] = None,
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3,
        dual_match_bonus: float = 0.0,
    ) -> RetrievalOutput:
        """
        Search listings and return a properly formatted RetrievalOutput with statistics.
        """
        rows, meta = self.search_with_meta(
            query=query,
            top_k=top_k,
            prefilter_limit=prefilter_limit,
            semantic_candidates_limit=semantic_candidates_limit,
            lexical_candidates_limit=lexical_candidates_limit,
            filters=filters,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
            dual_match_bonus=dual_match_bonus,
        )
        
        # Convert applied_filters to dict if it's a QueryFilters object
        applied_filters = meta.get("applied_filters", {})
        if hasattr(applied_filters, '__dataclass_fields__'):
            # It's a dataclass, convert to dict
            applied_filters = asdict(applied_filters)
        elif not isinstance(applied_filters, dict):
            applied_filters = {}
        
        retrieval_stats = RetrievalStats(
            retrieval_mode=meta.get("retrieval_mode", "lexical_only"),
            fallback_reason=meta.get("fallback_reason") or "none",
            requested_top_k=int(meta.get("top_k") or top_k),
            returned_count=len(rows),
            applied_filters=applied_filters,
            matched_signals=meta.get("matched_signals", []),
            contract_source=meta.get("contract_source"),
            prefilter_limit=prefilter_limit,
            semantic_candidates_limit=semantic_candidates_limit,
            lexical_candidates_limit=lexical_candidates_limit,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
            dual_match_bonus=dual_match_bonus,
        )
        
        return RetrievalOutput(
            items=rows,
            retrieval_stats=retrieval_stats,
        )

    def similar_listings(
        self,
        source: str,
        listing_id: str,
        context_query: Optional[str] = None,
        top_k: int = 10,
        prefilter_limit: int = 400,
        semantic_weight: float = 0.85,
        lexical_weight: float = 0.15,
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        source,
                        listing_id,
                        title,
                        property_type,
                        transaction_type,
                        city,
                        district,
                        ward,
                        street,
                        project,
                        legal_status,
                        direction,
                        search_document,
                        price_value_vnd,
                        area_m2
                    FROM {self.TABLE_NAME}
                    WHERE source = %s AND listing_id = %s
                    LIMIT 1
                    """,
                    (source, listing_id),
                )
                row = cur.fetchone()
                if not row:
                    return []

                columns = [d[0] for d in cur.description] if cur.description else []
                target = dict(zip(columns, row))
                target_doc = str(target.get("search_document") or "")
                target_area = self._safe_float(target.get("area_m2"))

                context_district: Optional[str] = None
                context_min_area_m2: Optional[float] = None
                context_min_area_is_exclusive = False
                context_price_direction: Optional[str] = None
                if context_query:
                    parsed_context = parse_user_query(context_query)
                    context_district = parsed_context.hard_filters.district
                    context_price_direction = parsed_context.hard_filters.price_direction
                    normalized_context = normalize_text(context_query)
                    larger_area_requested = any(
                        token in normalized_context
                        for token in [
                            "dien tich lon hon",
                            "lon hon",
                            "rong hon",
                            "dien tich rong hon",
                        ]
                    )
                    parsed_min_area = self._safe_float(parsed_context.hard_filters.min_area_m2)
                    if parsed_min_area is not None:
                        context_min_area_m2 = parsed_min_area
                        context_min_area_is_exclusive = larger_area_requested
                    elif larger_area_requested and target_area is not None:
                        context_min_area_m2 = target_area
                        context_min_area_is_exclusive = True

                district_to_match = context_district or str(target.get("district") or "").strip()

                cur.execute(
                    f"""
                    SELECT search_document_embedding
                    FROM {self.TABLE_NAME}
                    WHERE source = %s AND listing_id = %s
                    LIMIT 1
                    """,
                    (source, listing_id),
                )
                target_vec_row = cur.fetchone()
                target_vec = target_vec_row[0] if target_vec_row else None

                semantic_select = "0.0 AS semantic_score"
                semantic_params: List[Any] = []
                if target_vec is not None:
                    semantic_select = "CASE WHEN search_document_embedding IS NULL THEN 0.0 ELSE 1 - (search_document_embedding <=> %s::vector) END AS semantic_score"
                    semantic_params = [str(target_vec)]

                params: List[Any] = list(semantic_params)
                where: List[str] = [
                    "source = %s",
                    "listing_id <> %s",
                    "(property_type IS NOT DISTINCT FROM %s OR (%s::text IS NOT NULL AND property_type ILIKE %s))",
                    "transaction_type IS NOT DISTINCT FROM %s",
                    "city IS NOT DISTINCT FROM %s",
                    "search_document IS NOT NULL",
                    "BTRIM(search_document) <> ''",
                ]
                params.extend(
                    [
                        source,
                        listing_id,
                        target.get("property_type"),
                        target.get("property_type"),
                        f"%{target.get('property_type')}%" if target.get("property_type") else None,
                        target.get("transaction_type"),
                        target.get("city"),
                    ]
                )

                if district_to_match:
                    district_aliases = self._expand_district_aliases(district_to_match)
                    district_clause = self._build_text_match_clause("district", district_aliases, params)
                    if district_clause:
                        where.append(district_clause)

                if context_min_area_m2 is not None:
                    if context_min_area_is_exclusive:
                        where.append("area_m2 > %s")
                    else:
                        where.append("area_m2 >= %s")
                    params.append(context_min_area_m2)

                params.append(max(1, int(prefilter_limit)))

                cur.execute(
                    f"""
                    SELECT
                        source,
                        listing_id,
                        url,
                        title,
                        transaction_type,
                        property_type,
                        project,
                        city,
                        district,
                        ward,
                        street,
                        price_text,
                        price_value_vnd,
                        area_text,
                        area_m2,
                        legal_status,
                        bathrooms,
                        floors,
                        frontage_width_m,
                        road_access_width_m,
                        direction,
                        search_document,
                        {semantic_select}
                    FROM {self.TABLE_NAME}
                    WHERE {' AND '.join(where)}
                    ORDER BY updated_at DESC NULLS LAST
                    LIMIT %s
                    """,
                    tuple(params),
                )

                columns = [d[0] for d in cur.description] if cur.description else []
                out: List[Dict[str, Any]] = []
                for candidate_row in cur.fetchall():
                    record = dict(zip(columns, candidate_row))
                    doc = str(record.get("search_document") or "")
                    sem_score = float(record.get("semantic_score") or 0.0)
                    lex_score = self._lexical_score(target_doc, doc)
                    location_score = self._location_similarity_score(target, record)
                    budget_score = self._budget_similarity_score(target, record, tolerance_pct=0.2)
                    price_direction_score = self._price_direction_score(target, record, context_price_direction)
                    record["semantic_score"] = sem_score
                    record["lexical_score"] = lex_score
                    record["location_score"] = location_score
                    record["budget_score"] = budget_score
                    record["price_direction_score"] = price_direction_score
                    base_score = (semantic_weight * sem_score) + (lexical_weight * lex_score)
                    record["score"] = self._bounded_score(
                        base_score + (0.15 * location_score) + (0.1 * budget_score) + (0.12 * price_direction_score)
                    )
                    out.append(record)

                out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
                return out[: max(1, int(top_k))]

    def _budget_similarity_score(self, target: Dict[str, Any], candidate: Dict[str, Any], tolerance_pct: float = 0.2) -> float:
        """
        Calculate budget similarity score based on price range (default ±20%).
        Returns 1.0 if within range, 0.0 if outside.
        
        Args:
            target: Target listing dict with price_value_vnd
            candidate: Candidate listing dict with price_value_vnd
            tolerance_pct: Tolerance percentage (0.2 = ±20%), default 0.2
        
        Returns:
            float: Score from 0.0 to 1.0
        """
        target_price = self._safe_float(target.get("price_value_vnd"))
        candidate_price = self._safe_float(candidate.get("price_value_vnd"))
        
        if target_price is None or candidate_price is None or target_price <= 0:
            return 0.0
        
        lower_bound = target_price * (1 - tolerance_pct)
        upper_bound = target_price * (1 + tolerance_pct)
        
        if lower_bound <= candidate_price <= upper_bound:
            return 1.0
        return 0.0

    def _price_direction_score(
        self,
        target: Dict[str, Any],
        candidate: Dict[str, Any],
        price_direction: str | None,
    ) -> float:
        direction = str(price_direction or "").strip().lower()
        if direction not in {"cheaper", "expensive"}:
            return 0.0

        target_price = self._safe_float(target.get("price_value_vnd"))
        candidate_price = self._safe_float(candidate.get("price_value_vnd"))
        if target_price is None or candidate_price is None or target_price <= 0 or candidate_price <= 0:
            return 0.0

        if direction == "cheaper" and candidate_price >= target_price:
            return 0.0
        if direction == "expensive" and candidate_price <= target_price:
            return 0.0

        ratio = candidate_price / target_price if target_price else 0.0
        if ratio <= 0:
            return 0.0
        return max(0.0, 1.0 - abs(1.0 - ratio))

    def _location_similarity_score(self, target: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        keys = ["city", "district", "ward", "street", "project"]
        compared = 0
        matched = 0
        for key in keys:
            target_value = str(target.get(key) or "").strip().lower()
            candidate_value = str(candidate.get(key) or "").strip().lower()
            if not target_value or not candidate_value:
                continue
            compared += 1
            if target_value == candidate_value or target_value in candidate_value or candidate_value in target_value:
                matched += 1
        if compared == 0:
            return 0.0
        return float(matched / compared)


# Alias for backward compatibility
RetrievalService = ListingHybridSearch
