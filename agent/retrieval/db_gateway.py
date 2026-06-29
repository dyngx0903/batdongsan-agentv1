from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence
from urllib import request

import psycopg

from agent.common import ensure_api_key_from_config, get_logger, load_config
from agent.retrieval.get_embedding import EMBED_TASK_TYPE_QUERY, get_embedding


class RetrievalDbGateway:
    SOURCE_CANDIDATES: Sequence[str] = ("agent_listing_search_v1", "listings")
    TABLE_COLUMNS: Sequence[str] = (
        "source",
        "listing_id",
        "title",
        "url",
        "transaction_type",
        "property_type",
        "project",
        "city",
        "district",
        "ward",
        "price_text",
        "area_text",
        "price_value_vnd",
        "area_m2",
        "bedrooms",
        "bathrooms",
        "floors",
        "frontage_width_m",
        "road_access_width_m",
        "legal_status",
        "direction",
        "search_document",
    )

    def __init__(self, config_path: str | None = None) -> None:
        self.logger = get_logger("retrieval_db_gateway")
        self.config_path = config_path or str(Path(__file__).resolve().parents[2] / "CONFIG" / "global.yaml")
        ensure_api_key_from_config(self.config_path)
        self.embedding_model_name, self.embedding_task_type, self.embedding_output_dimensionality = (
            self._load_embedding_settings(self.config_path)
        )
        self.db_config = self._load_db_config(self.config_path)
        self.embedding_endpoint = (
            os.getenv("BDS_EMBEDDING_ENDPOINT")
            or os.getenv("BDS_EMBED_ENDPOINT")
            or self._load_embedding_endpoint(self.config_path)
        )
        self.embedding_timeout_sec = 5.0

    def _load_embedding_settings(self, config_path: str) -> tuple[str, str, int | None]:
        model_name = "gemini-embedding-001"
        task_type = EMBED_TASK_TYPE_QUERY
        output_dimensionality: int | None = None

        try:
            cfg = load_config(config_path)
        except Exception:
            return model_name, task_type, output_dimensionality

        if not isinstance(cfg, dict):
            return model_name, task_type, output_dimensionality

        embedding_cfg = cfg.get("EMBEDDING", {}) if isinstance(cfg.get("EMBEDDING"), dict) else {}

        model_name = str(embedding_cfg.get("model") or model_name).strip() or model_name
        task_type = str(embedding_cfg.get("task_type") or task_type).strip() or task_type

        raw_dim = embedding_cfg.get("output_dimensionality")
        if raw_dim is not None:
            try:
                parsed_dim = int(raw_dim)
                if parsed_dim > 0:
                    output_dimensionality = parsed_dim
            except (TypeError, ValueError):
                output_dimensionality = None

        return model_name, task_type, output_dimensionality

    def _load_db_config(self, config_path: str) -> Dict[str, Any]:
        db: Dict[str, Any] = {}
        try:
            cfg = load_config(config_path)
            db = cfg.get("db", {}) if isinstance(cfg, dict) else {}
        except FileNotFoundError:
            self.logger.warning("db_config_file_missing path=%s trying_env_fallback=true", config_path)

        env_db = {
            "host": os.getenv("BDS_DB_HOST") or os.getenv("PGHOST"),
            "port": os.getenv("BDS_DB_PORT") or os.getenv("PGPORT"),
            "user": os.getenv("BDS_DB_USER") or os.getenv("PGUSER"),
            "password": os.getenv("BDS_DB_PASSWORD") or os.getenv("PGPASSWORD"),
            "dbname": os.getenv("BDS_DB_NAME") or os.getenv("PGDATABASE"),
        }
        for key, value in env_db.items():
            if value and key not in db:
                db[key] = value

        required = ["host", "port", "user", "password", "dbname"]
        missing = [k for k in required if k not in db]
        if missing:
            raise ValueError(
                "Missing DB config keys. Provide CONFIG/global.yaml db section "
                f"or env vars (BDS_DB_* / PG*). Missing: {missing}"
            )
        return db

    def _load_embedding_endpoint(self, config_path: str) -> str | None:
        try:
            cfg = load_config(config_path)
        except Exception:
            return None
        if not isinstance(cfg, dict):
            return None
        retrieval_cfg = cfg.get("retrieval", {}) if isinstance(cfg.get("retrieval"), dict) else {}
        endpoint = retrieval_cfg.get("embedding_endpoint")
        if endpoint:
            return str(endpoint)
        return None

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(
            host=self.db_config["host"],
            port=self.db_config["port"],
            user=self.db_config["user"],
            password=self.db_config["password"],
            dbname=self.db_config["dbname"],
            options="-c client_encoding=UTF8",
        )

    @staticmethod
    def to_vector_literal(embedding: List[float] | None) -> str | None:
        if not embedding:
            return None
        try:
            return "[" + ",".join(str(float(x)) for x in embedding) + "]"
        except (TypeError, ValueError):
            return None

    def get_query_embedding(self, query: str) -> List[float] | None:
        sdk_embedding = get_embedding(
            text=query,
            task_type=self.embedding_task_type,
            model_name=self.embedding_model_name,
            output_dimensionality=self.embedding_output_dimensionality,
        )
        if sdk_embedding:
            return sdk_embedding

        if not self.embedding_endpoint:
            return None

        payload = json.dumps({"query": query}).encode("utf-8")
        req = request.Request(
            self.embedding_endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.embedding_timeout_sec) as resp:
                body = resp.read().decode("utf-8")
        except Exception as exc:
            raise RuntimeError(f"embedding_request_failed: {exc}") from exc

        try:
            parsed = json.loads(body)
            vector = parsed.get("embedding")
        except Exception as exc:
            raise RuntimeError(f"embedding_response_invalid_json: {exc}") from exc

        if not isinstance(vector, list) or not vector:
            raise RuntimeError("embedding_response_missing_vector")

        try:
            return [float(x) for x in vector]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"embedding_vector_invalid: {exc}") from exc

    def source_candidates(self) -> List[str]:
        available: List[str] = []
        with self._connect() as conn:
            with conn.cursor() as cur:
                for table_name in self.SOURCE_CANDIDATES:
                    try:
                        cur.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
                        available.append(table_name)
                    except psycopg.errors.UndefinedTable:
                        conn.rollback()
                        self.logger.info("db_contract_table_missing source=%s", table_name)
        return available

    def fetch_listing(self, source: str, listing_id: str) -> Dict[str, Any] | None:
        table_candidates = self.source_candidates()
        if not table_candidates:
            return None

        columns = ",\n                ".join(self.TABLE_COLUMNS)
        for table_name in table_candidates:
            sql = f"""
                SELECT
                    {columns}
                FROM {table_name}
                WHERE source = %s AND listing_id::text = %s
                LIMIT 1
            """
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, [source, str(listing_id)])
                    row = cur.fetchone()
                    if not row:
                        continue
                    cols = [d[0] for d in cur.description] if cur.description else []
                    record = dict(zip(cols, row))
                    record["_contract_source"] = table_name
                    return record
        return None

    def fetch_lexical_candidates(
        self,
        table_name: str,
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
                FROM {table_name}
                WHERE {where_sql}
                  AND to_tsvector('simple', COALESCE(search_document, '')) @@ {tsquery_fn}('simple', %s)
                ORDER BY lexical_score DESC, listing_id DESC
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
                    table_name=table_name,
                    where_sql=where_sql,
                    params=params,
                    prefilter_limit=lexical_candidates_limit,
                )

    def fetch_semantic_candidates(
        self,
        table_name: str,
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
            FROM {table_name}
            WHERE {where_sql}
              AND search_document_embedding IS NOT NULL
            ORDER BY search_document_embedding <=> %s::vector ASC
            LIMIT %s
        """

        rows: List[Dict[str, Any]] = []
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
        return rows

    def fetch_fallback_candidates(
        self,
        table_name: str,
        where_sql: str,
        params: List[Any],
        prefilter_limit: int,
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
                0.0 AS lexical_score
            FROM {table_name}
            WHERE {where_sql}
            ORDER BY listing_id DESC
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
