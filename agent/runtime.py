from __future__ import annotations

from pathlib import Path
import json
import os
import re
import time
import unicodedata
import uuid
from typing import Any, Dict

try:
    from google import genai
except Exception:
    genai = None

from agent.common import (
    canonicalize_llm_slots,
    ensure_api_key_from_config,
    get_logger,
    load_config,
    parse_user_query,
)
from agent.prompt_templates import (
    INPUT_UNDERSTANDING_PROMPT_VERSION,
    build_input_understanding_prompt,
    build_response_composer_prompt,
)
from agent.comparison_formatter import format_comparison
from agent.response_formatter import RuntimeResponseFormatter
from agent.session_manager import SessionManager
from agent.task_handler import TaskHandler
from agent.task_schema import parse_llm_query_understanding, parse_task_input
from agent.tools import (
    run_analytics_listings,
    run_compare_listings,
    run_explain_listing,
    run_respond_to_user,
    run_search_listings,
    run_similar_listings,
    run_suggest_area,
)


class AgentRuntime:
    """Main runtime entrypoint for batdongsan-agent."""

    def __init__(self, config_path: str | None = None) -> None:
        self.logger = get_logger("agent_runtime")
        self.config_path = config_path or str(Path(__file__).resolve().parents[1] / "CONFIG" / "global.yaml")
        ensure_api_key_from_config(self.config_path)
        self.session_manager = SessionManager()
        self.task_handler = TaskHandler()
        self.formatter = RuntimeResponseFormatter()
        self.runtime_flags = self._load_runtime_flags(config_path=self.config_path)

    @staticmethod
    def _normalize_text(value: str) -> str:
        text = str(value or "").strip().lower()
        text = text.replace("đ", "d")
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _extract_listing_refs_from_query(cls, query: str) -> Dict[str, Any]:
        text = str(query or "")
        refs = re.findall(r"\b([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)\b", text)
        if not refs:
            return {}

        out: Dict[str, Any] = {"listing_ref": refs[0]}
        if len(refs) >= 2:
            out["listing_ref_a"] = refs[0]
            out["listing_ref_b"] = refs[1]
        return out

    @staticmethod
    def _is_valid_listing_ref(value: Any) -> bool:
        ref = str(value or "").strip()
        return bool(re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", ref))

    @classmethod
    def _sanitize_listing_ref_metadata(cls, metadata: Dict[str, Any]) -> None:
        for key in ("listing_ref", "listing_ref_a", "listing_ref_b"):
            value = metadata.get(key)
            if not cls._is_valid_listing_ref(value):
                metadata.pop(key, None)

        listing_refs = metadata.get("listing_refs")
        if isinstance(listing_refs, list):
            normalized_refs: list[str] = []
            for item in listing_refs:
                if cls._is_valid_listing_ref(item):
                    one = str(item).strip()
                    if one not in normalized_refs:
                        normalized_refs.append(one)
            if normalized_refs:
                metadata["listing_refs"] = normalized_refs
            else:
                metadata.pop("listing_refs", None)

    @classmethod
    def _query_mentions_current_listing(cls, query: str) -> bool:
        text = cls._normalize_text(query)
        if not text:
            return False
        mention_tokens = [
            "can nay",
            "listing nay",
            "bat dong san nay",
            "tin nay",
            "can ho nay",
            "nha nay",
            "can do",
            "listing do",
            "can tren",
            "listing tren",
        ]
        return any(token in text for token in mention_tokens)

    @classmethod
    def _query_mentions_two_recent_listings(cls, query: str) -> bool:
        text = cls._normalize_text(query)
        if not text:
            return False
        pair_tokens = [
            "2 can do",
            "2 can nay",
            "2 can tren",
            "hai can do",
            "hai can nay",
            "hai can tren",
        ]
        return any(token in text for token in pair_tokens)

    def _collect_recent_session_listing_refs(self, session_id: str, limit: int = 2) -> list[str]:
        state = self.session_manager.get_or_create(session_id)
        candidate_keys = [
            "recent_listing_refs",
            "last_turn_listing_refs",
            "last_browse_listing_refs",
        ]
        refs: list[str] = []
        for key in candidate_keys:
            values = state.metadata.get(key)
            if not isinstance(values, list):
                continue
            for raw in values:
                ref = str(raw or "").strip()
                if not self._is_valid_listing_ref(ref):
                    continue
                if ref in refs:
                    continue
                refs.append(ref)
                if len(refs) >= max(1, int(limit)):
                    return refs
        return refs

    def _apply_followup_listing_refs(self, *, session_id: str, query: str, metadata: Dict[str, Any]) -> None:
        if self._query_mentions_two_recent_listings(query):
            refs = self._collect_recent_session_listing_refs(session_id=session_id, limit=2)
            if len(refs) >= 2:
                metadata.setdefault("listing_ref_a", refs[0])
                metadata.setdefault("listing_ref_b", refs[1])
                metadata.setdefault("listing_refs", refs[:2])

        has_single_ref = self._is_valid_listing_ref(metadata.get("listing_ref"))
        if not has_single_ref and self._query_mentions_current_listing(query):
            refs = self._collect_recent_session_listing_refs(session_id=session_id, limit=1)
            if refs:
                metadata["listing_ref"] = refs[0]

    @classmethod
    def _extract_listing_indices_from_query(cls, query: str) -> list[int]:
        text = cls._normalize_text(query)
        if not text:
            return []

        # Only treat numeric references as listing-order picks when the query
        # mentions result items to avoid colliding with bedrooms/price numbers.
        anchor_tokens = [
            "listing",
            "can",
            "tin",
            "lua chon",
            "option",
            "ket qua",
            "result",
            "danh sach",
        ]
        if not any(token in text for token in anchor_tokens):
            return []

        numbers: list[int] = []
        patterns = [
            r"\b(?:listing|can|tin|lua chon|option|ket qua|result)\s*(?:so|#)?\s*(\d{1,2})\b",
            r"\b(?:so|#)\s*(\d{1,2})\b",
            r"\b(?:thu|thứ)\s*(\d{1,2})\b",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, text):
                try:
                    value = int(match)
                except ValueError:
                    continue
                if value <= 0:
                    continue
                if value not in numbers:
                    numbers.append(value)
        return numbers[:2]

    @classmethod
    def _is_compare_intent_query(cls, query: str) -> bool:
        text = cls._normalize_text(query)
        if not text:
            return False
        return any(token in text for token in ["so sanh", "compare", " vs ", "vs ", " vs"])

    @classmethod
    def _should_autofill_listing_ref(cls, query: str) -> bool:
        text = cls._normalize_text(query)
        if not text:
            return False
        if cls._query_mentions_current_listing(query):
            return True
        return any(token in text for token in ["tuong tu", "similar", "giong"])

    @staticmethod
    def _extract_listing_ref_from_item(item: Any) -> str | None:
        if not isinstance(item, dict):
            return None

        for key in ("listing_ref", "ref", "winner_ref"):
            ref = str(item.get(key) or "").strip()
            if ref and "/" in ref:
                return ref

        source = str(item.get("source") or "").strip()
        listing_id = str(item.get("listing_id") or "").strip()
        if source and listing_id:
            return f"{source}/{listing_id}"
        return None

    @classmethod
    def _collect_listing_refs_from_tool_result(cls, tool_result: Dict[str, Any]) -> list[str]:
        refs: list[str] = []

        def _push(ref_value: Any) -> None:
            ref = str(ref_value or "").strip()
            if not ref or "/" not in ref:
                return
            if ref not in refs:
                refs.append(ref)

        for key in ("listing_ref", "winner_ref", "listing_ref_a", "listing_ref_b"):
            _push(tool_result.get(key))

        listing = tool_result.get("listing")
        ref_from_listing = cls._extract_listing_ref_from_item(listing)
        _push(ref_from_listing)

        items = tool_result.get("items")
        if isinstance(items, list):
            for item in items:
                _push(cls._extract_listing_ref_from_item(item))

        return refs

    def _remember_listing_context(self, session_id: str, tool_result: Dict[str, Any]) -> None:
        refs = self._collect_listing_refs_from_tool_result(tool_result)
        tool_name = str(tool_result.get("tool") or "").strip().lower()

        metadata_patch: Dict[str, Any] = {}

        if tool_name == "suggest_area":
            area_rows = tool_result.get("area_recommendations") or []
            suggested_areas: list[str] = []
            suggested_area_payload: list[Dict[str, Any]] = []
            if isinstance(area_rows, list):
                for item in area_rows:
                    if not isinstance(item, dict):
                        continue
                    area_name = str(item.get("area") or item.get("district") or "").strip()
                    if not area_name:
                        continue
                    if area_name not in suggested_areas:
                        suggested_areas.append(area_name)
                    suggested_area_payload.append(
                        {
                            "area": area_name,
                            "score": item.get("score"),
                            "reason": item.get("reason"),
                            "estimated_price": item.get("estimated_price"),
                            "inventory_level": item.get("inventory_level"),
                        }
                    )
            if suggested_areas:
                metadata_patch["last_suggested_area_districts"] = suggested_areas[:8]
                metadata_patch["last_suggested_area_rankings"] = suggested_area_payload[:8]
                metadata_patch["last_suggested_area_tool"] = "suggest_area"

        if not refs and not metadata_patch:
            return

        ordered_turn_refs: list[str] = []
        items = tool_result.get("items")
        if isinstance(items, list):
            for item in items:
                ref = self._extract_listing_ref_from_item(item)
                if ref and ref not in ordered_turn_refs:
                    ordered_turn_refs.append(ref)

        if not ordered_turn_refs:
            ordered_turn_refs = list(refs)

        state = self.session_manager.get_or_create(session_id)
        existing_refs = state.metadata.get("recent_listing_refs")
        if not isinstance(existing_refs, list):
            existing_refs = []

        merged_refs: list[str] = []
        for ref in refs + [str(item).strip() for item in existing_refs]:
            if ref and ref not in merged_refs:
                merged_refs.append(ref)

        if refs:
            metadata_patch.update(
                {
                    "last_listing_ref": refs[0],
                    "recent_listing_refs": merged_refs[:8],
                    "last_turn_listing_refs": ordered_turn_refs[:30],
                }
            )
        if tool_name in {"search_listings", "similar_listings"} and ordered_turn_refs:
            metadata_patch["last_browse_listing_refs"] = ordered_turn_refs[:30]

        self.session_manager.merge_metadata(session_id, metadata_patch)

    @classmethod
    def _route_tool(cls, query: str, metadata: Dict[str, Any]) -> str:
        forced = str(metadata.get("selected_tool") or metadata.get("force_tool") or "").strip().lower()
        allowed = {
            "search_listings",
            "explain_listing",
            "similar_listings",
            "compare_listings",
            "suggest_area",
            "analytics_listings",
            "respond_to_user",
        }
        if forced in allowed:
            return forced

        text = cls._normalize_text(query)
        has_ref = bool(re.search(r"\b[a-z0-9_-]+/[a-z0-9_-]+\b", text))
        has_context_ref = any(bool(str(metadata.get(key) or "").strip()) for key in ("listing_ref", "listing_ref_a", "listing_ref_b"))
        parsed = parse_user_query(query)

        # Explicit area-level average-price comparison should stay in analytics.
        compare_avg_area_query = (
            any(token in text for token in ["so sanh", "vs", "giua"])
            and any(token in text for token in ["gia trung binh", "trung binh", "average"])
            and any(token in text for token in ["quan", "huyen", "phuong", "xa", "district", "ward"])
        )
        if compare_avg_area_query:
            return "analytics_listings"

        family_ranking_query = (
            any(token in text for token in ["gia dinh", "family"])
            and any(token in text for token in ["nhieu listing", "nhieu nhat", "top", "cao nhat", "thap nhat"])
            and any(token in text for token in ["khu", "khu vuc", "quan", "huyen", "phuong", "xa"])
        )
        if family_ranking_query:
            return "analytics_listings"

        market_overview_tokens = [
            "gia thi truong",
            "mat bang gia",
            "gia khu vuc",
            "khong biet gia",
            "chua biet gia",
            "gia nhu nao",
            "gia sao",
            "xu huong gia",
            "tiem nang tang gia",
        ]
        if any(token in text for token in market_overview_tokens):
            return "analytics_listings"

        listing_price_ranking_query = (
            any(token in text for token in ["re nhat", "thap nhat", "cao nhat", "dat nhat"])
            and any(
                bool(getattr(parsed.hard_filters, field, None))
                for field in ("property_type", "transaction_type", "project", "street")
            )
        )
        if listing_price_ranking_query:
            return "search_listings"

        # Check for analytics queries first (count, average, statistics)
        analytics_tokens = [
            "bao nhieu",
            "so luong",
            "count",
            "trung binh",
            "average",
            "thap nhat",
            "cao nhat",
            "min",
            "max",
            "thong ke",
            "ti le",
            "ty le",
            "phan tram",
            "ratio",
        ]
        if any(token in text for token in analytics_tokens):
            return "analytics_listings"

        if any(token in text for token in ["so sanh", "compare", "vs"]):
            return "compare_listings"
        if any(token in text for token in ["tuong tu", "giong", "similar"]):
            return "similar_listings"

        suggested_areas = metadata.get("last_suggested_area_districts")
        if isinstance(suggested_areas, list) and suggested_areas:
            browse_tokens = ["xem", "listing", "can", "căn", "loc", "lọc", "chi tiet", "chi tiết", "tim", "tìm"]
            if any(token in text for token in browse_tokens):
                normalized_suggested_areas = [str(item or "").strip().lower() for item in suggested_areas if str(item or "").strip()]
                if any(area and area in text for area in normalized_suggested_areas):
                    return "search_listings"

        explain_tokens = [
            "giai thich",
            "phan tich can",
            "explain",
            "noi bat",
            "hop ly",
            "phu hop",
            "location",
            "vi tri",
            "the nao",
            "tai sao",
            "chi tiet",
            "thong tin chi tiet",
            "thong tin listing",
        ]
        mention_current_listing = any(token in text for token in ["can nay", "listing nay", "can ho nay", "listing do", "can do"])
        if any(token in text for token in explain_tokens) and (has_ref or has_context_ref or mention_current_listing):
            return "explain_listing"

        area_exploration_tokens = [
            "khu vuc nao",
            "nhung khu vuc nao",
            "o khu vuc nao",
            "khu nao phu hop",
            "khu vuc phu hop",
        ]
        has_area_exploration_phrase = any(token in text for token in area_exploration_tokens)
        has_area_context = bool(parsed.hard_filters.max_price_vnd is not None or parsed.hard_filters.min_price_vnd is not None or parsed.hard_filters.property_type)
        if has_area_exploration_phrase and has_area_context:
            return "suggest_area"

        if str(getattr(parsed, "use_case", "") or "").strip().lower() == "suggest_area":
            return "suggest_area"

        has_structured_listing_signals = any(
            [
                bool(parsed.hard_filters.property_type),
                bool(parsed.hard_filters.district),
                bool(parsed.hard_filters.city),
                bool(getattr(parsed.hard_filters, "street", None)),
                bool(parsed.hard_filters.project),
                parsed.hard_filters.max_price_vnd is not None,
                parsed.hard_filters.min_area_m2 is not None,
                parsed.hard_filters.min_bedrooms is not None,
            ]
        )

        search_intent_tokens = [
            "tim",
            "mua",
            "ban",
            "thue",
            "can ho",
            "nha",
            "chung cu",
            "bat dong san",
        ]
        if has_structured_listing_signals and any(token in text for token in search_intent_tokens):
            return "search_listings"

        suggest_tokens = [
            "khu nao",
            "khu vuc",
            "nen o dau",
            "nen o",
            "suggest area",
            "gia dinh",
            "lifestyle",
            "yen tinh",
            "nen chon",
            "tot hon",
            "so voi",
            "uu tien gi",
            "nguoi lon tuoi",
            "tu van",
            "chua biet",
            "hay can ho",
        ]
        if any(token in text for token in suggest_tokens):
            return "suggest_area"
        return "search_listings"

    @staticmethod
    def _allowed_tools() -> set[str]:
        return {
            "search_listings",
            "explain_listing",
            "similar_listings",
            "compare_listings",
            "suggest_area",
            "analytics_listings",
            "respond_to_user",
        }

    @classmethod
    def _forced_tool_from_metadata(cls, metadata: Dict[str, Any]) -> str | None:
        forced = str(metadata.get("selected_tool") or metadata.get("force_tool") or "").strip().lower()
        return forced if forced in cls._allowed_tools() else None

    @classmethod
    def _detect_out_of_scope_or_adversarial(cls, query: str) -> bool:
        text = cls._normalize_text(query)
        if not text:
            return False

        explicit_patterns = [
            "viet sql",
            "write sql",
            "database co bao nhieu dong",
            "cho toi tat ca du lieu",
            "in toan bo bang",
            "bang listings",
            "bypass filter",
            "tra toan bo data",
            "cho toi api key",
            "raw json",
            "embedding vector",
            "toan bo embedding",
            "dump db",
            "xuat toan bo du lieu",
        ]
        if any(token in text for token in explicit_patterns):
            return True

        # Broad safety net for internal system extraction requests.
        risky_tokens = ["sql", "database", "api key", "embedding", "internal", "db"]
        exfiltration_tokens = ["toan bo", "tat ca", "in", "xuat", "dump", "bypass"]
        if any(t in text for t in risky_tokens) and any(t in text for t in exfiltration_tokens):
            return True

        return False

    @classmethod
    def _detect_ambiguous_clarification_prompt(cls, query: str) -> str | None:
        text = cls._normalize_text(query)
        if not text:
            return None

        has_ref = bool(re.search(r"\b[a-z0-9_-]+/[a-z0-9_-]+\b", text))
        if has_ref:
            return None

        parsed = parse_user_query(query)
        if str(getattr(parsed, "use_case", "") or "").strip().lower() == "market_overview":
            return None

        # Property type alone (e.g. "nha", "can ho") is still too vague for direct search.
        has_structured_listing_signals = any(
            [
                bool(parsed.hard_filters.district),
                bool(parsed.hard_filters.city),
                bool(getattr(parsed.hard_filters, "street", None)),
                bool(parsed.hard_filters.project),
                parsed.hard_filters.max_price_vnd is not None,
                parsed.hard_filters.min_area_m2 is not None,
                parsed.hard_filters.min_bedrooms is not None,
            ]
        )
        if has_structured_listing_signals:
            return None

        ambiguous_patterns = [
            "gia re",
            "can nao on",
            "goi y giup toi vai can",
            "toi muon mua nha",
            "can nao dep",
            "gan trung tam",
            "vai lua chon tot",
            "dang mua",
            "ok nhat",
            "co gi moi",
        ]
        if not any(token in text for token in ambiguous_patterns):
            return None

        return (
            "De toi tu van dung nhu cau, ban vui long cho them 4 thong tin: "
            "(1) khu vuc quan/thanh pho nao? "
            "(2) ngan sach toi da bao nhieu? "
            "(3) loai hinh can ho hay nha pho? "
            "(4) can may phong ngu?"
        )

    @staticmethod
    def _extract_first_json_object(text: str) -> Dict[str, Any] | None:
        raw = str(text or "").strip()
        if not raw:
            return None

        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, flags=re.IGNORECASE)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                pass

        generic = re.search(r"\{[\s\S]*\}", raw)
        if generic:
            try:
                parsed = json.loads(generic.group(0))
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None

        return None

    def _route_tool_with_gemini(self, *, query: str, metadata: Dict[str, Any]) -> str | None:
        if genai is None:
            return None

        api_key = str((os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "")).strip()
        if not api_key:
            return None

        model_name = str(self.runtime_flags.get("llm_model") or "gemini-2.5-flash-lite").strip()
        allowed_tools = sorted(self._allowed_tools())
        prompt = (
            "Chon dung 1 tool phu hop nhat cho truy van bat dong san. "
            "Chi tra ve JSON dung format: {\"tool\":\"<tool_name>\"}. "
            "Khong tra ve van ban khac.\n\n"
            f"Allowed tools: {', '.join(allowed_tools)}\n"
            f"Query: {query}\n"
            "Metadata: "
            f"{json.dumps(metadata or {}, ensure_ascii=True)}"
        )

        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            text = str(getattr(response, "text", "") or "").strip()
            if not text:
                return None

            parsed = self._extract_first_json_object(text)
            tool = str((parsed or {}).get("tool") or (parsed or {}).get("selected_tool") or "").strip().lower()
            if not tool:
                token_match = re.search(r"\b(search_listings|explain_listing|similar_listings|compare_listings|suggest_area|analytics_listings|respond_to_user)\b", text.lower())
                if token_match:
                    tool = token_match.group(1)

            if tool in self._allowed_tools():
                return tool
            return None
        except Exception as exc:
            self.logger.warning("llm_router_fallback_failed model=%s error=%s", model_name, exc)
            return None

    def _parse_user_input_with_gemini(self, *, query: str, metadata: Dict[str, Any]) -> Any:
        if genai is None:
            return None

        api_key = str((os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "")).strip()
        if not api_key:
            return None

        model_name = str(self.runtime_flags.get("llm_model") or "gemini-2.5-flash-lite").strip()
        timeout_ms = self._coerce_int(self.runtime_flags.get("llm_input_timeout_ms"), default=1500, minimum=200, maximum=10_000)
        allowed_intents = sorted(self._allowed_tools())
        conversation_context = str(metadata.get("conversation_context") or "")
        prompt = build_input_understanding_prompt(
            query=query,
            metadata=metadata,
            conversation_context=conversation_context,
            allowed_intents=allowed_intents,
            timeout_ms=timeout_ms,
        )

        started = time.perf_counter()
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            if elapsed_ms > timeout_ms:
                self.logger.info("llm_input_parser_timeout model=%s elapsed_ms=%s budget_ms=%s", model_name, elapsed_ms, timeout_ms)
                return None

            text = str(getattr(response, "text", "") or "").strip()
            if not text:
                return None

            parsed_obj = self._extract_first_json_object(text)
            if not parsed_obj:
                return None
            return parse_llm_query_understanding(parsed_obj)
        except Exception as exc:
            self.logger.warning("llm_input_parser_failed model=%s error=%s", model_name, exc)
            return None

    def _build_conversation_context(self, session_id: str, *, max_messages: int = 10) -> str:
        state = self.session_manager.get_or_create(session_id)
        if not state.messages:
            return ""

        recent = state.messages[-max_messages:]
        lines: list[str] = []
        for msg in recent:
            role = str(msg.get("role") or "").strip().lower()
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            if role not in {"user", "assistant", "system"}:
                role = "assistant"
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _has_unknown_urls(text: str, allowed_urls: set[str]) -> bool:
        found = set(re.findall(r"https?://[^\s)\]\}]+", str(text or "")))
        if not found:
            return False
        normalized_allowed = {str(url or "").strip() for url in allowed_urls if str(url or "").strip()}
        return any(url not in normalized_allowed for url in found)

    @staticmethod
    def _strip_urls_and_markdown_links(text: str) -> str:
        cleaned = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", str(text or ""))
        cleaned = re.sub(r"https?://[^\s)\]}]+", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        return cleaned.strip()

    @staticmethod
    def _format_listing_ref(tool_result: Dict[str, Any], listing: Dict[str, Any]) -> str:
        source = str(listing.get("source") or tool_result.get("source") or "").strip()
        listing_id = str(listing.get("listing_id") or tool_result.get("listing_id") or "").strip()
        if source and listing_id:
            return f"{source}/{listing_id}"
        return str(tool_result.get("listing_ref") or "").strip()

    @staticmethod
    def _extract_search_document_highlights(search_document: str, *, max_items: int = 2) -> list[str]:
        text = re.sub(r"\s+", " ", str(search_document or "")).strip()
        if not text:
            return []

        chunks = [
            item.strip(" ,.;:-")
            for item in re.split(r"[\n\r\.\!\?;]+", text)
            if str(item or "").strip()
        ]
        if not chunks:
            return []

        keywords = [
            "hem",
            "ngo",
            "mat tien",
            "so hong",
            "phap ly",
            "vay ngan hang",
            "ho tro vay",
            "gan metro",
            "gan truong",
            "gan cho",
            "benh vien",
            "noi that",
            "view",
            "san thuong",
            "moi",
        ]

        highlights: list[str] = []
        noisy_meta_tokens = [
            "loai giao dich",
            "loai bat dong san",
            "thanh pho",
            "quan huyen",
            "phuong xa",
            "duong:",
        ]

        def _looks_noisy_meta(chunk: str) -> bool:
            lowered = AgentRuntime._normalize_text(chunk)
            hit = sum(1 for token in noisy_meta_tokens if token in lowered)
            return hit >= 3

        for chunk in chunks:
            lowered = AgentRuntime._normalize_text(chunk)
            if not lowered:
                continue
            if _looks_noisy_meta(chunk):
                continue
            if any(token in lowered for token in keywords):
                cleaned = AgentRuntime._strip_urls_and_markdown_links(chunk)
                if cleaned and cleaned not in highlights:
                    highlights.append(cleaned[:180])
            if len(highlights) >= max_items:
                break

        if not highlights:
            for chunk in chunks[:max_items]:
                if _looks_noisy_meta(chunk):
                    continue
                cleaned = AgentRuntime._strip_urls_and_markdown_links(chunk)
                if cleaned:
                    highlights.append(cleaned[:180])

        return highlights[:max_items]

    @staticmethod
    def _build_explain_summary(tool_result: Dict[str, Any], listing: Dict[str, Any]) -> str:
        listing_ref = AgentRuntime._format_listing_ref(tool_result, listing)
        title = str(listing.get("title") or tool_result.get("title") or "").strip()
        search_document = str(tool_result.get("search_document") or listing.get("search_document") or "")
        search_doc_highlights = AgentRuntime._extract_search_document_highlights(search_document)

        detail_parts: list[str] = []
        for key in ("price_text", "area_text"):
            value = str(tool_result.get(key) or listing.get(key) or "").strip()
            if value:
                detail_parts.append(value)

        bedrooms = tool_result.get("bedrooms", listing.get("bedrooms"))
        bathrooms = tool_result.get("bathrooms", listing.get("bathrooms"))
        floors = tool_result.get("floors", listing.get("floors"))

        if bedrooms is not None:
            detail_parts.append(f"{bedrooms} PN")
        if bathrooms is not None:
            detail_parts.append(f"{bathrooms} WC")
        if floors is not None:
            detail_parts.append(f"{floors} tang")

        road_width = tool_result.get("road_access_width_m", listing.get("road_access_width_m"))
        if road_width is not None:
            detail_parts.append(f"hem {road_width}m")

        district = str(tool_result.get("district") or listing.get("district") or "").strip()
        if district:
            detail_parts.append(district)
        if search_doc_highlights:
            detail_parts.extend(search_doc_highlights)

        base = f"Thong tin chi tiet cho {listing_ref}:" if listing_ref else "Thong tin chi tiet listing:"
        if title:
            base = f"{base} {title}."
        if detail_parts:
            return f"{base} " + "; ".join(detail_parts[:6]) + "."
        return base

    @staticmethod
    def _format_budget_vnd_compact(amount_vnd: int | float | None) -> str:
        try:
            value = float(amount_vnd or 0)
        except (TypeError, ValueError):
            return ""
        if value <= 0:
            return ""
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.1f} tỷ"
        return f"{value / 1_000_000:.0f} triệu"

    @classmethod
    def _extract_investment_goal(cls, query: str) -> str:
        text = cls._normalize_text(query)
        appreciation = any(token in text for token in ["tang gia", "tang truong", "appreciation", "lai von"])
        if appreciation:
            return "uu tien tang gia trung han va dai han"
        return "toi uu thanh khoan mua ban va bien an toan"

    @classmethod
    def _query_mentions_rental_intent(cls, query: str) -> bool:
        text = cls._normalize_text(query)
        return any(token in text for token in ["cho thue", "thue", "dong tien", "rent", "yield"])

    @staticmethod
    def _suggest_area_followup_questions(missing_fields: list[str]) -> list[str]:
        prompts = {
            "budget": "Ngân sách dự kiến khoảng bao nhiêu?",
            "commuting_destination": "Anh/chị muốn ưu tiên khu nào hoặc đi làm ở đâu?",
            "family_profile": "Anh/chị mua để ở hay đầu tư, và gia đình có mấy người?",
            "property_type": "Anh/chị ưu tiên căn hộ, nhà phố hay đất?",
        }
        out: list[str] = []
        for field in missing_fields:
            label = prompts.get(str(field or "").strip())
            if label and label not in out:
                out.append(label)
        return out[:3]

    @classmethod
    def _build_suggest_area_investment_summary(cls, query: str, tool_result: Dict[str, Any]) -> str:
        parsed = parse_user_query(query or "")
        area_rows = [
            row
            for row in (tool_result.get("area_rankings") or tool_result.get("area_recommendations") or [])
            if isinstance(row, dict) and str(row.get("area") or row.get("district") or "").strip()
        ][:3]

        min_budget = cls._format_budget_vnd_compact(parsed.hard_filters.min_price_vnd)
        max_budget = cls._format_budget_vnd_compact(parsed.hard_filters.max_price_vnd)
        if min_budget and max_budget:
            budget_text = f"khoảng {min_budget} - {max_budget}"
        elif max_budget:
            budget_text = f"tối đa {max_budget}"
        elif min_budget:
            budget_text = f"từ {min_budget}"
        else:
            budget_text = "chưa chốt rõ"

        city_text = str(parsed.hard_filters.city or "").strip() or "khu vực mục tiêu"
        property_text = str(parsed.hard_filters.property_type or "").strip() or "bất động sản"

        intro = (
            f"Với ngân sách {budget_text}, mình đang xếp hạng các khu có nhiều {property_text.lower()} phù hợp nhất trong {city_text}. "
            "Tiêu chí chính là mức độ khớp ngân sách, độ phù hợp loại hình, nguồn cung thực tế và vị trí."
        )

        lines: list[str] = []
        lines.append(intro)
        lines.append("")
        lines.append("## Khu vực phù hợp")
        lines.append("")

        if not area_rows:
            lines.append("Hiện hệ thống chưa có đủ listing phù hợp để xếp hạng chắc tay. Nếu anh/chị nới khu vực, đổi loại hình hoặc mở rộng ngân sách, mình sẽ lọc lại ngay.")
        else:
            for area in area_rows:
                district = str(area.get("area") or area.get("district") or "").strip()
                listing_count = int(area.get("listing_count") or 0)
                listing_in_budget_count = int(area.get("listing_in_budget_count") or 0)
                budget_coverage = float(area.get("budget_coverage") or 0.0)
                budget_fit = float(area.get("budget_fit") or 0.0)
                property_type_fit = float(area.get("property_type_fit") or area.get("property_match") or 0.0)
                inventory_score = float(area.get("inventory_score") or area.get("inventory_fit") or 0.0)
                location_score = float(area.get("location_score") or 0.0)
                commute = area.get("commute_minutes")
                market_comment = str(area.get("market_comment") or "").strip()
                area_comment = str(area.get("area_comment") or "").strip()
                common_property = str(area.get("common_property") or "").strip()
                price_range_text = str(area.get("price_range_text") or "").strip()
                estimated_price = str(area.get("estimated_price") or "").strip()
                confidence = str(area.get("confidence") or "").strip()
                matching_reasons = [
                    str(item).strip()
                    for item in (area.get("matching_reasons") or [])
                    if str(item).strip()
                ]
                non_matching_reasons = [
                    str(item).strip()
                    for item in (area.get("non_matching_reasons") or [])
                    if str(item).strip()
                ]

                rank = int(area.get("rank") or 0)
                medals = ["🥇", "🥈", "🥉"]
                medal = medals[min(rank - 1, 2)] if rank > 0 else "●"
                if rank:
                    lines.append(f"### {medal} {rank}. {district}")
                else:
                    lines.append(f"### {medal} {district}")
                # CHANGED (2026-06-27): Removed duplicate "% listing nằm trong ngân sách" line - already in matching_reasons from DAL
                if estimated_price:
                    lines.append(f"- Giá tham chiếu: {estimated_price}.")
                if price_range_text:
                    lines.append(f"- Mặt bằng giá: {price_range_text}")
                if common_property:
                    lines.append(f"- Loại hình phổ biến: {common_property}")
                if area_comment:
                    lines.append(f"- Diện tích phổ biến: {area_comment}")

                for reason in matching_reasons[:4]:
                    if reason not in {market_comment, area_comment, common_property}:
                        lines.append(f"- {reason}")
                if non_matching_reasons:
                    lines.append("**Lưu ý:**")
                    for risk in non_matching_reasons[:2]:
                        risk_clean = str(risk).replace("⚠️", "").strip()
                        lines.append(f"- ⚠️ {risk_clean}")
                lines.append("")

        if area_rows:
            top_area = area_rows[0]
            top_district = str(top_area.get("district") or top_area.get("area") or "").strip()
            lines.append("---")
            lines.append("")
            lines.append("⭐ **Khuyến nghị**")
            lines.append("")
            lines.append(f"Nếu chỉ chọn một khu vực để xem listing trước, mình đề xuất **{top_district}** vì:")
            lines.append("")
            lines.append("• Vị trí và khả năng tìm được bất động sản phù hợp ngân sách cân bằng tốt")
            lines.append("• Tiềm năng tăng giá lâu dài")
            lines.append("")
            lines.append("Khi anh/chị chọn 1 khu trong danh sách, mình sẽ lọc listing phù hợp ngay.")
        
        lines.append("")
        lines.append("Nếu muốn, mình có thể lọc tiếp theo loại hình, diện tích hoặc mức giá.")

        return "\n".join(lines).strip()

    def _execute_tool(self, tool_name: str, query: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        args: Dict[str, Any] = {
            "query": query,
            "user_query": query,
            "context_query": query,
            "top_k": metadata.get("top_k", 10),
            "parsed_filters": metadata.get("parsed_filters"),
            "config_path": self.config_path,
            "source": metadata.get("source"),
            "listing_id": metadata.get("listing_id"),
            "listing_ref": metadata.get("listing_ref"),
            "source_a": metadata.get("source_a"),
            "listing_id_a": metadata.get("listing_id_a"),
            "source_b": metadata.get("source_b"),
            "listing_id_b": metadata.get("listing_id_b"),
            "listing_ref_a": metadata.get("listing_ref_a"),
            "listing_ref_b": metadata.get("listing_ref_b"),
            "listing_refs": metadata.get("listing_refs"),
            "user_context": metadata.get("user_context"),
            "user_profile": metadata.get("user_profile"),
            "final_response": metadata.get("final_response"),
            "message": metadata.get("message"),
        }

        if tool_name == "search_listings":
            return run_search_listings(None, args)
        if tool_name == "explain_listing":
            return run_explain_listing(None, args)
        if tool_name == "similar_listings":
            return run_similar_listings(None, args)
        if tool_name == "compare_listings":
            return run_compare_listings(None, args)
        if tool_name == "suggest_area":
            return run_suggest_area(None, args)
        if tool_name == "analytics_listings":
            return run_analytics_listings(None, args)
        if tool_name == "respond_to_user":
            return run_respond_to_user(None, args)
        return run_respond_to_user(None, {"message": "Không tìm thấy tool phù hợp"})

    @staticmethod
    def _derive_state_and_branch(tool_result: Dict[str, Any], selected_tool: str) -> tuple[str, str, str]:
        status = str(tool_result.get("status") or "").strip().lower()
        fallback_mode = str(tool_result.get("fallback_mode") or "none")

        if status in {"invalid_input", "internal_error", "tool_error"}:
            return "failed", "error", fallback_mode if fallback_mode != "none" else status
        if status == "need_clarification":
            return "need_clarification", "need_clarification", fallback_mode
        if status == "not_found":
            return "no_result", "no_result", fallback_mode
        return "completed", str(tool_result.get("tool") or selected_tool or "unknown"), fallback_mode

    @staticmethod
    def _build_payload_from_tool(tool_result: Dict[str, Any], *, debug_mode: bool = False) -> Dict[str, Any]:
        tool = str(tool_result.get("tool") or "")
        message = str(tool_result.get("message") or "")

        if tool == "search_listings":
            return {
                "summary": str(tool_result.get("summary") or message),
                "top_options": tool_result.get("items") or [],
                "reasons": [f"Returned {int(tool_result.get('count') or 0)} listings"],
                "cautions": [] if tool_result.get("found") else ["No matching listing found"],
                "next_step": str(tool_result.get("next_step") or "Ban co the thu mo rong ngan sach hoac khu vuc."),
                "next_questions": [],
            }

        if tool == "explain_listing":
            listing = tool_result.get("listing") or {}
            analysis = tool_result.get("analysis") or {}
            summary_text = AgentRuntime._build_explain_summary(tool_result, listing)

            hard_reason_map = {
                "district_match": "Khu vuc phu hop voi tieu chi tim kiem",
                "property_type_match": "Loai hinh bat dong san dung nhu cau",
            }
            soft_reason_map = {
                "near_metro": "Gan giao thong cong cong/metro",
                "near_school": "Gan truong hoc",
                "wants_gym": "Co tien ich gym",
                "wants_pool": "Co ho boi",
            }
            reasons: list[str] = []
            for item in analysis.get("matched_hard_filters") or []:
                token = str(item or "").strip()
                reasons.append(hard_reason_map.get(token, token))
            for item in analysis.get("matched_soft_preferences") or []:
                token = str(item or "").strip()
                reasons.append(soft_reason_map.get(token, token))

            price_analysis = analysis.get("price_analysis") or {}
            location_analysis = analysis.get("location_analysis") or {}
            size_analysis = analysis.get("size_analysis") or {}

            budget_fit = str(price_analysis.get("budget_fit") or "").strip()
            if budget_fit and budget_fit.lower() not in {"unknown", "n/a", "na", "none", "null"}:
                reasons.append(f"Danh gia ngan sach: {budget_fit}")
            transit_note = str(location_analysis.get("transit_note") or "").strip()
            if transit_note:
                reasons.append(transit_note)
            size_note = str(size_analysis.get("note") or "").strip()
            if size_note:
                reasons.append(size_note)

            search_document = str(tool_result.get("search_document") or listing.get("search_document") or "")
            search_doc_highlights = AgentRuntime._extract_search_document_highlights(search_document)
            for highlight in search_doc_highlights[:2]:
                reasons.append(f"Tin dang de cap: {highlight}")

            cautions: list[str] = []
            legal_status = str(tool_result.get("legal_status") or "").strip()
            if not legal_status:
                cautions.append("Tin dang nay chua co thong tin phap ly ro rang")

            detail_option = {
                "detail_view": "full",
                "source": listing.get("source") or tool_result.get("source"),
                "listing_id": listing.get("listing_id") or tool_result.get("listing_id"),
                "listing_ref": tool_result.get("listing_ref") or AgentRuntime._format_listing_ref(tool_result, listing),
                "title": listing.get("title") or tool_result.get("title"),
                "url": listing.get("url") or tool_result.get("url"),
                "transaction_type": listing.get("transaction_type") or tool_result.get("transaction_type"),
                "property_type": listing.get("property_type") or tool_result.get("property_type"),
                "project": listing.get("project") or tool_result.get("project"),
                "city": listing.get("city") or tool_result.get("city"),
                "district": listing.get("district") or tool_result.get("district"),
                "ward": listing.get("ward") or tool_result.get("ward"),
                "street": listing.get("street") or tool_result.get("street"),
                "price_text": tool_result.get("price_text") or listing.get("price_text"),
                "area_text": tool_result.get("area_text") or listing.get("area_text"),
                "price_value_vnd": tool_result.get("price_value_vnd") or listing.get("price_value_vnd"),
                "area_m2": tool_result.get("area_m2") or listing.get("area_m2"),
                "bedrooms": tool_result.get("bedrooms") or listing.get("bedrooms"),
                "bathrooms": tool_result.get("bathrooms") or listing.get("bathrooms"),
                "floors": tool_result.get("floors") or listing.get("floors"),
                "frontage_width_m": tool_result.get("frontage_width_m") or listing.get("frontage_width_m"),
                "road_access_width_m": tool_result.get("road_access_width_m") or listing.get("road_access_width_m"),
                "legal_status": tool_result.get("legal_status") or listing.get("legal_status"),
                "direction": tool_result.get("direction") or listing.get("direction"),
                "structure": tool_result.get("structure") or listing.get("structure"),
                "interior": tool_result.get("interior") or listing.get("interior"),
                "access": tool_result.get("access") or listing.get("access"),
                "location_quality": tool_result.get("location_quality") or listing.get("location_quality"),
                "neighborhood_quality": tool_result.get("neighborhood_quality") or listing.get("neighborhood_quality"),
                "view": tool_result.get("view") or listing.get("view"),
                "suitable_for": tool_result.get("suitable_for") or listing.get("suitable_for"),
                "enrichment_matches": analysis.get("enrichment_matches") or tool_result.get("enrichment_matches") or [],
                "amenities_area": listing.get("amenities_area") or tool_result.get("amenities_area"),
                "amenities_building": listing.get("amenities_building") or tool_result.get("amenities_building"),
                "nearby_landmarks": listing.get("nearby_landmarks") or tool_result.get("nearby_landmarks"),
                "nearby_transport": listing.get("nearby_transport") or tool_result.get("nearby_transport"),
                "nearby_roads": listing.get("nearby_roads") or tool_result.get("nearby_roads"),
                "search_document": tool_result.get("search_document") or listing.get("search_document"),
                "similarity_summary": analysis.get("similarity_summary"),
                "fit_score": analysis.get("final_score"),
                "price_analysis": analysis.get("price_analysis") or {},
                "location_analysis": analysis.get("location_analysis") or {},
                "size_analysis": analysis.get("size_analysis") or {},
                "legal_analysis": analysis.get("legal_analysis") or {},
                "utilities": analysis.get("utilities") or [],
            }

            return {
                "summary": summary_text or message or "Listing explained",
                "top_options": [detail_option] if listing else [],
                "reasons": list(dict.fromkeys(str(item) for item in reasons if str(item).strip())),
                "cautions": ([] if tool_result.get("found") else ["Listing not found"]) + cautions,
                "next_step": "neu ban muon, minh co the so sanh can nay voi 1-2 lua chon trong danh sach de de quyet dinh hon",
                "next_questions": [],
            }

        if tool == "compare_listings":
            # Use detailed comparison formatter if listings are available
            listing_a = tool_result.get("listing_a") or {}
            listing_b = tool_result.get("listing_b") or {}
            user_query = str(tool_result.get("context_query") or tool_result.get("user_query") or "").strip()
            recommendation = tool_result.get("recommendation") or {}
            
            if listing_a and listing_b:
                # Format detailed comparison using spec format
                formatted_comparison = format_comparison(
                    listing_a=listing_a,
                    listing_b=listing_b,
                    user_query=user_query,
                    recommendation=recommendation,
                    user_profile=tool_result.get("user_profile") or recommendation.get("user_profile"),
                    debug_mode=debug_mode,
                )
                summary_text = formatted_comparison
            else:
                summary_text = message or "Cannot compare: missing listing(s)"
            
            return {
                "summary": summary_text,
                "top_options": [
                    {
                        "winner": recommendation.get("winner") or tool_result.get("winner"),
                        "winner_ref": recommendation.get("winner_ref") or tool_result.get("winner_ref"),
                        "listing_a_ref": recommendation.get("listing_ref_a") or tool_result.get("listing_ref_a"),
                        "listing_b_ref": recommendation.get("listing_ref_b") or tool_result.get("listing_ref_b"),
                    }
                ],
                "reasons": [str(recommendation.get("summary") or message or "")],
                "cautions": [] if tool_result.get("found") else ["Cannot compare due to missing listing"],
                "next_step": "Ban co the yeu cau tim them listing tuong tu de mo rong lua chon.",
                "next_questions": [],
            }

        if tool == "similar_listings":
            return {
                "summary": message or "Similar listings completed",
                "top_options": tool_result.get("items") or [],
                "reasons": [f"Returned {int(tool_result.get('count') or 0)} similar listings"],
                "cautions": [] if tool_result.get("found") else ["No similar listings found"],
                "next_step": "Ban co the yeu cau so sanh 2 listing bat ky trong danh sach.",
                "next_questions": [],
            }

        if tool == "suggest_area":
            missing_fields = tool_result.get("missing_fields") or []
            next_prompt = str(tool_result.get("next_clarification_prompt") or "").strip()
            query = str(tool_result.get("query") or tool_result.get("user_query") or "")
            if not query:
                query = str(tool_result.get("message") or "")
            need_clarification = bool(tool_result.get("need_clarification"))
            top_options = tool_result.get("area_rankings") or tool_result.get("area_recommendations") or []
            if need_clarification:
                summary_text = (
                    next_prompt
                    or "Mình cần thêm thông tin để tư vấn chuẩn hơn: bạn cho mình biết ngân sách, khu vực ưu tiên và loại bất động sản mong muốn nhé."
                )
            else:
                summary_text = AgentRuntime._build_suggest_area_investment_summary(query, tool_result)
            return {
                "summary": summary_text,
                "top_options": [] if need_clarification else list(top_options)[:3],
                "reasons": ["Xếp hạng khu vực theo ngân sách, loại hình và nguồn cung listing"],
                "cautions": [] if not missing_fields else ["Cần thêm bối cảnh để tư vấn sát hơn"],
                "next_step": next_prompt or "Hãy chọn một khu trong danh sách, mình sẽ lọc listing phù hợp ngay.",
                "next_questions": AgentRuntime._suggest_area_followup_questions([str(field) for field in missing_fields]),
                "disable_llm_compose": True,
            }

        if tool == "analytics_listings":
            results = tool_result.get("results") or {}
            query = str(tool_result.get("query") or "").strip()
            parsed_query = parse_user_query(query) if query else None
            market_overview_mode = bool(parsed_query and parsed_query.use_case == "market_overview")
            count = int(results.get("total_count") or 0)
            avg_price = results.get("avg_price_vnd")
            min_price = results.get("min_price_vnd")
            max_price = results.get("max_price_vnd")
            avg_area = results.get("avg_area_m2")
            metric_type = str(results.get("metric_type") or "default")
            district_breakdown = results.get("district_breakdown") or []
            max_ppm = results.get("max_price_per_m2_vnd")
            max_ppm_listing = results.get("max_price_per_m2_listing") or {}
            location = results.get("location_context", "")
            
            reasons = []
            if count > 0:
                reasons.append(f"Tim thay {count} listing")
                if metric_type == "avg_area_m2":
                    if avg_area is not None:
                        reasons.append(f"Dien tich trung binh: {avg_area:.2f} m2")
                elif metric_type == "max_price_per_m2":
                    if max_ppm is not None:
                        reasons.append(f"Gia/m2 cao nhat: {max_ppm / 1_000_000:.2f} trieu/m2")
                        listing_ref = str(max_ppm_listing.get("title") or max_ppm_listing.get("listing_id") or "").strip()
                        if listing_ref:
                            reasons.append(f"Listing: {listing_ref}")
                else:
                    if avg_price is not None:
                        reasons.append(f"Gia trung binh: {avg_price / 1_000_000_000:.2f} ty")
                    if min_price is not None and max_price is not None:
                        reasons.append(f"Gia tu {min_price / 1_000_000_000:.2f} den {max_price / 1_000_000_000:.2f} ty")
                    if district_breakdown:
                        breakdown_text = "; ".join(
                            f"{int(item.get('count') or 0)} o {str(item.get('district') or '').strip()}"
                            for item in district_breakdown[:4]
                            if str(item.get('district') or '').strip()
                        )
                        if breakdown_text:
                            reasons.append(f"Phan bo theo khu vuc: {breakdown_text}")

            next_step = "Bạn có thể tìm listing cụ thể hoặc tiêu chí chi tiết hơn." if count > 0 else "Vui lòng thay đổi tiêu chí tìm kiếm."
            next_questions: list[str] = []
            if market_overview_mode:
                next_step = "Sau khi nắm mặt bằng giá, mình sẽ giúp chốt ngân sách rồi tìm listing phù hợp."
                if parsed_query and not parsed_query.hard_filters.property_type:
                    next_questions.append("Bạn đang ưu tiên căn hộ, nhà phố hay đất nền (hoặc muốn xem tổng quan tất cả)?")
            if metric_type == "avg_area_m2" and count > 0:
                next_step = "Bạn có thể bổ sung khu vực, ngân sách, hoặc số phòng ngủ để thu hẹp kết quả diện tích."
            if metric_type == "max_price_per_m2" and count > 0:
                next_step = "Ban co the yeu cau so sanh them cac listing theo gia/m2 o cung khu vuc."
            
            return {
                "summary": message or f"Co {count} listing trong {location}",
                "top_options": [],
                "reasons": reasons,
                "cautions": [] if count > 0 else ["Không tìm thấy dữ liệu"],
                "next_step": next_step,
                "next_questions": next_questions,
            }

        return {
            "summary": message or "No response",
            "top_options": [],
            "reasons": [],
            "cautions": [],
            "next_step": "",
            "next_questions": [],
        }

    @staticmethod
    def _combine_payloads(payloads: list[Dict[str, Any]]) -> Dict[str, Any]:
        if not payloads:
            return {
                "summary": "Không có dữ liệu để trả lời.",
                "top_options": [],
                "reasons": [],
                "cautions": ["Không có kết quả từ công cụ"],
                "next_step": "Vui lòng bổ sung thêm tiêu chí tìm kiếm.",
                "next_questions": [],
            }

        top_options = []
        reasons = []
        cautions = []
        next_questions = []
        summaries = []
        next_step = ""
        disable_llm_compose = False

        for payload in payloads:
            if payload.get("summary"):
                summaries.append(str(payload.get("summary")))
            top_options.extend(payload.get("top_options") or [])
            reasons.extend(payload.get("reasons") or [])
            cautions.extend(payload.get("cautions") or [])
            next_questions.extend(payload.get("next_questions") or [])
            if payload.get("next_step"):
                next_step = str(payload.get("next_step"))
            disable_llm_compose = disable_llm_compose or bool(payload.get("disable_llm_compose"))

        return {
            "summary": summaries[-1] if summaries else "Da xu ly yeu cau.",
            "top_options": top_options,
            "reasons": list(dict.fromkeys(str(item) for item in reasons if str(item).strip())),
            "cautions": list(dict.fromkeys(str(item) for item in cautions if str(item).strip())),
            "next_step": next_step or "Ban co the yeu cau toi toi uu lai bo loc.",
            "next_questions": list(dict.fromkeys(str(item) for item in next_questions if str(item).strip())),
            "disable_llm_compose": disable_llm_compose,
        }

    @staticmethod
    def _render_response_message(payload: Dict[str, Any]) -> str:
        summary_text = AgentRuntime._strip_urls_and_markdown_links(str(payload.get("summary") or ""))
        lines = [summary_text]
        reasons = payload.get("reasons") or []
        cautions = payload.get("cautions") or []
        next_step = str(payload.get("next_step") or "").strip()
        next_questions = payload.get("next_questions") or []

        if reasons:
            lines.append("Diem phu hop: " + "; ".join(str(item) for item in reasons[:3]))
        if cautions:
            lines.append("Luu y them: " + "; ".join(str(item) for item in cautions[:3]))
        if next_questions:
            lines.append("De chot nhanh, minh can them: " + "; ".join(str(item) for item in next_questions[:3]))
        if next_step:
            lines.append("Neu ban muon, buoc tiep theo la: " + next_step)
        return "\n".join(line for line in lines if line.strip())

    @staticmethod
    def _looks_like_system_error_response(text: str) -> bool:
        normalized = AgentRuntime._normalize_text(str(text or ""))
        if not normalized:
            return False
        error_patterns = [
            "he thong dang gap loi",
            "khong the cung cap thong tin",
            "khong the truy xuat du lieu",
            "thu lai voi tieu chi khac",
            "rat tiec",
        ]
        return any(pattern in normalized for pattern in error_patterns)

    def _compose_response_with_gemini(
        self,
        *,
        query: str,
        payload: Dict[str, Any],
        conversation_context: str = "",
    ) -> str | None:
        if genai is None:
            return None

        api_key = str(
            (
                os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or ""
            )
        ).strip()
        if not api_key:
            return None

        model_name = str(self.runtime_flags.get("llm_model") or "gemini-2.5-flash-lite").strip()
        compact_payload = {
            "summary": payload.get("summary"),
            "reasons": (payload.get("reasons") or [])[:5],
            "cautions": (payload.get("cautions") or [])[:5],
            "next_step": payload.get("next_step"),
            "next_questions": (payload.get("next_questions") or [])[:5],
            "top_options": (payload.get("top_options") or [])[:5],
        }
        allowed_urls = {
            str(item.get("url") or "").strip()
            for item in (compact_payload.get("top_options") or [])
            if isinstance(item, dict)
        }
        prompt = build_response_composer_prompt(
            query=query,
            payload=compact_payload,
            conversation_context=conversation_context,
        )

        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            text = str(getattr(response, "text", "") or "").strip()
            if not text:
                return None
            if len(text) > 1600:
                return None
            if "```" in text:
                return None
            if self._has_unknown_urls(text, allowed_urls=allowed_urls):
                return None
            if allowed_urls:
                text = self._strip_urls_and_markdown_links(text)
                if not text:
                    return None
            return text or None
        except Exception as exc:
            self.logger.warning("llm_response_composer_failed model=%s error=%s", model_name, exc)
            return None

    @staticmethod
    def _derive_outcome(tool_results: list[Dict[str, Any]], primary_tool: str) -> tuple[str, str, str]:
        domain_results = [result for result in tool_results if str(result.get("tool") or "") != "respond_to_user"]
        if not domain_results:
            return "completed", primary_tool, "none"

        for result in domain_results:
            status = str(result.get("status") or "").strip().lower()
            if status in {"invalid_input", "internal_error", "tool_error"}:
                fallback = str(result.get("fallback_mode") or status)
                return "failed", "error", fallback

        has_success = False
        for result in domain_results:
            status = str(result.get("status") or "").strip().lower()
            found = bool(result.get("found"))
            if status in {"ok", "completed"} or found:
                has_success = True
                break

        if has_success:
            fallback = str(domain_results[-1].get("fallback_mode") or "none")
            return "completed", primary_tool, fallback

        for result in domain_results:
            if str(result.get("status") or "").strip().lower() == "need_clarification":
                return "need_clarification", "need_clarification", str(result.get("fallback_mode") or "need_clarification")

        if domain_results and all(str(result.get("status") or "").strip().lower() == "not_found" for result in domain_results):
            fallback = str(domain_results[-1].get("fallback_mode") or "none")
            return "no_result", "no_result", fallback

        fallback = str(domain_results[-1].get("fallback_mode") or "none")
        return "completed", primary_tool, fallback

    @staticmethod
    def _plan_next_tool(current_tool: str, tool_result: Dict[str, Any], turn: int, max_turns: int) -> str | None:
        status = str(tool_result.get("status") or "").strip().lower()
        if current_tool == "respond_to_user":
            return None

        if turn >= max_turns:
            return None

        if status in {"invalid_input", "internal_error", "tool_error"}:
            return "respond_to_user"

        if current_tool == "suggest_area" and turn < max_turns - 1:
            # Keep suggest_area responses focused; only continue when user asks for listing search explicitly.
            return "respond_to_user"

        if status == "need_clarification":
            return "respond_to_user"

        return "respond_to_user"

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return default

    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _coerce_int(value: Any, default: int = 0, minimum: int | None = None, maximum: int | None = None) -> int:
        try:
            out = int(value)
        except (TypeError, ValueError):
            out = int(default)
        if minimum is not None:
            out = max(minimum, out)
        if maximum is not None:
            out = min(maximum, out)
        return out

    def _load_runtime_flags(self, config_path: str | None) -> Dict[str, Any]:
        default_flags: Dict[str, Any] = {
            "enable_llm_router_fallback": True,
            "enable_llm_response_composer": True,
            "enable_llm_input_preprocessor": True,
            "enable_llm_input_primary": True,
            "enable_best_effort_response": True,
            "llm_input_min_confidence": 0.65,
            "llm_input_timeout_ms": 1500,
            "llm_model": "gemini-2.5-flash-lite",
        }

        resolved_path = config_path or str(Path(__file__).resolve().parents[1] / "CONFIG" / "global.yaml")
        try:
            cfg = load_config(resolved_path)
        except Exception as exc:
            self.logger.warning(
                "runtime_flags_load_failed path=%s error=%s using_defaults=true",
                resolved_path,
                exc,
            )
            return default_flags

        runtime_cfg = cfg.get("AGENT_RUNTIME", {}) if isinstance(cfg, dict) else {}
        feature_cfg = runtime_cfg.get("FEATURE_FLAGS", {}) if isinstance(runtime_cfg, dict) else {}

        router_flag = self._coerce_bool(
            feature_cfg.get("LLM_ROUTER_FALLBACK", runtime_cfg.get("LLM_ROUTER_FALLBACK_ENABLED")),
            default=default_flags["enable_llm_router_fallback"],
        )
        response_flag = self._coerce_bool(
            feature_cfg.get("LLM_RESPONSE_COMPOSER", runtime_cfg.get("LLM_RESPONSE_COMPOSER_ENABLED")),
            default=default_flags["enable_llm_response_composer"],
        )
        input_preprocessor_flag = self._coerce_bool(
            feature_cfg.get("LLM_INPUT_PREPROCESSOR", runtime_cfg.get("LLM_INPUT_PREPROCESSOR_ENABLED")),
            default=default_flags["enable_llm_input_preprocessor"],
        )
        input_primary_flag = self._coerce_bool(
            feature_cfg.get("LLM_INPUT_PRIMARY", runtime_cfg.get("LLM_INPUT_PRIMARY_ENABLED")),
            default=default_flags["enable_llm_input_primary"],
        )
        best_effort_flag = self._coerce_bool(
            feature_cfg.get("BEST_EFFORT_RESPONSE", runtime_cfg.get("BEST_EFFORT_RESPONSE_ENABLED")),
            default=default_flags["enable_best_effort_response"],
        )
        llm_input_min_confidence = self._coerce_float(
            runtime_cfg.get("LLM_INPUT_MIN_CONFIDENCE"),
            default=default_flags["llm_input_min_confidence"],
        )
        llm_input_min_confidence = min(1.0, max(0.0, llm_input_min_confidence))
        llm_input_timeout_ms = self._coerce_int(
            runtime_cfg.get("LLM_INPUT_TIMEOUT_MS"),
            default=default_flags["llm_input_timeout_ms"],
            minimum=200,
            maximum=10_000,
        )

        model_name = str(runtime_cfg.get("LLM_MODEL") or "").strip() or default_flags["llm_model"]
        return {
            "enable_llm_router_fallback": router_flag,
            "enable_llm_response_composer": response_flag,
            "enable_llm_input_preprocessor": input_preprocessor_flag,
            "enable_llm_input_primary": input_primary_flag,
            "enable_best_effort_response": best_effort_flag,
            "llm_input_min_confidence": llm_input_min_confidence,
            "llm_input_timeout_ms": llm_input_timeout_ms,
            "llm_model": model_name,
        }

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        process_sequence = []
        planner_mode = "rule_only"
        llm_used_for_routing = False
        llm_used_for_response = False
        llm_input_used = False
        llm_input_confidence = 0.0
        llm_input_fallback_reason = "disabled"
        llm_input_prompt_version = INPUT_UNDERSTANDING_PROMPT_VERSION
        llm_input_context_turns = 0
        llm_input_applied = False
        llm_canonical_filters: Dict[str, Any] = {}
        normalized_slots: Dict[str, Any] = {}
        selected_tool = None
        fallback_mode = "none"
        selected_tool_history: list[str] = []

        validation = self.task_handler.validate_input_task(task)
        if not validation["valid"]:
            process_sequence.append(
                self.task_handler.build_process_step(
                    step="input_validation",
                    status="failed",
                    tool="task_handler",
                    summary="; ".join(validation["errors"]),
                    error_type="validation_error",
                )
            )
            final_response = self.formatter.build_final_response(
                summary="Yêu cầu không hợp lệ.",
                top_options=[],
                reasons=[],
                cautions=validation["errors"],
                next_step="Vui long gui lai payload co truong messages hop le.",
                next_questions=[],
            )
            output = self.formatter.build_output(
                state="failed",
                branch="validation",
                process_sequence=process_sequence,
                session_id=str(task.get("sessionId") or ""),
                use_case=None,
                matched_signals=[],
                retrieval_stats={},
                confidence=0.0,
                fallback_mode="validation_error",
                final_response=final_response,
                planner_mode=planner_mode,
                llm_used_for_routing=llm_used_for_routing,
                llm_used_for_response=llm_used_for_response,
                llm_model=self.runtime_flags["llm_model"],
                selected_tool=selected_tool,
            )
            self.logger.info(
                "runtime_execution_result session_id=%s selected_tool=%s fallback_mode=%s planner_mode=%s",
                str(task.get("sessionId") or ""),
                selected_tool,
                "validation_error",
                planner_mode,
            )
            output["metadata"]["startedAt"] = self.task_handler.now_iso()
            output["metadata"]["duration_ms"] = int((time.perf_counter() - started) * 1000)
            return output

        session_id = str(task.get("sessionId") or uuid.uuid4().hex)
        normalized_task = self.task_handler.normalize_input_task(task, session_id=session_id)
        parsed = parse_task_input(normalized_task)
        debug_mode = self._coerce_bool(parsed.metadata.get("debug_mode"), default=False)

        request_router_flag = self._coerce_bool(
            parsed.metadata.get("enable_llm_router_fallback"),
            default=bool(self.runtime_flags["enable_llm_router_fallback"]),
        )
        request_response_flag = self._coerce_bool(
            parsed.metadata.get("enable_llm_response_composer"),
            default=bool(self.runtime_flags["enable_llm_response_composer"]),
        )
        request_input_preprocessor_flag = self._coerce_bool(
            parsed.metadata.get("enable_llm_input_preprocessor"),
            default=bool(self.runtime_flags["enable_llm_input_preprocessor"]),
        )
        request_input_primary_flag = self._coerce_bool(
            parsed.metadata.get("enable_llm_input_primary"),
            default=bool(self.runtime_flags["enable_llm_input_primary"]),
        )
        request_best_effort_flag = self._coerce_bool(
            parsed.metadata.get("enable_best_effort_response"),
            default=bool(self.runtime_flags["enable_best_effort_response"]),
        )
        planner_mode = "llm_first_with_rule_fallback" if request_router_flag else "rule_only"
        if request_input_preprocessor_flag:
            llm_input_fallback_reason = "enabled_waiting_parse"
        else:
            llm_input_fallback_reason = "disabled"

        self.logger.info(
            "runtime_mode_selected session_id=%s planner_mode=%s llm_router_fallback_enabled=%s llm_response_composer_enabled=%s llm_input_preprocessor_enabled=%s llm_input_primary_enabled=%s llm_input_min_confidence=%s llm_input_timeout_ms=%s best_effort_enabled=%s llm_model=%s",
            session_id,
            planner_mode,
            request_router_flag,
            request_response_flag,
            request_input_preprocessor_flag,
            request_input_primary_flag,
            self.runtime_flags["llm_input_min_confidence"],
            self.runtime_flags["llm_input_timeout_ms"],
            request_best_effort_flag,
            self.runtime_flags["llm_model"],
        )

        process_sequence.append(
            self.task_handler.build_process_step(
                step="input_validation",
                status="ok",
                tool="task_handler",
                summary="Input payload normalized and parsed",
                session_id=session_id,
            )
        )

        self.session_manager.append_messages(
            session_id,
            [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                }
                for msg in parsed.messages
            ],
        )
        self.session_manager.merge_metadata(session_id, parsed.metadata)

        query = self.task_handler.extract_user_query(
            [{"role": msg.role, "content": msg.content} for msg in parsed.messages]
        )
        conversation_context = self._build_conversation_context(session_id, max_messages=10)
        llm_input_context_turns = len([line for line in conversation_context.splitlines() if line.strip()])

        extracted_refs = self._extract_listing_refs_from_query(query)
        for key, value in extracted_refs.items():
            parsed.metadata.setdefault(key, value)

        query_listing_indices = self._extract_listing_indices_from_query(query)
        selected_refs_from_indices: list[str] = []
        if query_listing_indices:
            session_state = self.session_manager.get_or_create(session_id)
            ordered_refs = session_state.metadata.get("last_browse_listing_refs")
            if not isinstance(ordered_refs, list) or not ordered_refs:
                ordered_refs = session_state.metadata.get("last_turn_listing_refs")
            if isinstance(ordered_refs, list) and ordered_refs:
                selected_refs: list[str] = []
                for one_based_idx in query_listing_indices:
                    zero_based_idx = one_based_idx - 1
                    if zero_based_idx < 0 or zero_based_idx >= len(ordered_refs):
                        continue
                    candidate = str(ordered_refs[zero_based_idx] or "").strip()
                    if candidate and "/" in candidate and candidate not in selected_refs:
                        selected_refs.append(candidate)

                if selected_refs:
                    selected_refs_from_indices = list(selected_refs)
                    parsed.metadata.setdefault("listing_ref", selected_refs[0])
                    if len(selected_refs) >= 2:
                        parsed.metadata.setdefault("listing_ref_a", selected_refs[0])
                        parsed.metadata.setdefault("listing_ref_b", selected_refs[1])

        # Resolve deictic phrases like "can nay/can do/2 can tren" from session memory.
        self._apply_followup_listing_refs(session_id=session_id, query=query, metadata=parsed.metadata)

        # Convenience behavior: for compare intent without explicit refs,
        # auto-pick 2 most recent browse results.
        if self._is_compare_intent_query(query):
            ref_a = str(parsed.metadata.get("listing_ref_a") or "").strip()
            ref_b = str(parsed.metadata.get("listing_ref_b") or "").strip()
            if not (ref_a and ref_b):
                session_state = self.session_manager.get_or_create(session_id)
                ordered_refs = session_state.metadata.get("last_browse_listing_refs")
                if not isinstance(ordered_refs, list) or len(ordered_refs) < 2:
                    ordered_refs = session_state.metadata.get("last_turn_listing_refs")
                if not isinstance(ordered_refs, list) or len(ordered_refs) < 2:
                    ordered_refs = session_state.metadata.get("recent_listing_refs")

                if isinstance(ordered_refs, list) and ordered_refs:
                    fallback_refs: list[str] = [
                        str(item).strip()
                        for item in ordered_refs
                        if str(item).strip() and "/" in str(item).strip()
                    ]

                    selected_compare_refs: list[str] = []
                    for ref in selected_refs_from_indices:
                        if ref not in selected_compare_refs:
                            selected_compare_refs.append(ref)
                    for ref in fallback_refs:
                        if ref not in selected_compare_refs:
                            selected_compare_refs.append(ref)
                        if len(selected_compare_refs) >= 2:
                            break

                    if len(selected_compare_refs) >= 2:
                        parsed.metadata.setdefault("listing_ref_a", selected_compare_refs[0])
                        parsed.metadata.setdefault("listing_ref_b", selected_compare_refs[1])
                        parsed.metadata.setdefault("listing_refs", selected_compare_refs[:2])

        has_explicit_listing_ref = any(
            self._is_valid_listing_ref(parsed.metadata.get(key))
            for key in ("listing_ref", "listing_ref_a", "listing_ref_b")
        )
        if not has_explicit_listing_ref and self._should_autofill_listing_ref(query):
            session_state = self.session_manager.get_or_create(session_id)
            last_listing_ref = str(session_state.metadata.get("last_listing_ref") or "").strip()
            if last_listing_ref:
                parsed.metadata["listing_ref"] = last_listing_ref

        # Policy guard: block/redirect adversarial requests before any retrieval routing.
        if self._detect_out_of_scope_or_adversarial(query):
            parsed.metadata["selected_tool"] = "respond_to_user"
            parsed.metadata["guard_response_message"] = (
                "Tôi không thể hỗ trợ yêu cầu truy cập dữ liệu nội bộ hoặc vượt quá chính sách an toàn. "
                "Neu ban muon, toi co the ho tro phan tich tong hop thi truong bat dong san o muc an toan."
            )

        # Clarification guard: avoid guessing filters for highly ambiguous buyer intent.
        if str(parsed.metadata.get("selected_tool") or "").strip().lower() != "respond_to_user":
            clarification_prompt = self._detect_ambiguous_clarification_prompt(query)
            if clarification_prompt:
                parsed.metadata["selected_tool"] = "respond_to_user"
                parsed.metadata["guard_response_message"] = clarification_prompt

        forced_tool = self._forced_tool_from_metadata(parsed.metadata)
        llm_primary_tool: str | None = None
        if request_input_preprocessor_flag and not forced_tool:
            llm_input_metadata = dict(parsed.metadata)
            llm_input_metadata["conversation_context"] = conversation_context
            llm_understanding = self._parse_user_input_with_gemini(query=query, metadata=llm_input_metadata)
            if llm_understanding is None:
                llm_input_fallback_reason = "parse_failed_or_unavailable"
            else:
                llm_input_used = True
                llm_input_confidence = float(getattr(llm_understanding, "confidence", 0.0) or 0.0)
                normalized_slots = {
                    "intent": str(getattr(llm_understanding, "intent", "") or ""),
                    "slots": getattr(llm_understanding, "slots", {}) or {},
                    "listing_refs": getattr(llm_understanding, "listing_refs", {}) or {},
                    "user_profile": getattr(llm_understanding, "user_profile", {}) or {},
                    "missing_slots": getattr(llm_understanding, "missing_slots", []) or [],
                    "clarification_question": str(getattr(llm_understanding, "clarification_question", "") or ""),
                    "safety": {
                        "pii_risk": str(getattr(getattr(llm_understanding, "safety", None), "pii_risk", "low") or "low"),
                        "hallucination_risk": str(getattr(getattr(llm_understanding, "safety", None), "hallucination_risk", "low") or "low"),
                    },
                }
                if llm_input_confidence < float(self.runtime_flags["llm_input_min_confidence"]):
                    llm_input_fallback_reason = "low_confidence"
                else:
                    llm_input_fallback_reason = "parsed_high_confidence"
                    canonical_slots = canonicalize_llm_slots(normalized_slots.get("slots") or {})
                    parser_hard_filters = {
                        key: value
                        for key, value in vars(parse_user_query(query).hard_filters).items()
                        if value is not None and str(value).strip() != ""
                    }
                    merged_filters = dict(canonical_slots)
                    # Keep deterministic parser filters from user query as the source of truth when available.
                    merged_filters.update(parser_hard_filters)
                    llm_canonical_filters = dict(merged_filters)

                    # Assist mode: merge normalized slots/profile/refs into metadata for tool execution.
                    for k, v in merged_filters.items():
                        if v is not None and str(v).strip() != "":
                            parsed.metadata.setdefault(k, v)
                    if merged_filters:
                        parsed.metadata["parsed_filters"] = merged_filters
                    for k, v in (normalized_slots.get("listing_refs") or {}).items():
                        if v is not None and str(v).strip() != "":
                            parsed.metadata.setdefault(k, v)
                    user_profile = normalized_slots.get("user_profile") or {}
                    if isinstance(user_profile, dict) and user_profile:
                        existing_profile = parsed.metadata.get("user_profile")
                        if not isinstance(existing_profile, dict):
                            existing_profile = {}
                        merged_profile = dict(existing_profile)
                        for k, v in user_profile.items():
                            if v is not None and str(v).strip() != "":
                                merged_profile.setdefault(k, v)
                        parsed.metadata["user_profile"] = merged_profile

                    intent = str(normalized_slots.get("intent") or "").strip().lower()
                    if request_input_primary_flag and intent in self._allowed_tools():
                        llm_primary_tool = intent
                        llm_input_applied = True
                        llm_input_fallback_reason = "primary_applied"
                    else:
                        llm_input_fallback_reason = "assist_applied"
        elif request_input_preprocessor_flag and forced_tool:
            llm_input_fallback_reason = "forced_tool_bypass"

        # Ensure listing refs are strict source/id values before routing/tool execution.
        self._sanitize_listing_ref_metadata(parsed.metadata)

        # Re-apply deictic mapping after LLM slot merge/sanitization.
        self._apply_followup_listing_refs(session_id=session_id, query=query, metadata=parsed.metadata)

        has_valid_listing_ref_after_llm = any(
            self._is_valid_listing_ref(parsed.metadata.get(key))
            for key in ("listing_ref", "listing_ref_a", "listing_ref_b")
        )
        if not has_valid_listing_ref_after_llm and self._should_autofill_listing_ref(query):
            session_state = self.session_manager.get_or_create(session_id)
            last_listing_ref = str(session_state.metadata.get("last_listing_ref") or "").strip()
            if self._is_valid_listing_ref(last_listing_ref):
                parsed.metadata["listing_ref"] = last_listing_ref

        max_turns = 3
        rule_tool = self._route_tool(query=query, metadata=parsed.metadata)
        llm_router_tool: str | None = None
        if llm_primary_tool:
            llm_used_for_routing = True
        if request_router_flag and not forced_tool and not llm_primary_tool:
            llm_router_tool = self._route_tool_with_gemini(query=query, metadata=parsed.metadata)
            if llm_router_tool:
                llm_used_for_routing = True

        text_for_override = self._normalize_text(query)
        compare_avg_area_query = (
            any(token in text_for_override for token in ["so sanh", "vs", "giua"])
            and any(token in text_for_override for token in ["gia trung binh", "trung binh", "average"])
            and any(token in text_for_override for token in ["quan", "huyen", "phuong", "xa", "district", "ward"])
        )
        family_ranking_query = (
            any(token in text_for_override for token in ["gia dinh", "family"])
            and any(token in text_for_override for token in ["nhieu listing", "nhieu nhat", "top", "cao nhat", "thap nhat"])
            and any(token in text_for_override for token in ["khu", "khu vuc", "quan", "huyen", "phuong", "xa"])
        )
        if rule_tool == "analytics_listings" and llm_router_tool and (compare_avg_area_query or family_ranking_query):
            llm_router_tool = None
            llm_used_for_routing = False

        if rule_tool == "analytics_listings" and (compare_avg_area_query or family_ranking_query):
            llm_primary_tool = None
            llm_router_tool = None
            llm_used_for_routing = False

        current_tool = llm_primary_tool or llm_router_tool or rule_tool

        if current_tool in {"explain_listing", "similar_listings"} and not self._is_valid_listing_ref(parsed.metadata.get("listing_ref")):
            session_state = self.session_manager.get_or_create(session_id)
            last_listing_ref = str(session_state.metadata.get("last_listing_ref") or "").strip()
            if self._is_valid_listing_ref(last_listing_ref):
                parsed.metadata["listing_ref"] = last_listing_ref

        selected_tool = current_tool
        tool_results: list[Dict[str, Any]] = []
        payloads: list[Dict[str, Any]] = []

        for turn in range(1, max_turns + 1):
            tool_for_turn = current_tool
            if turn == max_turns and tool_for_turn != "respond_to_user":
                tool_for_turn = "respond_to_user"

            selected_tool_history.append(tool_for_turn)
            process_step_status = "ok"
            process_step_summary = f"Executed tool {tool_for_turn} (turn {turn})"

            turn_metadata = dict(parsed.metadata)
            if tool_for_turn == "respond_to_user":
                merged_payload = self._combine_payloads(payloads)
                guard_message = str(turn_metadata.get("guard_response_message") or "").strip()
                if guard_message and not payloads:
                    final_response_text = guard_message
                else:
                    final_response_text = self._render_response_message(merged_payload)
                last_domain_result = next(
                    (res for res in reversed(tool_results) if str(res.get("tool") or "") != "respond_to_user"),
                    {},
                )
                last_domain_tool = str(last_domain_result.get("tool") or "").strip().lower()
                skip_llm_compose_for_compare = last_domain_tool == "compare_listings"
                if (
                    request_response_flag
                    and not guard_message
                    and not bool(merged_payload.get("disable_llm_compose"))
                    and not skip_llm_compose_for_compare
                ):
                    llm_text = self._compose_response_with_gemini(
                        query=query,
                        payload=merged_payload,
                        conversation_context=self._build_conversation_context(session_id, max_messages=12),
                    )
                    if llm_text:
                        last_status = str(last_domain_result.get("status") or "").strip().lower()
                        if last_status in {"ok", "completed"} and self._looks_like_system_error_response(llm_text):
                            final_response_text = self._render_response_message(merged_payload)
                        else:
                            final_response_text = llm_text
                            llm_used_for_response = True
                turn_metadata["final_response"] = final_response_text
                turn_metadata["message"] = turn_metadata["final_response"]

            route_started = time.perf_counter()
            try:
                tool_result = self._execute_tool(tool_for_turn, query=query, metadata=turn_metadata)
            except Exception as exc:
                self.logger.exception(
                    "runtime_tool_execution_failed session_id=%s tool=%s turn=%s error=%s",
                    session_id,
                    tool_for_turn,
                    turn,
                    exc,
                )
                tool_result = {
                    "tool": tool_for_turn,
                    "status": "tool_error",
                    "found": False,
                    "message": f"Tool execution failed: {exc}",
                    "retrieval_stats": {},
                    "matched_signals": [],
                    "use_case": "house_buy",
                    "fallback_mode": "tool_error",
                }
                process_step_status = "failed"
                process_step_summary = f"Tool {tool_for_turn} failed (turn {turn})"
            route_latency_ms = int((time.perf_counter() - route_started) * 1000)

            process_sequence.append(
                self.task_handler.build_process_step(
                    step=f"route_and_execute_turn_{turn}",
                    status=process_step_status,
                    tool=tool_result.get("tool") or tool_for_turn,
                    summary=process_step_summary,
                    session_id=session_id,
                    latency_ms=route_latency_ms,
                    fallback_reason=str(tool_result.get("fallback_mode") or "none"),
                    retrieval_mode=(tool_result.get("retrieval_stats") or {}).get("retrieval_mode"),
                    error_type="tool_error" if process_step_status == "failed" else None,
                )
            )

            tool_results.append(tool_result)
            self._remember_listing_context(session_id=session_id, tool_result=tool_result)
            if (tool_result.get("tool") or tool_for_turn) != "respond_to_user":
                payloads.append(self._build_payload_from_tool(tool_result, debug_mode=debug_mode))

            next_tool = self._plan_next_tool(
                current_tool=str(tool_result.get("tool") or tool_for_turn),
                tool_result=tool_result,
                turn=turn,
                max_turns=max_turns,
            )
            if not next_tool:
                break
            current_tool = next_tool

        merged_payload = self._combine_payloads(payloads)
        respond_messages = [
            str(result.get("message") or "").strip()
            for result in tool_results
            if str(result.get("tool") or "") == "respond_to_user"
        ]
        respond_messages = [item for item in respond_messages if item]
        if respond_messages:
            existing_summary = str(merged_payload.get("summary") or "").strip()
            # Keep structured summary from domain tools (e.g. compare_listings)
            # and only fallback to respond_to_user text when summary is empty.
            if not existing_summary:
                merged_payload["summary"] = respond_messages[-1]

        # If LLM response composer produced a final text, prefer that as user-facing summary.
        # This keeps tool outputs structured while making the final answer natural.
        if llm_used_for_response and respond_messages:
            merged_payload["summary"] = respond_messages[-1]

        # When response text is composed by LLM, keep guidance in the natural
        # paragraph and suppress rigid next_step rendering in UI.
        if llm_used_for_response:
            merged_payload["next_step"] = ""
            merged_payload["next_questions"] = []

        final_response = self.formatter.build_final_response(
            summary=str(merged_payload.get("summary") or ""),
            top_options=merged_payload.get("top_options") or [],
            reasons=merged_payload.get("reasons") or [],
            cautions=merged_payload.get("cautions") or [],
            next_step=str(merged_payload.get("next_step") or ""),
            next_questions=merged_payload.get("next_questions") or [],
        )

        final_state, final_branch, fallback_mode = self._derive_outcome(tool_results, selected_tool or "search_listings")

        if final_state == "no_result" and request_best_effort_flag and (selected_tool or "") != "analytics_listings":
            final_state = "completed"
            final_branch = "best_effort_response"
            fallback_mode = "best_effort_response"
            final_response = self.formatter.build_final_response(
                summary=(
                    "Chua tim thay listing khop 100% voi bo loc hien tai. "
                    "Toi da tao goi y best effort de ban tiep tuc thu hep hoac mo rong tieu chi."
                ),
                top_options=final_response.get("top_options") or [],
                reasons=(final_response.get("reasons") or []) + ["Da thu tim theo bo loc hien tai"],
                cautions=(final_response.get("cautions") or []) + ["Không có kết quả khớp hoàn toàn"],
                next_step=(
                    final_response.get("next_step")
                    or "Hay thu mo rong ngan sach, khu vuc, hoac loai hinh bat dong san de tang kha nang tim thay."
                ),
                next_questions=final_response.get("next_questions") or [],
            )

        domain_results = [result for result in tool_results if str(result.get("tool") or "") != "respond_to_user"]
        last_domain_result = domain_results[-1] if domain_results else {}

        retrieval_stats: Dict[str, Any] = {}
        for result in domain_results:
            if result.get("retrieval_stats"):
                retrieval_stats = result.get("retrieval_stats") or retrieval_stats

        matched_signals: list[str] = []
        for result in domain_results:
            matched_signals.extend(result.get("matched_signals") or [])
        matched_signals = list(dict.fromkeys(str(item) for item in matched_signals if str(item).strip()))

        self.logger.info(
            "runtime_execution_result session_id=%s selected_tool=%s fallback_mode=%s planner_mode=%s",
            session_id,
            selected_tool,
            fallback_mode,
            planner_mode,
        )

        output = self.formatter.build_output(
            state=final_state,
            branch=final_branch,
            process_sequence=process_sequence,
            session_id=session_id,
            use_case=str(last_domain_result.get("use_case") or "house_buy"),
            matched_signals=matched_signals,
            retrieval_stats=retrieval_stats,
            confidence=0.75 if any(bool(item.get("found", False)) for item in domain_results) else 0.45,
            fallback_mode=fallback_mode,
            final_response=final_response,
            planner_mode=planner_mode,
            llm_used_for_routing=llm_used_for_routing,
            llm_used_for_response=llm_used_for_response,
            llm_model=self.runtime_flags["llm_model"],
            selected_tool=selected_tool,
        )
        output["metadata"]["llm_input_preprocessor_enabled"] = request_input_preprocessor_flag
        output["metadata"]["llm_input_primary_enabled"] = request_input_primary_flag
        output["metadata"]["llm_input_applied"] = llm_input_applied
        output["metadata"]["llm_input_used"] = llm_input_used
        output["metadata"]["llm_input_confidence"] = llm_input_confidence
        output["metadata"]["llm_input_fallback_reason"] = llm_input_fallback_reason
        output["metadata"]["llm_input_prompt_version"] = llm_input_prompt_version
        output["metadata"]["llm_input_context_turns"] = llm_input_context_turns
        output["metadata"]["llm_input_min_confidence"] = self.runtime_flags["llm_input_min_confidence"]
        output["metadata"]["llm_input_timeout_ms"] = self.runtime_flags["llm_input_timeout_ms"]
        output["metadata"]["normalized_slots"] = normalized_slots
        output["metadata"]["parsed_filters"] = llm_canonical_filters
        output["metadata"]["selected_tool_history"] = selected_tool_history
        output["metadata"]["react_turns"] = len(selected_tool_history)
        output["metadata"]["last_tool_result"] = last_domain_result
        output["metadata"]["startedAt"] = self.task_handler.now_iso()
        output["metadata"]["duration_ms"] = int((time.perf_counter() - started) * 1000)

        assistant_summary = str(final_response.get("summary") or "").strip()
        if assistant_summary:
            self.session_manager.append_messages(
                session_id,
                [
                    {
                        "role": "assistant",
                        "content": assistant_summary,
                        "timestamp": self.task_handler.now_iso(),
                    }
                ],
            )

        return output
