"""
Comparison formatter for side-by-side listing comparison.
Implements the detailed spec for comparison output format:
- Kết luận (conclusion with clear recommendation)
- So sánh nhanh (3-5 key differences as bullets)
- Bảng so sánh (markdown table with key metrics and ⭐ stars)
- Phân tích chi tiết (financial, space, usability)
- Trade-offs (for each listing)
- Gợi ý tiếp theo (next actions)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from google import genai
except Exception:
    genai = None

from agent.common import ensure_api_key_from_config, load_config, parse_user_query
from utils.vn_normalizer import normalize_query_pipeline


class ComparisonFormatter:
    """Formats listing comparison results according to detailed Vietnamese spec."""

    def __init__(self, config_path: str | None = None):
        self.max_enrichment_lines = 2  # Max enrichment items to include per field
        self.max_enrichment_chars = 150  # Max total chars for enrichment
        self.config_path = config_path or str(Path(__file__).resolve().parents[1] / "CONFIG" / "global.yaml")
        ensure_api_key_from_config(self.config_path)
        self.runtime_flags = self._load_runtime_flags()
        self.llm_model = str(self.runtime_flags.get("llm_model") or "gemini-2.5-flash-lite").strip()

    def _load_runtime_flags(self) -> Dict[str, Any]:
        try:
            cfg = load_config(self.config_path)
        except Exception:
            return {}

        runtime_cfg = cfg.get("AGENT_RUNTIME") if isinstance(cfg.get("AGENT_RUNTIME"), dict) else {}
        feature_flags = runtime_cfg.get("FEATURE_FLAGS") if isinstance(runtime_cfg.get("FEATURE_FLAGS"), dict) else {}
        return {
            "llm_response_composer": bool(feature_flags.get("LLM_RESPONSE_COMPOSER")),
            "llm_model": runtime_cfg.get("LLM_MODEL"),
        }

    def _should_use_llm_composer(self) -> bool:
        if genai is None:
            return False
        if not bool(self.runtime_flags.get("llm_response_composer")):
            return False
        api_key = str((os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "")).strip()
        return bool(api_key)

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

    @staticmethod
    def _serialize_query_filters(parsed_query: Any) -> Dict[str, Any]:
        if parsed_query is None:
            return {}

        hard_filters = getattr(parsed_query, "hard_filters", None)
        soft_preferences = getattr(parsed_query, "soft_preferences", None)
        user_profile = getattr(parsed_query, "user_profile", None)

        hard_payload = {
            key: getattr(hard_filters, key, None)
            for key in (
                "transaction_type",
                "property_type",
                "city",
                "district",
                "ward",
                "street",
                "project",
                "max_price_vnd",
                "min_price_vnd",
                "min_area_m2",
                "min_bedrooms",
                "min_bathrooms",
                "legal_status",
                "direction",
            )
        } if hard_filters is not None else {}

        soft_payload = {
            key: getattr(soft_preferences, key, None)
            for key in (
                "near_metro",
                "near_school",
                "family_friendly",
                "near_entertainment",
                "wants_gym",
                "wants_pool",
            )
        } if soft_preferences is not None else {}

        user_payload = {
            key: getattr(user_profile, key, None)
            for key in ("family_size", "has_children", "has_elderly", "commuting_destination")
        } if user_profile is not None else {}

        return {
            "hard_requirements": {k: v for k, v in hard_payload.items() if v not in (None, "", [], {})},
            "soft_preferences": {k: v for k, v in soft_payload.items() if v not in (None, "", [], {})},
            "user_profile": {k: v for k, v in user_payload.items() if v not in (None, "", [], {})},
        }

    @staticmethod
    def _compact_listing_payload(listing: Dict[str, Any]) -> Dict[str, Any]:
        keys = (
            "source",
            "listing_id",
            "title",
            "url",
            "price_value_vnd",
            "area_m2",
            "bedrooms",
            "bathrooms",
            "floors",
            "legal_status",
            "district",
            "city",
            "project",
            "property_type",
            "transaction_type",
            "location_quality",
            "neighborhood_quality",
            "view",
            "suitable_for",
            "amenities_area",
            "amenities_building",
            "nearby_landmarks",
            "nearby_transport",
            "nearby_roads",
        )
        return {key: listing.get(key) for key in keys if listing.get(key) not in (None, "", [], {})}

    @staticmethod
    def _build_confidence_details(
        listing_a: Dict[str, Any],
        listing_b: Dict[str, Any],
        parsed_query: Any,
        recommendation: Dict[str, Any],
    ) -> Dict[str, Any]:
        filled_fields = 0
        for listing in (listing_a, listing_b):
            for key in ("price_value_vnd", "area_m2", "bedrooms", "bathrooms", "legal_status", "district"):
                if listing.get(key) not in (None, "", []):
                    filled_fields += 1

        score = 0.35 + min(0.35, filled_fields * 0.05)
        if parsed_query is not None:
            hard_filters = getattr(parsed_query, "hard_filters", None)
            user_profile = getattr(parsed_query, "user_profile", None)
            if hard_filters and any(getattr(hard_filters, key, None) is not None for key in ("max_price_vnd", "district", "city", "min_bedrooms", "min_area_m2")):
                score += 0.15
            if user_profile and any(getattr(user_profile, key, None) not in (None, False) for key in ("family_size", "has_children", "has_elderly", "commuting_destination")):
                score += 0.1
        if recommendation.get("user_profile_used"):
            score += 0.05

        score = min(score, 0.95)
        if score >= 0.8:
            level = "High"
        elif score >= 0.6:
            level = "Medium"
        else:
            level = "Low"

        reasons = []
        if filled_fields >= 8:
            reasons.append("nhiều trường cấu trúc có sẵn")
        else:
            reasons.append("có thể thiếu một số tín hiệu mềm")
        if recommendation.get("user_profile_used"):
            reasons.append("đã có user profile để cân theo nhu cầu")
        else:
            reasons.append("phần persona chủ yếu suy luận từ query và thuộc tính listing")

        return {
            "score": round(score, 2),
            "level": level,
            "reasons": reasons,
            "text": f"Confidence: **{level}** ({score:.2f}).\nLý do: {', '.join(reasons)}.",
        }

    def _compose_compare_response(
        self,
        *,
        payload: Dict[str, Any],
    ) -> Dict[str, str] | None:
        if not self._should_use_llm_composer():
            return None

        prompt = (
            "Bạn là một writer cho phần so sánh bất động sản bằng tiếng Việt có dấu. "
            "Chỉ viết lại phần diễn giải từ dữ liệu đã được chuẩn hóa, KHÔNG tự tính lại điểm.\n\n"
            "Ràng buộc bắt buộc:\n"
            "- Do not change scores.\n"
            "- Do not invent missing facts.\n"
            "- Only explain using provided evidence.\n"
            "- If evidence is missing, say so.\n"
            "- Return concise Vietnamese output.\n"
            "- Output exactly one JSON object. The object MUST contain the following keys:\n"
            "  - persona_recommendation: string\n"
            "  - conclusion: string\n"
            "  - detailed_analysis: OBJECT with keys (financial, space, usability, location_connectivity, living_environment).\n"
            "      Each value should be a short string (1-3 sentences).\n"
            "  - tradeoffs: string\n"
            "  - confidence: string\n"
            "- Do NOT wrap the JSON in markdown fences; return raw JSON only.\n\n"
            f"Payload:\n{json.dumps(payload, ensure_ascii=False)}"

             "HƯỚNG DẪN CHO TỪNG PHẦN:\n"

                 "1. financial\n"
                 "- So sánh giá tổng, giá/m² hoặc chi phí nếu có.\n"
                 "- Giải thích ý nghĩa của chênh lệch.\n"
                 "- Không nói listing nào tốt hơn nếu không có evidence.\n\n"

                 "2. space\n"
                 "- Chỉ nói về diện tích, số phòng ngủ, số WC hoặc các thuộc tính không gian.\n"
                 "- Nếu có phần trăm chênh lệch thì nêu ngắn gọn.\n\n"

                 "3. usability\n"
                 "- Chỉ mô tả các mục đích sử dụng được evidence hỗ trợ.\n"
                 "- Không tự suy diễn 'phù hợp gia đình đông người' nếu không có dữ liệu về diện tích hoặc số phòng.\n\n"

                 "4. location_connectivity\n"
                 "- Chỉ mô tả vị trí, giao thông, metro, trường học, bệnh viện, tiện ích.\n"
                 "- Không lặp lại nội dung môi trường sống.\n\n"

                 "5. living_environment\n"
                 "- Chỉ mô tả môi trường sống như an ninh, yên tĩnh, dân trí, mật độ dân cư.\n"
                 "- Không lặp lại thông tin giao thông hoặc tiện ích.\n\n"

                 "6. tradeoffs\n"
                 "- Bắt buộc nêu ít nhất 1 điểm được và mất của mỗi bên nếu dữ liệu cho phép.\n"
                 "- Tránh kết luận một chiều.\n\n"

                 "7. persona_recommendation\n"
                 "- Nếu payload có persona_matches hoặc persona_reasoning thì sử dụng.\n"
                 "- Nếu không có dữ liệu persona, ghi rõ 'Chưa có đủ dữ liệu để cá nhân hóa khuyến nghị'.\n\n"

                 "8. conclusion\n"
                 "- Giải thích ngắn gọn vì sao listing thắng có điểm cao hơn.\n"
                 "- Chỉ sử dụng các yếu tố xuất hiện trong payload.\n"
                 "- Nếu điểm số chênh lệch nhỏ (<0.5 điểm), nhấn mạnh rằng hai lựa chọn tương đối cân bằng.\n"
                 "- Kết luận nên theo dạng 'Nếu ưu tiên X thì chọn A, nếu ưu tiên Y thì chọn B' khi phù hợp.\n\n"

                 "9. confidence\n"
                 "- Đánh giá độ tin cậy của phân tích.\n"
                 "- Nếu thiếu nhiều dữ liệu quan trọng (pháp lý, năm xây dựng, phí quản lý, nội thất...) thì giảm confidence.\n"
                 "- Trả về dạng ngắn gọn: Cao / Trung bình / Thấp kèm 1 câu giải thích.\n\n"
        )
        #     prompt = (
        #         "Bạn là chuyên gia phân tích bất động sản. "
        #         "Nhiệm vụ của bạn là diễn giải kết quả so sánh đã được tính toán sẵn thành tiếng Việt tự nhiên, ngắn gọn và có căn cứ.\n\n"

                
        #         "QUY TẮC QUAN TRỌNG:\n"
        #         "- KHÔNG tự tính lại điểm số.\n"
        #         "- KHÔNG thay đổi score, ranking hoặc winner.\n"
        #         "- KHÔNG suy diễn ngoài dữ liệu được cung cấp.\n"
        #         "- Mọi nhận định phải dựa trên evidence có trong payload.\n"
        #         "- Nếu thiếu dữ liệu để kết luận, phải nói rõ 'chưa có đủ dữ liệu'.\n"
        #         "- Không sử dụng các từ tuyệt đối như 'chắc chắn', 'tốt nhất', 'hoàn hảo'.\n"
        #         "- Không lặp lại cùng một ý ở nhiều section.\n"
        #         "- Ưu tiên giải thích trade-off thay vì chỉ tuyên bố listing nào tốt hơn.\n"
        #         "- Chỉ đề cập buyer persona nếu payload cung cấp evidence hỗ trợ.\n"
        #         "- Nếu payload có reasons hoặc evidence, hãy sử dụng chúng để giải thích.\n\n"

        #         "HƯỚNG DẪN CHO TỪNG PHẦN:\n"

        #         "1. financial\n"
        #         "- So sánh giá tổng, giá/m² hoặc chi phí nếu có.\n"
        #         "- Giải thích ý nghĩa của chênh lệch.\n"
        #         "- Không nói listing nào tốt hơn nếu không có evidence.\n\n"

        #         "2. space\n"
        #         "- Chỉ nói về diện tích, số phòng ngủ, số WC hoặc các thuộc tính không gian.\n"
        #         "- Nếu có phần trăm chênh lệch thì nêu ngắn gọn.\n\n"

        #         "3. usability\n"
        #         "- Chỉ mô tả các mục đích sử dụng được evidence hỗ trợ.\n"
        #         "- Không tự suy diễn 'phù hợp gia đình đông người' nếu không có dữ liệu về diện tích hoặc số phòng.\n\n"

        #         "4. location_connectivity\n"
        #         "- Chỉ mô tả vị trí, giao thông, metro, trường học, bệnh viện, tiện ích.\n"
        #         "- Không lặp lại nội dung môi trường sống.\n\n"

        #         "5. living_environment\n"
        #         "- Chỉ mô tả môi trường sống như an ninh, yên tĩnh, dân trí, mật độ dân cư.\n"
        #         "- Không lặp lại thông tin giao thông hoặc tiện ích.\n\n"

        #         "6. tradeoffs\n"
        #         "- Bắt buộc nêu ít nhất 1 điểm được và mất của mỗi bên nếu dữ liệu cho phép.\n"
        #         "- Tránh kết luận một chiều.\n\n"

        #         "7. persona_recommendation\n"
        #         "- Nếu payload có persona_matches hoặc persona_reasoning thì sử dụng.\n"
        #         "- Nếu không có dữ liệu persona, ghi rõ 'Chưa có đủ dữ liệu để cá nhân hóa khuyến nghị'.\n\n"

        #         "8. conclusion\n"
        #         "- Giải thích ngắn gọn vì sao listing thắng có điểm cao hơn.\n"
        #         "- Chỉ sử dụng các yếu tố xuất hiện trong payload.\n"
        #         "- Nếu điểm số chênh lệch nhỏ (<0.5 điểm), nhấn mạnh rằng hai lựa chọn tương đối cân bằng.\n"
        #         "- Kết luận nên theo dạng 'Nếu ưu tiên X thì chọn A, nếu ưu tiên Y thì chọn B' khi phù hợp.\n\n"

        #         "9. confidence\n"
        #         "- Đánh giá độ tin cậy của phân tích.\n"
        #         "- Nếu thiếu nhiều dữ liệu quan trọng (pháp lý, năm xây dựng, phí quản lý, nội thất...) thì giảm confidence.\n"
        #         "- Trả về dạng ngắn gọn: Cao / Trung bình / Thấp kèm 1 câu giải thích.\n\n"

        #         "OUTPUT:\n"
        #         "Trả về đúng 1 JSON object.\n"
        #         "Không markdown.\n"
        #         "Không giải thích ngoài JSON.\n\n"

        #         "{\n"
        #         '  "persona_recommendation": "...",\n'
        #         '  "conclusion": "...",\n'
        #         '  "detailed_analysis": {\n'
        #         '      "financial": "...",\n'
        #         '      "space": "...",\n'
        #         '      "usability": "...",\n'
        #         '      "location_connectivity": "...",\n'
        #         '      "living_environment": "..."\n'
        #         "  },\n"
        #         '  "tradeoffs": "...",\n'
        #         '  "confidence": "..."\n'
        #         "}\n\n"

        #         f"Payload:\n{json.dumps(payload, ensure_ascii=False)}"
                
        # )
            

        try:
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
            response = client.models.generate_content(
                model=self.llm_model,
                contents=prompt,
            )
            text = str(getattr(response, "text", "") or "").strip()
            if not text:
                return None

            parsed = self._extract_first_json_object(text)
            if not isinstance(parsed, dict):
                return None

            result: Dict[str, Any] = {}
            # Keep detailed_analysis as object if the model returned an object.
            for key in ("persona_recommendation", "conclusion", "detailed_analysis", "tradeoffs", "confidence"):
                if key not in parsed:
                    continue
                val = parsed.get(key)
                if key == "detailed_analysis" and isinstance(val, dict):
                    result[key] = val
                    continue
                # Otherwise coerce to string for textual fields
                value = str(val or "").strip()
                if value:
                    result[key] = value
            return result or None
        except Exception:
            return None

    def format_comparison(
        self,
        listing_a: Dict[str, Any],
        listing_b: Dict[str, Any],
        user_query: str | None = None,
        recommendation: Dict[str, Any] | None = None,
        user_profile: Dict[str, Any] | None = None,
        debug_mode: bool = False,
    ) -> str:
        """
        Format a comparison between two listings according to spec.
        
        Returns formatted markdown string with all required sections:
        - Kết luận
        - So sánh nhanh
        - Bảng so sánh
        - Phân tích chi tiết
        - Trade-offs
        - Gợi ý tiếp theo
        """
        if not listing_a or not listing_b:
            return "Không thể so sánh: thiếu thông tin listing."

        recommendation = recommendation or {}
        # Determine effective query source: prefer explicit user_query, then recommendation context
        effective_query = (
            (user_query or "").strip()
            or str(recommendation.get("context_query") or recommendation.get("user_query") or "").strip()
        )
        parsed_query = None
        if effective_query:
            try:
                parsed_query = parse_user_query(effective_query)
            except Exception:
                parsed_query = None
        soft_signals = self._extract_soft_preference_signals(getattr(parsed_query, "normalized_query", None) or effective_query or "")
        structured_filters = self._serialize_query_filters(parsed_query)

        # Extract key metrics
        price_a = self._safe_float(listing_a.get("price_value_vnd"))
        price_b = self._safe_float(listing_b.get("price_value_vnd"))
        area_a = self._safe_float(listing_a.get("area_m2"))
        area_b = self._safe_float(listing_b.get("area_m2"))
        bed_a = self._safe_int(listing_a.get("bedrooms"))
        bed_b = self._safe_int(listing_b.get("bedrooms"))
        bath_a = self._safe_int(listing_a.get("bathrooms"))
        bath_b = self._safe_int(listing_b.get("bathrooms"))
        floors_a = self._safe_int(listing_a.get("floors"))
        floors_b = self._safe_int(listing_b.get("floors"))
        legal_a = str(listing_a.get("legal_status") or "").strip()
        legal_b = str(listing_b.get("legal_status") or "").strip()

        # Compute price per m2
        ppm_a = price_a / area_a if price_a and area_a and area_a > 0 else None
        ppm_b = price_b / area_b if price_b and area_b and area_b > 0 else None

        score_a = self._score_listing(listing_a, listing_b, parsed_query, recommendation, "A", soft_signals)
        score_b = self._score_listing(listing_b, listing_a, parsed_query, recommendation, "B", soft_signals)
        confidence_details = self._build_confidence_details(listing_a, listing_b, parsed_query, recommendation)

        hard_requirement_check = self._build_hard_requirement_check(listing_a, listing_b, parsed_query)
        score_summary = self._build_scoring_summary(listing_a, listing_b, parsed_query, recommendation, soft_signals)
        decision_trace = self._build_decision_trace(listing_a, listing_b, parsed_query, recommendation, soft_signals)
        confidence = confidence_details["text"]
        persona_recommendation = self._build_persona_recommendation(listing_a, listing_b, parsed_query, user_profile, soft_signals)

        # Build sections
        conclusion = self._build_conclusion(
            listing_a,
            listing_b,
            price_a,
            price_b,
            area_a,
            area_b,
            bed_a,
            bed_b,
            parsed_query,
            recommendation,
            hard_requirement_check,
            score_summary,
            soft_signals,
        )
        quick_compare = self._build_quick_compare(
            listing_a, listing_b, price_a, price_b, ppm_a, ppm_b, area_a, area_b,
            bed_a, bed_b, bath_a, bath_b,
        )
        comparison_table = self._build_comparison_table(
            listing_a, listing_b, price_a, price_b, ppm_a, ppm_b, area_a, area_b,
            bed_a, bed_b, bath_a, bath_b, floors_a, floors_b, legal_a, legal_b
        )
        detailed_analysis = self._build_detailed_analysis(
            listing_a, listing_b, price_a, price_b, ppm_a, ppm_b,
            area_a, area_b, bed_a, bed_b
        )
        tradeoffs = self._build_tradeoffs(listing_a, listing_b)
        next_steps = self._build_next_steps()

        llm_payload = {
            "user_query": effective_query,
            "listing_a": self._compact_listing_payload(listing_a),
            "listing_b": self._compact_listing_payload(listing_b),
            "hard_requirements": structured_filters.get("hard_requirements") or {},
            "soft_preferences": [str(signal.get("label") or "").strip() for signal in soft_signals if str(signal.get("label") or "").strip()],
            "soft_preference_flags": structured_filters.get("soft_preferences") or {},
            "score_summary": {
                "A": score_a,
                "B": score_b,
                "margin": round(abs(score_a["total"] - score_b["total"]), 2),
                "winner": recommendation.get("winner") or ("A" if score_a["total"] > score_b["total"] else "B" if score_b["total"] > score_a["total"] else "tie"),
            },
            "evidence": {
                "hard_requirement_check": hard_requirement_check,
                "decision_trace": decision_trace,
                "comparison_table": comparison_table,
                "quick_compare": quick_compare,
                "detailed_analysis": detailed_analysis,
                "tradeoffs": tradeoffs,
            },
            "persona_candidates": {
                "A": self._generate_listing_persona(listing_a, "A"),
                "B": self._generate_listing_persona(listing_b, "B"),
            },
            "recommendation": {
                "winner": recommendation.get("winner") or ("A" if score_a["total"] > score_b["total"] else "B" if score_b["total"] > score_a["total"] else "tie"),
                "margin": round(abs(score_a["total"] - score_b["total"]), 2),
                "confidence": confidence_details["score"],
                "summary": recommendation.get("summary") or "",
            },
        }

        llm_sections = self._compose_compare_response(payload=llm_payload)
        if llm_sections:
            persona_recommendation = llm_sections.get("persona_recommendation") or persona_recommendation
            conclusion = llm_sections.get("conclusion") or conclusion

            # Handle structured detailed_analysis returned as an object by the LLM.
            detailed_from_llm = llm_sections.get("detailed_analysis")
            if isinstance(detailed_from_llm, dict):
                header_map = {
                    "financial": "### 💰 Tài chính",
                    "space": "### 📐 Không gian",
                    "usability": "###  Khả năng sử dụng",
                    "location_connectivity": "### 📍 Vị trí & Kết nối",
                    "living_environment": "### 🏠 Môi trường sống",
                }
                parts: List[str] = []
                for key in ("financial", "space", "usability", "location_connectivity", "living_environment"):
                    val = detailed_from_llm.get(key)
                    if val:
                        parts.append(f"{header_map.get(key)}\n{str(val).strip()}")
                if parts:
                    detailed_analysis = "\n\n".join(parts)
            elif isinstance(detailed_from_llm, str) and detailed_from_llm.strip():
                detailed_analysis = detailed_from_llm

            tradeoffs = llm_sections.get("tradeoffs") or tradeoffs
            confidence = llm_sections.get("confidence") or confidence

        # Assemble final output
        sections: List[str] = []
        if debug_mode and hard_requirement_check.strip():
            sections.extend([
                "## Tiêu chí bắt buộc",
                hard_requirement_check,
                "",
            ])

        sections.extend([
            "## 📈 Bảng đánh giá",
            score_summary,
            "",
            "## 📋 Bảng so sánh",
            comparison_table,
            "",
            quick_compare,
            "",
            "## 📊 Phân tích chi tiết",
            detailed_analysis,
            "",
            "## ⚖️ Trade-offs",
            tradeoffs,
            "",
            "## 👥 Mức độ phù hợp theo cá nhân",
            persona_recommendation,
            "",
            "## 🏆 Kết luận",
            conclusion,
            "",
            "## ➕ Gợi ý tiếp theo",
            next_steps,
        ])

        output = "\n".join(sections).rstrip()

        return output

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Safely convert value to float, return None if invalid."""
        try:
            if value is None:
                return None
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        """Safely convert value to int, return None if invalid."""
        try:
            if value is None:
                return None
            return int(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _to_markdown_table_cell(value: Any) -> str:
        """Render cell text safely for markdown tables while preserving line breaks."""
        text = str(value or "").strip()
        if not text:
            return "-"
        # Keep visual line breaks inside one table cell.
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
        # Prevent accidental extra columns when content contains '|'.
        text = text.replace("|", "\\|")
        return text

    def _build_conclusion(
        self,
        listing_a: Dict[str, Any],
        listing_b: Dict[str, Any],
        price_a: float | None,
        price_b: float | None,
        area_a: float | None,
        area_b: float | None,
        bed_a: int | None,
        bed_b: int | None,
        parsed_query: Any = None,
        recommendation: Dict[str, Any] | None = None,
        hard_requirement_check: str | None = None,
        score_summary: str | None = None,
        soft_signals: List[Dict[str, Any]] | None = None,
    ) -> str:
        """
        Build conclusion with one winner, then A/B suitability bullets.
        """
        recommendation = recommendation or {}
        winner = str(recommendation.get("winner") or "").strip().upper() or None
        reason = str(recommendation.get("summary") or "").strip()

        user_profile = getattr(parsed_query, "user_profile", None) if parsed_query is not None else None
        hard_filters = getattr(parsed_query, "hard_filters", None) if parsed_query is not None else None
        score_a = self._score_listing(listing_a, listing_b, parsed_query, recommendation, "A", soft_signals)
        score_b = self._score_listing(listing_b, listing_a, parsed_query, recommendation, "B", soft_signals)
        score_gap = abs(score_a["total"] - score_b["total"])

        primary, secondary = (listing_a, listing_b) if score_a["total"] >= score_b["total"] else (listing_b, listing_a)
        primary_label, secondary_label = ("A", "B") if score_a["total"] >= score_b["total"] else ("B", "A")

        primary_points, secondary_points = self._build_conclusion_points(primary, secondary)
        nuanced = score_gap <= 1.0 and bool(primary_points or secondary_points)

        if nuanced:
            reason = (
                f"nhỉnh hơn nhẹ nếu ưu tiên {', '.join(primary_points[:2]) or 'các tiêu chí chi tiết'}; "
                f"Listing {secondary_label} vẫn đáng cân nhắc nếu ưu tiên {', '.join(secondary_points[:2]) or 'các tiêu chí còn lại'}"
            )

        # Prefer fit score / user profile over price-only heuristics.
        if score_a["total"] != score_b["total"]:
            winner = "A" if score_a["total"] > score_b["total"] else "B"
            reason = f"điểm phù hợp cao hơn ({score_a['total']:.2f} so với {score_b['total']:.2f})"

        if hard_filters is not None:
            max_price = getattr(hard_filters, "max_price_vnd", None)
            min_bedrooms = getattr(hard_filters, "min_bedrooms", None)
            district_req = getattr(hard_filters, "district", None)
            city_req = getattr(hard_filters, "city", None)

            def _hard_pass(listing: Dict[str, Any]) -> bool:
                price = self._safe_float(listing.get("price_value_vnd"))
                bedrooms = self._safe_int(listing.get("bedrooms"))
                district = normalize_query_pipeline(str(listing.get("district") or listing.get("project") or ""))
                city = normalize_query_pipeline(str(listing.get("city") or ""))
                if max_price is not None and price is not None and price > float(max_price):
                    return False
                if min_bedrooms is not None and bedrooms is not None and bedrooms < int(min_bedrooms):
                    return False
                if district_req and normalize_query_pipeline(str(district_req)) not in district:
                    return False
                if city_req and normalize_query_pipeline(str(city_req)) not in city:
                    return False
                return True

            a_pass = _hard_pass(listing_a)
            b_pass = _hard_pass(listing_b)
            if a_pass and not b_pass:
                winner = "A"
                reason = "đáp ứng hard requirement tốt hơn"
            elif b_pass and not a_pass:
                winner = "B"
                reason = "đáp ứng hard requirement tốt hơn"

        # If still no strong winner, use space for family-style queries before price.
        family_size = self._safe_int(getattr(user_profile, "family_size", None) if user_profile is not None else None)
        if winner is None and family_size is not None and area_a and area_b and area_a != area_b:
            winner = "A" if area_a > area_b else "B"
            area_diff = abs(area_a - area_b)
            reason = f"diện tích lớn hơn {area_diff:.0f}m², hợp hơn cho gia đình {family_size} người"

        if winner is None and soft_signals:
            soft_a = self._soft_preference_fit_score(listing_a, soft_signals)
            soft_b = self._soft_preference_fit_score(listing_b, soft_signals)
            if soft_a != soft_b:
                winner = "A" if soft_a > soft_b else "B"
                reason = f"phù hợp hơn với nhu cầu {self._soft_signals_to_phrase(soft_signals)}"

        # Keep price as a fallback only.
        if winner is None and price_a and price_b and price_a != price_b:
            winner = "A" if price_a < price_b else "B"
            price_diff = abs(price_a - price_b)
            reason = f"giá tổng thấp hơn {self._format_price(price_diff)}"

        if winner is None and price_a and area_a and price_b and area_b:
            ppm_a = price_a / area_a if area_a > 0 else None
            ppm_b = price_b / area_b if area_b > 0 else None
            if ppm_a and ppm_b and ppm_a != ppm_b:
                winner = "A" if ppm_a < ppm_b else "B"
                reason = f"giá/m² thấp hơn khoảng {self._format_price_per_m2(abs(ppm_a - ppm_b))}"

        if winner is None and bed_a and bed_b and bed_a != bed_b:
            winner = "A" if bed_a > bed_b else "B"
            reason = f"có {abs(bed_a - bed_b)} phòng ngủ hơn"

        if nuanced and winner in {"A", "B"}:
            winner_points = primary_points or ["các điểm chi tiết nổi trội"]
            counter_points = secondary_points or ["các yếu tố còn lại"]
            reason = (
                f"Listing {winner} nhỉnh hơn nếu ưu tiên {', '.join(winner_points[:2])}; "
                f"Listing {'B' if winner == 'A' else 'A'} vẫn đáng cân nhắc nếu ưu tiên {', '.join(counter_points[:2])}"
            )

        if not reason and score_summary:
            reason = "có tổng điểm phù hợp tốt hơn sau khi cân theo nhu cầu"

        if winner:
            winner_line = f"**Nên chọn Listing {winner}** vì {reason}."
        else:
            winner_line = "**Chưa có winner tuyệt đối** với dữ liệu hiện tại."

        if hard_requirement_check and "❌" in hard_requirement_check and winner is None:
            winner_line = "**Chưa có winner tuyệt đối** vì vẫn còn hard requirement chưa rõ hoặc chưa đạt."

        suitable_a_values = self._normalize_value_list(listing_a.get("suitable_for"))
        suitable_b_values = self._normalize_value_list(listing_b.get("suitable_for"))

        bullet_a = (
            f"- Listing A phù hợp với: {', '.join(suitable_a_values)}."
            if suitable_a_values
            else "- Listing A phù hợp với nhu cầu cần cân bằng ngân sách và tiêu chuẩn ở cơ bản."
        )
        bullet_b = (
            f"- Listing B phù hợp với: {', '.join(suitable_b_values)}."
            if suitable_b_values
            else "- Listing B phù hợp với nhu cầu ưu tiên không gian hoặc tiêu chuẩn sống cao hơn."
        )

        return "\n".join([winner_line, bullet_a, bullet_b])

    def _build_conclusion_points(
        self,
        primary: Dict[str, Any],
        secondary: Dict[str, Any],
    ) -> Tuple[List[str], List[str]]:
        points_primary: List[str] = []
        points_secondary: List[str] = []

        def add_pair(condition: bool, positive: str, negative: str) -> None:
            if condition:
                if positive not in points_primary:
                    points_primary.append(positive)
                if negative not in points_secondary:
                    points_secondary.append(negative)

        price_primary = self._safe_float(primary.get("price_value_vnd"))
        price_secondary = self._safe_float(secondary.get("price_value_vnd"))
        area_primary = self._safe_float(primary.get("area_m2"))
        area_secondary = self._safe_float(secondary.get("area_m2"))
        bed_primary = self._safe_int(primary.get("bedrooms"))
        bed_secondary = self._safe_int(secondary.get("bedrooms"))
        bath_primary = self._safe_int(primary.get("bathrooms"))
        bath_secondary = self._safe_int(secondary.get("bathrooms"))

        price_per_m2_primary = price_primary / area_primary if price_primary and area_primary and area_primary > 0 else None
        price_per_m2_secondary = price_secondary / area_secondary if price_secondary and area_secondary and area_secondary > 0 else None

        if price_per_m2_primary and price_per_m2_secondary and price_per_m2_primary != price_per_m2_secondary:
            add_pair(
                price_per_m2_primary < price_per_m2_secondary,
                "giá/m² tốt hơn",
                "giá/m² thấp hơn",
            )

        if price_primary and price_secondary and price_primary != price_secondary:
            add_pair(
                price_primary < price_secondary,
                "ngân sách thấp hơn",
                "ngân sách thấp hơn",
            )

        if area_primary and area_secondary and area_primary != area_secondary:
            add_pair(
                area_primary > area_secondary,
                "diện tích rộng hơn",
                "diện tích rộng hơn",
            )

        if bed_primary and bed_secondary and bed_primary != bed_secondary:
            add_pair(
                bed_primary > bed_secondary,
                "nhiều phòng ngủ hơn",
                "nhiều phòng ngủ hơn",
            )

        if bath_primary and bath_secondary and bath_primary != bath_secondary:
            add_pair(
                bath_primary > bath_secondary,
                "nhiều WC hơn",
                "nhiều WC hơn",
            )

        loc_primary = self._normalize_value_list(primary.get("location_quality"))
        loc_secondary = self._normalize_value_list(secondary.get("location_quality"))
        hood_primary = self._normalize_value_list(primary.get("neighborhood_quality"))
        hood_secondary = self._normalize_value_list(secondary.get("neighborhood_quality"))

        if loc_primary or loc_secondary:
            primary_loc_text = normalize_query_pipeline(" ".join(loc_primary))
            secondary_loc_text = normalize_query_pipeline(" ".join(loc_secondary))
            if any(token in primary_loc_text for token in ["trung tam", "cao cap", "thuan tien", "view", "de song", "tot"]):
                points_primary.append("vị trí/kết nối tốt")
            if any(token in secondary_loc_text for token in ["trung tam", "cao cap", "thuan tien", "view", "de song", "tot"]):
                points_secondary.append("vị trí/kết nối tốt")

        if hood_primary or hood_secondary:
            primary_hood_text = normalize_query_pipeline(" ".join(hood_primary))
            secondary_hood_text = normalize_query_pipeline(" ".join(hood_secondary))
            if any(token in primary_hood_text for token in ["yen tinh", "an ninh", "an toan", "song dong", "than thien"]):
                points_primary.append("khu dân cư phù hợp")
            if any(token in secondary_hood_text for token in ["yen tinh", "an ninh", "an toan", "song dong", "than thien"]):
                points_secondary.append("khu dân cư phù hợp")

        points_primary = list(dict.fromkeys(points_primary))
        points_secondary = list(dict.fromkeys(points_secondary))

        if not points_primary:
            points_primary.append("các tiêu chí cấu trúc nổi trội")
        if not points_secondary:
            points_secondary.append("các tiêu chí còn lại")

        return points_primary, points_secondary

    def _build_hard_requirement_check(
        self,
        listing_a: Dict[str, Any],
        listing_b: Dict[str, Any],
        parsed_query: Any,
    ) -> str:
        if parsed_query is None:
            return "Chưa có hard requirement rõ ràng từ user."

        hard_filters = getattr(parsed_query, "hard_filters", None)
        if hard_filters is None:
            return "Chưa có hard requirement rõ ràng từ user."

        rows: List[List[str]] = []

        def mark(ok: bool, value: Any) -> str:
            prefix = "✅" if ok else "❌"
            return f"{prefix} {self._to_markdown_table_cell(value)}"

        price_a = self._safe_float(listing_a.get("price_value_vnd"))
        price_b = self._safe_float(listing_b.get("price_value_vnd"))
        area_a = self._safe_float(listing_a.get("area_m2"))
        area_b = self._safe_float(listing_b.get("area_m2"))
        bed_a = self._safe_int(listing_a.get("bedrooms"))
        bed_b = self._safe_int(listing_b.get("bedrooms"))

        if getattr(hard_filters, "max_price_vnd", None) is not None:
            budget = float(hard_filters.max_price_vnd)
            rows.append([
                f"Ngân sách <= {self._format_price(budget)}",
                mark(price_a is not None and price_a <= budget, self._format_price(price_a) if price_a is not None else "-"),
                mark(price_b is not None and price_b <= budget, self._format_price(price_b) if price_b is not None else "-"),
            ])

        if getattr(hard_filters, "district", None):
            district_req = str(hard_filters.district)
            rows.append([
                f"Khu vực = {district_req}",
                mark(self._value_matches_requirement(listing_a.get("district") or listing_a.get("project"), district_req), listing_a.get("district") or listing_a.get("project") or "-"),
                mark(self._value_matches_requirement(listing_b.get("district") or listing_b.get("project"), district_req), listing_b.get("district") or listing_b.get("project") or "-"),
            ])

        if getattr(hard_filters, "city", None):
            city_req = str(hard_filters.city)
            rows.append([
                f"Thành phố = {city_req}",
                mark(self._value_matches_requirement(listing_a.get("city"), city_req), listing_a.get("city") or "-"),
                mark(self._value_matches_requirement(listing_b.get("city"), city_req), listing_b.get("city") or "-"),
            ])

        if getattr(hard_filters, "property_type", None):
            type_req = str(hard_filters.property_type)
            rows.append([
                f"Loại BĐS = {type_req}",
                mark(self._value_matches_requirement(listing_a.get("property_type"), type_req), listing_a.get("property_type") or "-"),
                mark(self._value_matches_requirement(listing_b.get("property_type"), type_req), listing_b.get("property_type") or "-"),
            ])

        if getattr(hard_filters, "min_area_m2", None) is not None:
            min_area = float(hard_filters.min_area_m2)
            rows.append([
                f"Diện tích >= {min_area:.0f}m²",
                mark(area_a is not None and area_a >= min_area, f"{area_a:.0f}m²" if area_a is not None else "-"),
                mark(area_b is not None and area_b >= min_area, f"{area_b:.0f}m²" if area_b is not None else "-"),
            ])

        if getattr(hard_filters, "min_bedrooms", None) is not None:
            min_bedrooms = int(hard_filters.min_bedrooms)
            rows.append([
                f"Phòng ngủ >= {min_bedrooms}",
                mark(bed_a is not None and bed_a >= min_bedrooms, bed_a if bed_a is not None else "-"),
                mark(bed_b is not None and bed_b >= min_bedrooms, bed_b if bed_b is not None else "-"),
            ])

        if not rows:
            return ("Không phát hiện yêu cầu bắt buộc từ người dùng.\n\n"
                    "Agent đã kiểm tra các tiêu chí sau:\n"
                    "- Ngân sách tối đa (max_price_vnd)\n"
                    "- Diện tích tối thiểu (min_area_m2)\n"
                    "- Số phòng ngủ tối thiểu (min_bedrooms)\n"
                    "- Khu vực cụ thể (district)\n"
                    "- Thành phố (city)\n"
                    "- Loại bất động sản (property_type)\n\n"
                    "Hiện tại chưa tìm thấy yêu cầu nào cụ thể, nên sẽ so sánh dựa trên điểm phù hợp tổng thể.")

        table = "| Requirement | Listing A | Listing B |\n|---|---|---|\n"
        for row in rows:
            table += f"| {self._to_markdown_table_cell(row[0])} | {self._to_markdown_table_cell(row[1])} | {self._to_markdown_table_cell(row[2])} |\n"
        return table

    def _build_scoring_summary(
        self,
        listing_a: Dict[str, Any],
        listing_b: Dict[str, Any],
        parsed_query: Any,
        recommendation: Dict[str, Any],
        soft_signals: List[Dict[str, Any]] | None = None,
    ) -> str:
        score_a = self._score_listing(listing_a, listing_b, parsed_query, recommendation, "A", soft_signals)
        score_b = self._score_listing(listing_b, listing_a, parsed_query, recommendation, "B", soft_signals)
        total_winner = "A" if score_a["total"] > score_b["total"] else "B" if score_b["total"] > score_a["total"] else "="

        table = "| Nhóm | A | B | Thắng |\n|---|---:|---:|---|\n"
        for label in ("budget", "space", "location", "legal", "lifestyle", "soft_focus"):
            pretty = {
                "budget": "Tài chính",
                "space": "Không gian",
                "location": "Khu vực",
                "legal": "Pháp lý",
                "lifestyle": "Khả năng dùng",
                "soft_focus": "Soft focus",
            }[label]
            a_value = score_a[label]
            b_value = score_b[label]
            # Handle N/A for soft_focus when no soft signals
            if label == "soft_focus" and (a_value is None or b_value is None):
                a_str = "N/A" if a_value is None else f"{a_value:.1f}/10"
                b_str = "N/A" if b_value is None else f"{b_value:.1f}/10"
                winner = "="
                table += f"| {pretty} | {a_str} | {b_str} | {winner} |\n"
            else:
                winner = "A" if a_value > b_value else "B" if b_value > a_value else "="
                table += f"| {pretty} | {a_value:.1f}/10 | {b_value:.1f}/10 | {winner} |\n"

        table += f"| **Tổng** | **{score_a['total']:.2f}/10** | **{score_b['total']:.2f}/10** | **{total_winner}** |\n"
        if recommendation.get("user_profile_used"):
            table += "\n- Bảng điểm đã cân theo hồ sơ người dùng, không chỉ theo giá."
        return table

    def _build_decision_trace(
        self,
        listing_a: Dict[str, Any],
        listing_b: Dict[str, Any],
        parsed_query: Any,
        recommendation: Dict[str, Any],
        soft_signals: List[Dict[str, Any]] | None = None,
    ) -> str:
        score_a = self._score_listing(listing_a, listing_b, parsed_query, recommendation, "A", soft_signals)
        score_b = self._score_listing(listing_b, listing_a, parsed_query, recommendation, "B", soft_signals)

        lines = [
            f"- Tài chính: A {score_a['budget']:.1f} vs B {score_b['budget']:.1f}",
            f"- Không gian: A {score_a['space']:.1f} vs B {score_b['space']:.1f}",
            f"- Khu vực: A {score_a['location']:.1f} vs B {score_b['location']:.1f}",
            f"- Pháp lý: A {score_a['legal']:.1f} vs B {score_b['legal']:.1f}",
            f"- Khả năng dùng: A {score_a['lifestyle']:.1f} vs B {score_b['lifestyle']:.1f}",
        ]
        # Only show soft_focus if available
        if score_a['soft_focus'] is not None and score_b['soft_focus'] is not None:
            lines.append(f"- Soft focus: A {score_a['soft_focus']:.1f} vs B {score_b['soft_focus']:.1f}")
        else:
            lines.append("- Soft focus: N/A — chưa có ưu tiên mềm từ người dùng")
        
        lines.append(f"- Tổng điểm: A {score_a['total']:.2f}, B {score_b['total']:.2f}")
        
        if score_a.get("hard_fail") or score_b.get("hard_fail"):
            lines.append("- Listing nào không đạt hard requirement sẽ bị hạ ưu tiên trước khi xét điểm mềm.")
        return "\n".join(lines)

    def _build_confidence(
        self,
        listing_a: Dict[str, Any],
        listing_b: Dict[str, Any],
        parsed_query: Any,
        recommendation: Dict[str, Any],
    ) -> str:
        return self._build_confidence_details(listing_a, listing_b, parsed_query, recommendation)["text"]

    def _build_persona_recommendation(
        self,
        listing_a: Dict[str, Any],
        listing_b: Dict[str, Any],
        parsed_query: Any,
        user_profile: Dict[str, Any] | None,
        soft_signals: List[Dict[str, Any]] | None = None,
    ) -> str:
        # Generate specific personas for each listing based on characteristics
        persona_a = self._generate_listing_persona(listing_a, "A")
        persona_b = self._generate_listing_persona(listing_b, "B")
        
        # Build context-aware overall audience description
        context_audience: List[str] = []
        if parsed_query is not None:
            profile = getattr(parsed_query, "user_profile", None)
            soft = getattr(parsed_query, "soft_preferences", None)
            if profile and getattr(profile, "family_size", None):
                context_audience.append(f"gia đình {profile.family_size} người")
            elif profile and (getattr(profile, "has_children", False) or getattr(profile, "has_elderly", False)):
                context_audience.append("gia đình nhiều thế hệ")
            if soft and getattr(soft, "near_metro", False):
                context_audience.append("người đi làm cần kết nối di chuyển")
            if soft and getattr(soft, "family_friendly", False):
                context_audience.append("gia đình ở thực")

        if soft_signals:
            context_audience.append(self._soft_signals_to_phrase(soft_signals))

        if user_profile and user_profile.get("budget_vnd"):
            context_audience.append("người mua có ngân sách xác định")

        context_line = f"Bối cảnh người tìm kiếm: {', '.join(context_audience)}." if context_audience else ""

        return "\n".join(filter(None, [
            context_line,
            f"- Listing A: {persona_a}",
            f"- Listing B: {persona_b}",
        ]))

    def _generate_listing_persona(self, listing: Dict[str, Any], side: str) -> str:
        """
        Generate a specific persona description for a listing based on its characteristics.
        Analyzes price/m², area, bedrooms, location, and amenities to suggest ideal buyer profile.
        """
        price = self._safe_float(listing.get("price_value_vnd"))
        area = self._safe_float(listing.get("area_m2"))
        bedrooms = self._safe_int(listing.get("bedrooms"))
        bathrooms = self._safe_int(listing.get("bathrooms"))
        project = str(listing.get("project") or "").strip()
        district = str(listing.get("district") or "").strip()
        
        # Calculate price per m²
        ppm = price / area if price and area and area > 0 else None
        
        personas: List[str] = []
        reasons: List[str] = []
        
        # 1. Analyze investment potential vs. residential use
        if ppm and ppm < 80_000_000:  # ~80M/m² is threshold (low price/m²)
            personas.append("Nhà đầu tư")
            reasons.append(f"giá/m² thấp ({ppm/1_000_000:.1f}M/m²) → khai thác dòng tiền")
        elif area and area >= 100:
            personas.append("Nhà đầu tư hoặc gia đình lớn")
            reasons.append(f"diện tích lớn ({area:.0f}m²) → đầu tư hoặc gia đình rộng")
        
        # 2. Analyze family fit
        if bedrooms and bedrooms >= 3:
            if "gia đình" not in " ".join(personas):
                personas.append("Gia đình")
            reasons.append(f"{bedrooms} phòng ngủ + {bathrooms or 1} toilet → gia đình")
        elif bedrooms == 2:
            personas.append("Gia đình nhỏ hoặc cặp đôi")
            reasons.append("2 phòng ngủ → gia đình nhỏ hoặc cặp đôi")
        elif bedrooms == 1:
            personas.append("Single, sinh viên hoặc người bận")
            reasons.append("1 phòng ngủ → căn hộ linh hoạt")
        
        # 3. Analyze location convenience
        amenity_text = " ".join(self._normalize_value_list(listing.get("amenities_area"))).lower()
        transport_text = " ".join(self._normalize_value_list(listing.get("nearby_transport"))).lower()
        
        if "metro" in transport_text or "mrt" in transport_text:
            personas.append("Người đi làm")
            reasons.append("gần metro → kết nối di chuyển hàng ngày")
        
        if "trường" in amenity_text or "school" in amenity_text:
            if "gia đình" not in " ".join(personas):
                personas.append("Gia đình có con")
            reasons.append("gần trường học → tiện cho gia đình có con")
        
        # 4. Analyze suitability tags if available
        suitable_tags = self._normalize_value_list(listing.get("suitable_for"))
        if suitable_tags:
            reasons.append(f"phù hợp với: {', '.join(suitable_tags[:2])}")
        
        # 5. Finalize persona description
        if not personas:
            # Fallback based on basic characteristics
            if price and area:
                personas.append("Người tìm cân bằng giá-không gian")
        
        # Build final description
        persona_str = " / ".join(personas)
        reason_str = " • ".join(reasons) if reasons else "cân bằng chi phí-không gian"
        
        return f"{persona_str} ({reason_str})"

    def _score_listing(
        self,
        listing: Dict[str, Any],
        other_listing: Dict[str, Any],
        parsed_query: Any,
        recommendation: Dict[str, Any],
        side: str,
        soft_signals: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        price = self._safe_float(listing.get("price_value_vnd"))
        other_price = self._safe_float(other_listing.get("price_value_vnd"))
        area = self._safe_float(listing.get("area_m2"))
        other_area = self._safe_float(other_listing.get("area_m2"))
        bedrooms = self._safe_int(listing.get("bedrooms"))
        other_bedrooms = self._safe_int(other_listing.get("bedrooms"))

        hard_filters = getattr(parsed_query, "hard_filters", None) if parsed_query is not None else None
        soft_preferences = getattr(parsed_query, "soft_preferences", None) if parsed_query is not None else None
        user_profile = getattr(parsed_query, "user_profile", None) if parsed_query is not None else None

        hard_fail = False
        budget_score = 5.0
        if hard_filters and getattr(hard_filters, "max_price_vnd", None) is not None and price is not None:
            budget = float(hard_filters.max_price_vnd)
            budget_score = 10.0 if price <= budget else max(0.0, 10.0 - ((price / budget) - 1.0) * 10.0)
            if price > budget:
                hard_fail = True
        elif price is not None and other_price is not None and price != other_price:
            cheaper = min(price, other_price)
            budget_score = 10.0 if price == cheaper else max(0.0, 10.0 - (price - cheaper) / max(cheaper, 1.0) * 5.0)

        if hard_filters and getattr(hard_filters, "min_price_vnd", None) is not None and price is not None:
            if price < float(hard_filters.min_price_vnd):
                hard_fail = True

        area_score = 5.0
        if area is not None and other_area is not None and max(area, other_area) > 0:
            area_score = 10.0 * (area / max(area, other_area))

        bedroom_score = 5.0
        if bedrooms is not None and other_bedrooms is not None and max(bedrooms, other_bedrooms) > 0:
            bedroom_score = 10.0 * (bedrooms / max(bedrooms, other_bedrooms))

        space_score = round(area_score * 0.7 + bedroom_score * 0.3, 2)

        listing_district = normalize_query_pipeline(str(listing.get("district") or ""))
        listing_city = normalize_query_pipeline(str(listing.get("city") or ""))
        district_req = normalize_query_pipeline(str(getattr(hard_filters, "district", "") or "")) if hard_filters else ""
        city_req = normalize_query_pipeline(str(getattr(hard_filters, "city", "") or "")) if hard_filters else ""

        location_score = 5.0
        if district_req:
            location_score = 10.0 if district_req in listing_district or district_req in normalize_query_pipeline(str(listing.get("project") or "")) else 0.0
            if location_score == 0.0:
                hard_fail = True
        elif city_req:
            location_score = 10.0 if city_req in listing_city else 0.0
            if location_score == 0.0:
                hard_fail = True
        else:
            transport_text = " ".join(str(listing.get(key) or "") for key in ("nearby_transport", "nearby_landmarks", "amenities_area", "location_quality")).lower()
            if soft_preferences and getattr(soft_preferences, "near_metro", False) and "metro" in transport_text:
                location_score = 8.5
            elif transport_text:
                location_score = 6.0

        legal_score = min(10.0, self._legal_rank(str(listing.get("legal_status") or "")) * 2.5)

        lifestyle_score = 5.0
        suitability_text = " ".join(self._normalize_value_list(listing.get("suitable_for"))).lower()
        amenity_parts: List[str] = []
        for key in ("amenities_building", "amenities_area", "nearby_transport", "nearby_landmarks"):
            amenity_parts.extend(self._normalize_value_list(listing.get(key)))
        amenity_text = " ".join(amenity_parts).lower()
        if user_profile and getattr(user_profile, "family_size", None):
            if "gia dinh" in suitability_text or "family" in suitability_text:
                lifestyle_score = 10.0
            elif bedrooms is not None and bedrooms >= 3:
                lifestyle_score = 8.0
            else:
                lifestyle_score = 6.0
        elif soft_preferences and getattr(soft_preferences, "family_friendly", False):
            lifestyle_score = 10.0 if "gia dinh" in suitability_text else 7.0
        elif soft_preferences and getattr(soft_preferences, "near_metro", False):
            lifestyle_score = 10.0 if "metro" in amenity_text else 6.0
        elif suitability_text:
            lifestyle_score = 7.0

        soft_location_score = self._soft_preference_fit_score(listing, soft_signals)
        soft_focus_score = soft_location_score if soft_signals else None
        if soft_signals:
            location_score = max(location_score, soft_location_score)
            lifestyle_score = max(lifestyle_score, min(10.0, soft_location_score + 1.0))

        score_weights = {
            "budget": 0.35,
            "space": 0.25,
            "location": 0.2,
            "legal": 0.1,
            "lifestyle": 0.1,
            "soft_focus": 0.0,
        }
        if user_profile and getattr(user_profile, "family_size", None):
            score_weights.update({"space": 0.32, "lifestyle": 0.16, "budget": 0.22})
        if hard_filters and getattr(hard_filters, "max_price_vnd", None) is not None:
            score_weights["budget"] = 0.4
        if hard_filters and (getattr(hard_filters, "district", None) or getattr(hard_filters, "city", None)):
            score_weights["location"] = 0.28
        if soft_signals:
            score_weights.update({"budget": 0.22, "space": 0.16, "location": 0.2, "lifestyle": 0.12, "soft_focus": 0.3})

        total_weight = sum(score_weights.values()) or 1.0
        for key in score_weights:
            score_weights[key] = score_weights[key] / total_weight

        total = (
            budget_score * score_weights["budget"]
            + space_score * score_weights["space"]
            + location_score * score_weights["location"]
            + legal_score * score_weights["legal"]
            + lifestyle_score * score_weights["lifestyle"]
            + (soft_focus_score * score_weights["soft_focus"] if soft_focus_score is not None else 0.0)
        )

        if hard_fail:
            total = min(total, 4.0)

        winner = side
        if recommendation.get("winner") in {"A", "B"}:
            winner = str(recommendation.get("winner"))

        return {
            "budget": round(budget_score, 2),
            "space": round(space_score, 2),
            "location": round(location_score, 2),
            "legal": round(legal_score, 2),
            "lifestyle": round(lifestyle_score, 2),
            "soft_focus": round(soft_focus_score, 2) if soft_focus_score is not None else None,
            "total": round(total, 2),
            "hard_fail": hard_fail,
            "winner": winner,
        }

    def _extract_soft_preference_signals(self, query_text: str) -> List[Dict[str, Any]]:
        text = normalize_query_pipeline(str(query_text or ""))
        if not text:
            return []

        signal_catalog: List[tuple[str, List[str], float]] = [
            ("gần nơi vui chơi giải trí", ["vui choi", "giai tri", "entertainment", "mall", "shopping mall", "trung tam thuong mai", "phố đi bộ", "pho di bo", "cafe", "nhà hàng", "nha hang", "rạp chiếu phim", "rap chieu phim", "khu vui chơi", "cong vien", "công viên", "bar", "nightlife"], 1.0),
            ("gần metro / giao thông", ["metro", "mrt", "lrt", "nha ga", "ga tau", "tram xe", "tau dien", "station"], 0.95),
            ("gần trường học", ["truong", "school", "mam non", "hoc vien", "tieu hoc", "trung hoc", "dai hoc"], 0.9),
            ("khu yên tĩnh", ["yen tinh", "it on", "an tinh", "quiet"], 0.8),
            ("nhiều tiện ích", ["nhieu tien ich", "day du tien ich", "amenities", "tien ich"], 0.85),
            ("gia đình ở thực", ["gia dinh", "family", "tre em", "con nho", "o lau dai", "ở lâu dài"], 0.75),
            ("thể thao / gym", ["gym", "fitness", "phong tap", "the thao", "sports"], 0.8),
            ("hồ bơi", ["ho boi", "be boi", "pool", "swimming"], 0.75),
        ]

        signals: List[Dict[str, Any]] = []
        for label, keywords, weight in signal_catalog:
            if any(keyword in text for keyword in keywords):
                signals.append({"label": label, "keywords": keywords, "weight": weight})
        return signals

    def _soft_preference_fit_score(
        self,
        listing: Dict[str, Any],
        soft_signals: List[Dict[str, Any]] | None,
    ) -> float:
        if not soft_signals:
            return 5.0

        listing_text = self._listing_soft_text(listing)
        score = 5.0
        matched_weight = 0.0
        total_weight = 0.0

        for signal in soft_signals:
            weight = float(signal.get("weight") or 1.0)
            total_weight += weight
            if any(keyword in listing_text for keyword in signal.get("keywords") or []):
                matched_weight += weight

        if total_weight > 0:
            ratio = matched_weight / total_weight
            score = 2.5 + (ratio * 7.5)

        return max(0.0, min(10.0, round(score, 2)))

    def _listing_soft_text(self, listing: Dict[str, Any]) -> str:
        parts: List[str] = []
        for key in (
            "title",
            "project",
            "district",
            "ward",
            "street",
            "location_quality",
            "neighborhood_quality",
            "view",
            "suitable_for",
            "amenities_building",
            "amenities_area",
            "nearby_landmarks",
            "nearby_transport",
            "nearby_roads",
            "access",
        ):
            parts.extend(self._normalize_value_list(listing.get(key)))
        return normalize_query_pipeline(" ".join(parts))

    @staticmethod
    def _soft_signals_to_phrase(soft_signals: List[Dict[str, Any]] | None) -> str:
        if not soft_signals:
            return "nhu cầu mềm của user"
        labels = [str(signal.get("label") or "").strip() for signal in soft_signals if str(signal.get("label") or "").strip()]
        if not labels:
            return "nhu cầu mềm của user"
        return ", ".join(dict.fromkeys(labels[:3]))

    @staticmethod
    def _value_matches_requirement(value: Any, requirement: Any) -> bool:
        value_text = normalize_query_pipeline(str(value or "")).strip()
        requirement_text = normalize_query_pipeline(str(requirement or "")).strip()
        if not value_text or not requirement_text:
            return False
        return requirement_text in value_text or value_text in requirement_text

    def _build_quick_compare(
        self,
        listing_a: Dict[str, Any],
        listing_b: Dict[str, Any],
        price_a: float | None,
        price_b: float | None,
        ppm_a: float | None,
        ppm_b: float | None,
        area_a: float | None,
        area_b: float | None,
        bed_a: int | None,
        bed_b: int | None,
        bath_a: int | None,
        bath_b: int | None,
    ) -> str:
        """
        Build quick comparison: 3-5 bullet points of key differences.
        Each bullet focuses on ONE important difference.
        """
        bullets = []

        # Check price difference
        if price_a and price_b:
            if abs(price_a - price_b) / min(price_a, price_b) * 100 > 10:
                cheaper = "A" if price_a < price_b else "B"
                price_diff = abs(price_a - price_b)
                bullets.append(f"Listing {cheaper} rẻ hơn {self._format_price(price_diff)}")

        # Check price per m2
        if ppm_a and ppm_b:
            if abs(ppm_a - ppm_b) / min(ppm_a, ppm_b) * 100 > 10:
                cheaper_ppm = "A" if ppm_a < ppm_b else "B"
                bullets.append(f"Listing {cheaper_ppm} tiết kiệm giá/m²")

        # Check area
        if area_a and area_b:
            if abs(area_a - area_b) / min(area_a, area_b) * 100 > 20:
                larger = "A" if area_a > area_b else "B"
                area_diff = abs(area_a - area_b)
                bullets.append(f"Listing {larger} rộng hơn ~{area_diff:.0f}m²")

        # Check bedrooms
        if bed_a and bed_b and bed_a != bed_b:
            more_beds = "A" if bed_a > bed_b else "B"
            bullets.append(f"Listing {more_beds} có {abs(bed_a - bed_b)} phòng ngủ hơn")

        # Check bathrooms
        if bath_a and bath_b and bath_a != bath_b:
            more_baths = "A" if bath_a > bath_b else "B"
            bullets.append(f"Listing {more_baths} có {abs(bath_a - bath_b)} WC hơn")

        # Check legal status
        legal_a = str(listing_a.get("legal_status") or "").strip()
        legal_b = str(listing_b.get("legal_status") or "").strip()
        if legal_a and legal_b and legal_a != legal_b:
            if "sở hữu" in legal_a.lower() or "da cap" in legal_a.lower():
                bullets.append("Listing A có pháp lý rõ ràng hơn")
            elif "sở hữu" in legal_b.lower() or "da cap" in legal_b.lower():
                bullets.append("Listing B có pháp lý rõ ràng hơn")

        # Check enrichment highlights (suitable_for)
        suitable_a = str(listing_a.get("suitable_for") or "").strip()
        suitable_b = str(listing_b.get("suitable_for") or "").strip()
        if suitable_a and not suitable_b:
            bullets.append(f"Listing A phù hợp với: {suitable_a}")
        elif suitable_b and not suitable_a:
            bullets.append(f"Listing B phù hợp với: {suitable_b}")

        # Limit to 5 bullets
        bullets = bullets[:5]

        # Format as bullet list
        if bullets:
            return "\n".join([f"- {b}" for b in bullets])
        return "- Cả hai listing tương đương nhau"

    def _build_comparison_table(
        self,
        listing_a: Dict[str, Any],
        listing_b: Dict[str, Any],
        price_a: float | None,
        price_b: float | None,
        ppm_a: float | None,
        ppm_b: float | None,
        area_a: float | None,
        area_b: float | None,
        bed_a: int | None,
        bed_b: int | None,
        bath_a: int | None,
        bath_b: int | None,
        floors_a: int | None,
        floors_b: int | None,
        legal_a: str,
        legal_b: str,
    ) -> str:
        """
        Build comparison table with max 8-10 rows.
        Order: price, price/m², area, bedrooms, bathrooms, legal, location, enrichment.
        Use ⭐ for clear advantages.
        """
        rows = []

        # Row 1: Giá tổng
        if price_a and price_b:
            price_a_str = self._format_price(price_a)
            price_b_str = self._format_price(price_b)

            # Keep total price neutral in table to avoid over-weighting absolute budget.
            rows.append(["Giá tổng", price_a_str, price_b_str])

        # Row 2: Giá/m²
        if ppm_a and ppm_b:
            cheaper_ppm = "A" if ppm_a < ppm_b else "B"
            ppm_a_str = f"~{ppm_a/1e6:.1f}Tr/m²"
            ppm_b_str = f"~{ppm_b/1e6:.1f}Tr/m²"

            star_a = " ⭐" if ppm_a < ppm_b else ""
            star_b = " ⭐" if ppm_b < ppm_a else ""
            
            rows.append(["Giá/m²", f"{ppm_a_str}{star_a}", f"{ppm_b_str}{star_b}"])

        # Row 3: Diện tích
        if area_a and area_b:
            larger = "A" if area_a > area_b else "B"
            area_a_str = f"{area_a:.0f}m²"
            area_b_str = f"{area_b:.0f}m²"

            star_a = " ⭐" if area_a > area_b else ""
            star_b = " ⭐" if area_b > area_a else ""
            
            rows.append(["Diện tích", f"{area_a_str}{star_a}", f"{area_b_str}{star_b}"])

        # Row 4: Phòng ngủ
        if bed_a is not None or bed_b is not None:
            bed_a_str = str(bed_a) if bed_a is not None else "-"
            bed_b_str = str(bed_b) if bed_b is not None else "-"

            star_a = " ⭐" if bed_a and bed_b and bed_a > bed_b else ""
            star_b = " ⭐" if bed_a and bed_b and bed_b > bed_a else ""
            
            rows.append(["Phòng ngủ", f"{bed_a_str}{star_a}", f"{bed_b_str}{star_b}"])

        # Row 5: Phòng tắm
        if bath_a is not None or bath_b is not None:
            bath_a_str = str(bath_a) if bath_a is not None else "-"
            bath_b_str = str(bath_b) if bath_b is not None else "-"

            star_a = " ⭐" if bath_a and bath_b and bath_a > bath_b else ""
            star_b = " ⭐" if bath_a and bath_b and bath_b > bath_a else ""
            
            rows.append(["WC", f"{bath_a_str}{star_a}", f"{bath_b_str}{star_b}"])

        # Row 6: Số tầng
        if floors_a is not None or floors_b is not None:
            floors_a_str = str(floors_a) if floors_a is not None else "-"
            floors_b_str = str(floors_b) if floors_b is not None else "-"

            star_a = " ⭐" if floors_a and floors_b and floors_a > floors_b else ""
            star_b = " ⭐" if floors_a and floors_b and floors_b > floors_a else ""

            rows.append(["Số tầng", f"{floors_a_str}{star_a}", f"{floors_b_str}{star_b}"])

        # Row 7: Pháp lý
        if legal_a and legal_b:
            legal_a_short = self._shorten_legal_status(legal_a)
            legal_b_short = self._shorten_legal_status(legal_b)

            rows.append(["Pháp lý", legal_a_short, legal_b_short])

        # Row 8: Khu vực / Dự án
        district_a = str(listing_a.get("district") or "").strip()
        district_b = str(listing_b.get("district") or "").strip()
        project_a = str(listing_a.get("project") or "").strip()
        project_b = str(listing_b.get("project") or "").strip()
        
        label_a = project_a if project_a else district_a
        label_b = project_b if project_b else district_b
        
        if label_a or label_b:
            rows.append(["Khu vực", label_a or "-", label_b or "-"])

        # Enrichment rows: one category per row, only when at least one side has data.
        enrichment_rows = self._extract_enrichment_rows(listing_a, listing_b)
        rows.extend(enrichment_rows)

        # Format as markdown table
        if not rows:
            return "| Tiêu chí | Listing A | Listing B |\n|---|---|---|\n"

        table = "| Tiêu chí | Listing A | Listing B |\n"
        table += "|---|---|---|\n"
        for row in rows:
            metric = self._to_markdown_table_cell(row[0])
            value_a = self._to_markdown_table_cell(row[1])
            value_b = self._to_markdown_table_cell(row[2])
            table += f"| {metric} | {value_a} | {value_b} |\n"

        return table

    def _build_detailed_analysis(
        self,
        listing_a: Dict[str, Any],
        listing_b: Dict[str, Any],
        price_a: float | None,
        price_b: float | None,
        ppm_a: float | None,
        ppm_b: float | None,
        area_a: float | None,
        area_b: float | None,
        bed_a: int | None,
        bed_b: int | None,
    ) -> str:
        """Build detailed analysis section with subsections for financial, space, usability."""
        sections = []

        # 💰 Financial analysis
        finance = self._analyze_financial(price_a, price_b, ppm_a, ppm_b)
        if finance != "-":
            sections.append(f"### 💰 Tài chính\n{finance}")

        # 📐 Space analysis
        space = self._analyze_space(area_a, area_b, bed_a, bed_b)
        if space != "-":
            sections.append(f"### 📐 Không gian\n{space}")

        # 🏠 Usability analysis
        usability = self._analyze_usability(listing_a, listing_b)
        if usability != "-":
            sections.append(f"### 🏠 Khả năng sử dụng\n{usability}")

        # 📍 Location & connectivity analysis
        location_connectivity = self._analyze_location_connectivity(listing_a, listing_b)
        if location_connectivity != "-":
            sections.append(f"### 📍 Vị trí & Kết nối\n{location_connectivity}")

        # 🏠 Living environment analysis
        living_environment = self._analyze_living_environment(listing_a, listing_b)
        if living_environment != "-":
            sections.append(f"### 🏠 Môi trường sống & Khả năng sử dụng\n{living_environment}")

        return "\n\n".join(sections) if sections else "-"

    def _analyze_financial(
        self,
        price_a: float | None,
        price_b: float | None,
        ppm_a: float | None,
        ppm_b: float | None,
    ) -> str:
        """Analyze and compare financial aspects."""
        lines: List[str] = []

        if price_a and price_b and price_a != price_b:
            cheaper = "A" if price_a < price_b else "B"
            expensive = "B" if cheaper == "A" else "A"
            price_gap = abs(price_a - price_b)
            lines.append(
                f"Giá tổng: Listing {cheaper} thấp hơn Listing {expensive} khoảng {self._format_price(price_gap)}."
            )

        if ppm_a and ppm_b and ppm_a != ppm_b:
            cheaper_ppm = "A" if ppm_a < ppm_b else "B"
            ppm_diff = abs(ppm_a - ppm_b)
            lines.append(
                f"Giá/m²: Listing {cheaper_ppm} tốt hơn khoảng {self._format_price_per_m2(ppm_diff)}; đây là lựa chọn hợp hơn nếu ưu tiên hiệu suất vốn."
            )

        if not lines:
            lines.append("Chưa đủ dữ liệu tài chính để kết luận rõ ràng về giá tổng hoặc giá/m².")

        return "\n".join(lines)

    def _analyze_space(
        self,
        area_a: float | None,
        area_b: float | None,
        bed_a: int | None,
        bed_b: int | None,
    ) -> str:
        """Analyze and compare space aspects."""
        lines: List[str] = []

        if area_a and area_b and area_a != area_b:
            larger = "A" if area_a > area_b else "B"
            smaller = "B" if larger == "A" else "A"
            area_diff = abs(area_a - area_b)
            lines.append(
                f"Diện tích: Listing {larger} rộng hơn Listing {smaller} khoảng {area_diff:.0f}m², hợp hơn nếu ưu tiên không gian sinh hoạt hoặc khả năng bố trí nội thất thoáng."
            )

        if bed_a and bed_b and bed_a != bed_b:
            more_beds = "A" if bed_a > bed_b else "B"
            fewer_beds = "B" if more_beds == "A" else "A"
            lines.append(
                f"Phòng ngủ: Listing {more_beds} có nhiều phòng hơn Listing {fewer_beds}, phù hợp hơn cho gia đình đông người hoặc nhu cầu tách phòng."
            )

        if not lines:
            lines.append("Chưa đủ dữ liệu cấu trúc để kết luận rõ ràng về lợi thế không gian.")

        return "\n".join(lines)

    def _analyze_usability(
        self,
        listing_a: Dict[str, Any],
        listing_b: Dict[str, Any],
    ) -> str:
        """Analyze and compare usability for living or investment."""
        lines = []

        def _collect_text(listing: Dict[str, Any], keys: List[str]) -> str:
            parts: List[str] = []
            for key in keys:
                parts.extend(self._normalize_value_list(listing.get(key)))
            return " ".join(parts).lower()

        def _quality_hint(text: str) -> str:
            normalized = normalize_query_pipeline(text)
            if not normalized:
                return ""
            positive_tokens = ["tot", "dep", "trung tam", "thuan tien", "song dong", "yên tĩnh", "yen tinh","yên bình","yen binh" "an ninh", "an toan", "cap", "cao cap", "view", "cong vien", "tiện ích", "an ninh", "an toàn", "đẹp", "đẹp mắt", "đẹp hơn", "đẹp nhất", "trung tâm", "thuận tiện", "sông", "sông nước", "yên tĩnh", "yên bình", "an ninh", "an toàn", "cấp", "cao cấp", "view đẹp", "công viên", "tiện ích đầy đủ"]
            negative_tokens = ["xa", "it tien ich", "thiếu", "thieu", "binh thuong", "on", "khong ro"]
            if any(token in normalized for token in positive_tokens):
                return "tích cực"
            if any(token in normalized for token in negative_tokens):
                return "hạn chế"
            return "trung tính"

        location_text_a = _collect_text(listing_a, ["location_quality", "view", "nearby_landmarks", "nearby_transport"])
        location_text_b = _collect_text(listing_b, ["location_quality", "view", "nearby_landmarks", "nearby_transport"])
        neighborhood_text_a = _collect_text(listing_a, ["neighborhood_quality", "suitable_for", "amenities_area", "amenities_building"])
        neighborhood_text_b = _collect_text(listing_b, ["neighborhood_quality", "suitable_for", "amenities_area", "amenities_building"])

        if location_text_a or location_text_b:
            hint_a = _quality_hint(location_text_a)
            hint_b = _quality_hint(location_text_b)
            if hint_a or hint_b:
                lines.append(
                    f"- Chất lượng vị trí: A {hint_a or 'không rõ'}; B {hint_b or 'không rõ'}."
                )

        if neighborhood_text_a or neighborhood_text_b:
            hint_a = _quality_hint(neighborhood_text_a)
            hint_b = _quality_hint(neighborhood_text_b)
            if hint_a or hint_b:
                lines.append(
                    f"- Chất lượng khu dân cư: A {hint_a or 'không rõ'}; B {hint_b or 'không rõ'}."
                )

        suitable_a_values = self._normalize_value_list(listing_a.get("suitable_for"))
        suitable_b_values = self._normalize_value_list(listing_b.get("suitable_for"))

        if suitable_a_values and suitable_b_values:
            lines.append(f"- Căn A phù hợp hơn với nhóm: {', '.join(suitable_a_values)}.")
            lines.append(f"- Căn B phù hợp hơn với nhóm: {', '.join(suitable_b_values)}.")
        elif suitable_a_values:
            lines.append(f"- Căn A phù hợp hơn với nhóm: {', '.join(suitable_a_values)}.")
        elif suitable_b_values:
            lines.append(f"- Căn B phù hợp hơn với nhóm: {', '.join(suitable_b_values)}.")

        if not lines:
            lines.append("Chưa đủ dữ liệu để đánh giá mức độ phù hợp sử dụng.")

        return "\n".join(lines) if lines else "-"

    def _analyze_location_connectivity(
        self,
        listing_a: Dict[str, Any],
        listing_b: Dict[str, Any],
    ) -> str:
        """Analyze location connectivity, roads, transport, POIs, and centrality."""

        def _collect_text(listing: Dict[str, Any], keys: List[str]) -> str:
            parts: List[str] = []
            for key in keys:
                parts.extend(self._normalize_value_list(listing.get(key)))
            return normalize_query_pipeline(" ".join(parts))

        def _signal(text: str) -> str:
            if not text:
                return "không rõ"
            strong_terms = ["trung tam", "trung tâm", "thuong mai", "thuận tiện", "thuan tien", "metro", "ga", "bus", "ben xe", "cau", "duong lon", "pho di bo", "cong vien", "mall", "school", "benh vien"]
            if any(term in text for term in strong_terms):
                return "tốt"
            if any(term in text for term in ["hẻm", "hem", "xa trung tam", "xa", "it ket noi", "it tien ich"]):
                return "cần cân nhắc"
            return "trung tính"

        location_a = _collect_text(listing_a, ["location_quality", "nearby_transport", "nearby_landmarks", "nearby_roads", "access", "view", "district", "project"])
        location_b = _collect_text(listing_b, ["location_quality", "nearby_transport", "nearby_landmarks", "nearby_roads", "access", "view", "district", "project"])

        lines: List[str] = []
        lines.append(
            f"- Listing A: {_signal(location_a)} về kết nối đường sá/giao thông/điểm tiện ích xung quanh."
        )
        lines.append(
            f"- Listing B: {_signal(location_b)} về kết nối đường sá/giao thông/điểm tiện ích xung quanh."
        )

        if any(token in location_a for token in ["trung tam", "trung tâm", "metro", "ga", "mall", "cong vien", "bệnh viện", "benh vien"]):
            lines.append("- Listing A có lợi thế về vị trí trung tâm hoặc tiếp cận POI/tiện ích nhanh hơn.")
        if any(token in location_b for token in ["trung tam", "trung tâm", "metro", "ga", "mall", "cong vien", "bệnh viện", "benh vien"]):
            lines.append("- Listing B có lợi thế về vị trí trung tâm hoặc tiếp cận POI/tiện ích nhanh hơn.")

        if not lines:
            return "-"
        return "\n".join(lines)

    def _analyze_living_environment(
        self,
        listing_a: Dict[str, Any],
        listing_b: Dict[str, Any],
    ) -> str:
        """Analyze living environment, safety, rental use, and business suitability."""

        def _collect_text(listing: Dict[str, Any], keys: List[str]) -> str:
            parts: List[str] = []
            for key in keys:
                parts.extend(self._normalize_value_list(listing.get(key)))
            return normalize_query_pipeline(" ".join(parts))

        def _environment_label(text: str) -> str:
            if not text:
                return "không rõ"
            if any(term in text for term in ["an ninh", "an toàn", "yen tinh", "yên tĩnh", "khu dan cu", "khu dân cư", "gia dinh", "family", "o lau dai", "ở lâu dài"]):
                return "hợp ở lâu dài"
            if any(term in text for term in ["cho thue", "cho thuê", "rent", "dòng tiền", "dau tu", "đầu tư"]):
                return "hợp cho thuê/đầu tư"
            if any(term in text for term in ["kinh doanh", "shophouse", "mat tien", "mặt tiền", "buôn bán"]):
                return "hợp kinh doanh"
            return "trung tính"

        env_a = _collect_text(listing_a, ["neighborhood_quality", "suitable_for", "amenities_area", "amenities_building", "location_quality"])
        env_b = _collect_text(listing_b, ["neighborhood_quality", "suitable_for", "amenities_area", "amenities_building", "location_quality"])

        lines: List[str] = []
        lines.append(f"- Listing A: {_environment_label(env_a)} về khu dân cư/an ninh/khả năng ở thực.")
        lines.append(f"- Listing B: {_environment_label(env_b)} về khu dân cư/an ninh/khả năng ở thực.")

        if any(token in env_a for token in ["cho thue", "cho thuê", "rent", "dòng tiền", "dau tu", "đầu tư"]):
            lines.append("- Listing A có tín hiệu phù hợp hơn cho mục tiêu cho thuê hoặc đầu tư.")
        if any(token in env_b for token in ["cho thue", "cho thuê", "rent", "dòng tiền", "dau tu", "đầu tư"]):
            lines.append("- Listing B có tín hiệu phù hợp hơn cho mục tiêu cho thuê hoặc đầu tư.")

        if any(token in env_a for token in ["kinh doanh", "shophouse", "mat tien", "mặt tiền", "buôn bán"]):
            lines.append("- Listing A có tiềm năng tốt hơn nếu bạn ưu tiên kinh doanh hoặc khai thác thương mại.")
        if any(token in env_b for token in ["kinh doanh", "shophouse", "mat tien", "mặt tiền", "buôn bán"]):
            lines.append("- Listing B có tiềm năng tốt hơn nếu bạn ưu tiên kinh doanh hoặc khai thác thương mại.")

        if not lines:
            return "-"
        return "\n".join(lines)

    def _build_tradeoffs(
        self,
        listing_a: Dict[str, Any],
        listing_b: Dict[str, Any],
    ) -> str:
        """Build trade-offs section: 1-2 drawbacks for each listing."""
        tradeoff_a = "Listing A"
        drawbacks_a: List[str] = []

        tradeoff_b = "Listing B"
        drawbacks_b: List[str] = []

        # Check price
        price_a = self._safe_float(listing_a.get("price_value_vnd"))
        price_b = self._safe_float(listing_b.get("price_value_vnd"))
        if price_a and price_b and price_a > price_b:
            drawbacks_a.append(f"giá cao hơn Listing B khoảng {self._format_price(price_a - price_b)}")
        if price_b and price_a and price_b > price_a:
            drawbacks_b.append(f"giá cao hơn Listing A khoảng {self._format_price(price_b - price_a)}")

        # Check area
        area_a = self._safe_float(listing_a.get("area_m2"))
        area_b = self._safe_float(listing_b.get("area_m2"))
        if area_a and area_b and area_a < area_b:
            drawbacks_a.append(f"diện tích nhỏ hơn Listing B khoảng {area_b - area_a:.0f}m²")
        if area_b and area_a and area_b < area_a:
            drawbacks_b.append(f"diện tích nhỏ hơn Listing A khoảng {area_a - area_b:.0f}m²")

        # Check bedrooms
        bed_a = self._safe_int(listing_a.get("bedrooms"))
        bed_b = self._safe_int(listing_b.get("bedrooms"))
        if bed_a and bed_b and bed_a < bed_b:
            drawbacks_a.append(f"ít phòng hơn Listing B ({bed_a} so với {bed_b})")
        if bed_b and bed_a and bed_b < bed_a:
            drawbacks_b.append(f"ít phòng hơn Listing A ({bed_b} so với {bed_a})")

        # Check legal status
        legal_a = str(listing_a.get("legal_status") or "").strip()
        legal_b = str(listing_b.get("legal_status") or "").strip()
        
        if not legal_a and legal_b:
            drawbacks_a.append("pháp lý không rõ ràng")
        if not legal_b and legal_a:
            drawbacks_b.append("pháp lý không rõ ràng")

        suitable_a_values = self._normalize_value_list(listing_a.get("suitable_for"))
        suitable_b_values = self._normalize_value_list(listing_b.get("suitable_for"))
        if suitable_a_values and not suitable_b_values:
            drawbacks_b.append(f"ít tín hiệu phù hợp nhu cầu hơn Listing A ({', '.join(suitable_a_values[:2])})")
        if suitable_b_values and not suitable_a_values:
            drawbacks_a.append(f"ít tín hiệu phù hợp nhu cầu hơn Listing B ({', '.join(suitable_b_values[:2])})")

        # Format section
        lines = []
        
        if drawbacks_a:
            lines.append(f"- **{tradeoff_a}**: {', '.join(drawbacks_a[:2])}.")
        else:
            lines.append(f"- **{tradeoff_a}**: chưa thấy nhược điểm lớn nổi bật.")

        if drawbacks_b:
            lines.append(f"- **{tradeoff_b}**: {', '.join(drawbacks_b[:2])}.")
        else:
            lines.append(f"- **{tradeoff_b}**: chưa thấy nhược điểm lớn nổi bật.")

        return "\n".join(lines)

    def _build_next_steps(self) -> str:
        """Build next steps / suggestions section."""
        suggestions = [
            "- Xem chi tiết cả hai listing để hiểu rõ hơn",
            "- Liên hệ chủ/môi giới để thực hiện thăm nhà",
            "- Tìm thêm listing tương tự để so sánh rộng hơn",
            "- Kiểm tra pháp lý kỹ lưỡng trước khi quyết định",
        ]
        return "\n".join(suggestions[:3])

    def _extract_key_enrichment(
        self,
        listing: Dict[str, Any],
        show_amenities: bool = False,
    ) -> str:
        """Render enrichment as structured plain text, only with non-empty categories."""
        schema_rows = [
            ("tuyến đường gần", ["nearby_roads"]),
            ("khả năng tiếp cận", ["access"]),
            ("phương tiện công cộng", ["nearby_transport"]),
            ("địa điểm nổi bật", ["nearby_landmarks"]),
            ("vị trí", ["location_quality", "view"]),
            ("khu dân cư", ["neighborhood_quality"]),
            ("tiện ích nội khu", ["amenities_building"]),
            ("tiện ích khu vực", ["amenities_area"]),
            
        ]

        highlights: List[str] = []
        for label, keys in schema_rows:
            merged_values: List[str] = []
            for key in keys:
                for value in self._normalize_value_list(listing.get(key)):
                    if value not in merged_values:
                        merged_values.append(value)

            # Skip empty categories so the table only shows meaningful enrichment.
            if merged_values:
                rendered = ", ".join(merged_values)
                highlights.append(f"- {label}: {rendered}")

        if not highlights:
            return "-"

        # Keep one bullet per line for readability in text-based channels.
        result = "\n".join(highlights)
        if len(result) > 320:
            result = result[:317].rstrip() + "..."
        return result

    def _extract_enrichment_rows(
        self,
        listing_a: Dict[str, Any],
        listing_b: Dict[str, Any],
    ) -> List[List[str]]:
        """Build enrichment rows as separate table rows per category."""
        schema_rows = [
            ("Đường", ["nearby_roads"]),
            ("Khả năng tiếp cận", ["access"]),
            ("Phương tiện công cộng", ["nearby_transport"]),
            ("Địa điểm nổi bật", ["nearby_landmarks"]),
            ("Vị trí", ["location_quality", "view"]),
            ("Khu dân cư", ["neighborhood_quality"]),
            ("Tiện ích nội khu", ["amenities_building"]),
            ("Tiện ích khu vực", ["amenities_area"]),
            ("Phù hợp", ["suitable_for"]),
        ]

        rows: List[List[str]] = []
        for label, keys in schema_rows:
            values_a: List[str] = []
            values_b: List[str] = []

            for key in keys:
                for value in self._normalize_value_list(listing_a.get(key)):
                    if value not in values_a:
                        values_a.append(value)
                for value in self._normalize_value_list(listing_b.get(key)):
                    if value not in values_b:
                        values_b.append(value)

            if not values_a and not values_b:
                continue

            rendered_a = ", ".join(values_a) if values_a else "-"
            rendered_b = ", ".join(values_b) if values_b else "-"
            rows.append([label, rendered_a, rendered_b])

        return rows

    @staticmethod
    def _normalize_value_list(value: Any) -> List[str]:
        """Normalize enrichment values from list/dict/JSON-string and drop empty tokens."""
        def _is_empty_token(text: str) -> bool:
            normalized = text.strip().lower()
            return normalized in {"", "[]", "{}", "null", "none", "n/a", "na", "khong", "không"}

        def _clean_text(raw: Any) -> str:
            text = str(raw or "").strip()
            if not text:
                return ""
            # Remove html-like tags and noisy wrappers from LLM/raw backend outputs.
            text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
            text = text.replace("```", "")
            text = re.sub(r"<[^>]+>", "", text)
            text = text.strip().strip("\"'")

            # Unwrap simple JSON-like key/value wrappers and array/object shells.
            text = re.sub(r'^\s*"?[A-Za-z0-9_\-\s]+"?\s*:\s*', "", text)
            text = re.sub(r'^[\[\{]+\s*', "", text)
            text = re.sub(r'\s*[\]\}]+$', "", text)

            text = re.sub(r"\s+", " ", text).strip()
            return text

        out: List[str] = []

        def _push(text: Any) -> None:
            cleaned = _clean_text(text)
            if _is_empty_token(cleaned):
                return
            if cleaned not in out:
                out.append(cleaned)

        if value is None:
            return out

        if isinstance(value, (list, tuple, set)):
            for item in value:
                _push(item)
            return out

        if isinstance(value, dict):
            for item in value.values():
                _push(item)
            return out

        text = str(value).strip()
        if not text:
            return out

        # Parse JSON-like strings such as "[]", "[\"view nội khu\"]", or nested dict/list.
        if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    for item in parsed.values():
                        if isinstance(item, (list, tuple, set)):
                            for sub in item:
                                _push(sub)
                        else:
                            _push(item)
                elif isinstance(parsed, (list, tuple, set)):
                    for item in parsed:
                        _push(item)
                else:
                    _push(parsed)
                return out
            except Exception:
                pass

        # Fallback split for plain text with delimiters.
        chunks = re.split(r"[\n;,|]", text)
        for chunk in chunks:
            _push(chunk)
        return out

    @staticmethod
    def _legal_rank(legal_status: str) -> int:
        """Rank legal clarity: red/pink book > ownership issued > sales contract > unknown."""
        text = str(legal_status or "").lower()
        if any(token in text for token in ["so do", "sổ đỏ", "so hong", "sổ hồng"]):
            return 4
        if any(token in text for token in ["da cap", "đã cấp", "so huu", "sở hữu"]):
            return 3
        if any(token in text for token in ["hop dong mua ban", "hợp đồng mua bán"]):
            return 2
        if text.strip():
            return 1
        return 0

    @staticmethod
    def _shorten_legal_status(legal_status: str) -> str:
        """Shorten legal status text for table display."""
        if not legal_status:
            return "-"
        
        # Common mappings
        mappings = {
            "sở hữu": "Sở hữu ✓",
            "đã cấp": "Đã cấp ✓",
            "chờ cấp": "Chờ cấp",
            "không rõ": "Không rõ",
        }
        
        lower = legal_status.lower()
        for key, val in mappings.items():
            if key in lower:
                return val
        
        # If too long, extract first 20 chars
        if len(legal_status) > 20:
            return legal_status[:20] + "..."
        
        return legal_status

    @staticmethod
    def _format_price(price_vnd: float) -> str:
        """Format price in Vietnamese currency."""
        if price_vnd >= 1e9:
            return f"{price_vnd / 1e9:.1f}Tỷ"
        elif price_vnd >= 1e6:
            return f"{price_vnd / 1e6:.0f}Tr"
        elif price_vnd >= 1e3:
            return f"{price_vnd / 1e3:.0f}K"
        else:
            return f"{price_vnd:.0f}"

    @staticmethod
    def _format_price_per_m2(price_per_m2_vnd: float) -> str:
        """Format unit-price delta into readable VND per m2 units."""
        if price_per_m2_vnd >= 1e9:
            return f"{price_per_m2_vnd / 1e9:.2f} tỷ/m²"
        if price_per_m2_vnd >= 1e6:
            return f"{price_per_m2_vnd / 1e6:.1f} triệu/m²"
        if price_per_m2_vnd >= 1e3:
            return f"{price_per_m2_vnd / 1e3:.0f} nghìn/m²"
        return f"{price_per_m2_vnd:.0f} đ/m²"


# Create a singleton instance for easy access
_formatter = ComparisonFormatter()


def format_comparison(
    listing_a: Dict[str, Any],
    listing_b: Dict[str, Any],
    user_query: str | None = None,
    recommendation: Dict[str, Any] | None = None,
    user_profile: Dict[str, Any] | None = None,
    debug_mode: bool = False,
) -> str:
    """
    Convenience function to format comparison.
    
    Args:
        listing_a: First listing dict
        listing_b: Second listing dict
        user_query: Optional user query for context
        recommendation: Optional recommendation dict from DAL
        
    Returns:
        Formatted comparison string ready for display
    """
    return _formatter.format_comparison(
        listing_a=listing_a,
        listing_b=listing_b,
        user_query=user_query,
        recommendation=recommendation,
        user_profile=user_profile,
        debug_mode=debug_mode,
    )
