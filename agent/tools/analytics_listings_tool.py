from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List

from agent.common import get_logger, ExecutionMetrics
from agent.data_access import AgentListingDataAccess


logger = get_logger("tool_analytics_listings")


_PROPERTY_TYPE_CANONICAL_ALIASES: Dict[str, List[str]] = {
    "Chung cư": ["chung cu", "can ho", "apartment", "condo"],
    "Nhà phố": ["nha pho", "townhouse", "shophouse", "nha pho thuong mai"],
    "Đất": ["dat", "dat nen", "dat nen du an"],
    "Nhà riêng": ["nha rieng", "nha", "biet thu", "villa"],
}


def _normalize_vi_text(value: Any) -> str:
    text = str(value or "").strip().lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).strip()


def _is_property_type_count_intent(query: str) -> bool:
    qn = _normalize_vi_text(query)
    count_tokens = [
        "so luong",
        "bao nhieu",
        "phan bo",
        "co cau",
        "ti trong",
        "ty trong",
        "ti le",
        "ty le",
        "so voi",
        "phan tram",
    ]
    return any(token in qn for token in count_tokens)


def _detect_property_type_ratio_understanding(query: str) -> Dict[str, Any] | None:
    qn = _normalize_vi_text(query)
    ratio_tokens = ["ti le", "ty le", "so voi", "ratio", "phan tram", "giua", "vs"]
    if not any(token in qn for token in ratio_tokens):
        return None

    detected_positions: Dict[str, int] = {}
    for canonical, aliases in _PROPERTY_TYPE_CANONICAL_ALIASES.items():
        best_pos: int | None = None
        for alias in aliases:
            pattern = rf"\b{re.escape(alias)}\b"
            match = re.search(pattern, qn)
            if match:
                pos = int(match.start())
                if best_pos is None or pos < best_pos:
                    best_pos = pos
        if best_pos is not None:
            detected_positions[canonical] = best_pos

    if len(detected_positions) < 2:
        return None

    compare_values = [
        item[0]
        for item in sorted(detected_positions.items(), key=lambda pair: pair[1])[:2]
    ]

    return {
        "intent": "analytics_ratio",
        "metric": "count_ratio",
        "dimension": "property_type",
        "compare_values": compare_values,
    }


def _property_type_matches_group(property_type: Any, canonical: str) -> bool:
    pnorm = _normalize_vi_text(property_type)
    aliases = _PROPERTY_TYPE_CANONICAL_ALIASES.get(canonical) or [_normalize_vi_text(canonical)]
    return any(alias in pnorm for alias in aliases)


def _fmt_ty(value: Any) -> str | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return f"{val / 1_000_000_000:.2f} tỷ"


def _fmt_trieu_per_m2(value: Any) -> str | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return f"{val / 1_000_000:.2f} triệu/m²"


def _extract_reference_price_per_m2_vnd(query: str) -> float | None:
    text = _normalize_vi_text(query)
    if not text:
        return None

    # Examples: "87,72 trieu/m2", "0.09 ty/m2", "90tr/m2".
    patterns = [
        r"(\d+(?:[\.,]\d+)?)\s*(ty|ti|trieu|tr)\s*/\s*m2",
        r"(\d+(?:[\.,]\d+)?)\s*(ty|ti|trieu|tr)\s*(?:tren|moi)?\s*m2",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        try:
            amount = float(match.group(1).replace(",", "."))
        except (TypeError, ValueError):
            continue
        unit = str(match.group(2) or "").strip().lower()
        if unit in {"ty", "ti"}:
            return amount * 1_000_000_000
        if unit in {"trieu", "tr"}:
            return amount * 1_000_000
    return None


def _extract_reference_price_vnd(query: str) -> float | None:
    text = _normalize_vi_text(query)
    if not text:
        return None

    # Examples: "4.6 ty", "4,6 tỷ", "4600 trieu".
    # Keep this separate from per-m2 references so total price comparisons do not get lost.
    pattern = r"(\d+(?:[\.,]\d+)?)\s*(ty|ti|trieu|tr)(?!\s*/\s*m2)(?!\s*(?:tren|moi)?\s*m2)"
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        amount = float(match.group(1).replace(",", "."))
    except (TypeError, ValueError):
        return None

    unit = str(match.group(2) or "").strip().lower()
    if unit in {"ty", "ti"}:
        return amount * 1_000_000_000
    if unit in {"trieu", "tr"}:
        return amount * 1_000_000
    return None


def _market_heat_label(count: int) -> str:
    if count >= 300:
        return "sôi động"
    if count >= 80:
        return "khá sôi động"
    if count >= 20:
        return "mức trung bình"
    return "khá trầm"


def _price_level_label(avg_price_vnd: Any) -> str:
    try:
        val = float(avg_price_vnd)
    except (TypeError, ValueError):
        return "khó định vị"
    if val >= 20_000_000_000:
        return "mặt bằng cao"
    if val >= 8_000_000_000:
        return "mặt bằng trung-cao"
    if val >= 3_000_000_000:
        return "mặt bằng trung bình"
    return "mặt bằng mềm"


def _clean_signal_text(value: Any, max_len: int = 72) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _is_meaningful_signal(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    if value in {"-", ",", ";", "n/a", "N/A", "khong", "không"}:
        return False
    return any(ch.isalnum() for ch in value)


def _build_consultative_analytics_response(query: str, results: Dict[str, Any]) -> str:
    count = int(results.get("total_count") or results.get("count") or 0)
    avg_price = results.get("avg_price_vnd")
    min_price = results.get("min_price_vnd")
    max_price = results.get("max_price_vnd")
    avg_price_per_m2 = results.get("avg_price_per_m2_vnd")
    avg_area = results.get("avg_area_m2")
    location = str(results.get("location_context") or "thị trường mục tiêu").strip()
    metric_type = str(results.get("metric_type") or "default")
    district_breakdown = results.get("district_breakdown") or []
    ranking_scope = str(results.get("ranking_scope") or "district")
    ranking_rows = results.get("ranking_districts") or []
    insights = results.get("insights") or {}
    property_mix = results.get("breakdown") or []
    property_mix_insights = (insights.get("property_mix") or {}).get("top_types") or []
    legal_insights = insights.get("legal") or {}
    infra_insights = insights.get("infrastructure") or {}
    amenities_insights = insights.get("amenities") or {}
    user_fit_insights = insights.get("user_fit") or {}

    avg_price_text = _fmt_ty(avg_price) or "chưa đủ dữ liệu"
    price_range_text = ""
    min_text = _fmt_ty(min_price)
    max_text = _fmt_ty(max_price)
    if min_text and max_text:
        price_range_text = f"Biên giá ghi nhận từ {min_text} đến {max_text}."
    avg_ppm_text = _fmt_trieu_per_m2(avg_price_per_m2)
    reference_price_vnd = _extract_reference_price_vnd(query)
    reference_ppm_vnd = _extract_reference_price_per_m2_vnd(query)

    lines: List[str] = []

    # 1) HEADER SUMMARY
    lines.append(
        f"{location} hiện có khoảng **{count}** tin đăng, cho thấy nhịp giao dịch **{_market_heat_label(count)}**."
    )
    if avg_ppm_text:
        lines.append(
            f"Giá trung bình vào khoảng **{avg_price_text}**, tương đương mặt bằng **{avg_ppm_text}**; đây là khu vực {_price_level_label(avg_price)}."
        )
    else:
        lines.append(f"Giá trung bình vào khoảng **{avg_price_text}**, phản ánh khu vực {_price_level_label(avg_price)}.")

        if district_breakdown:
            lines.append("")
            lines.append("Khu vực nổi bật:")
            top_rows = district_breakdown[:3]
        for item in top_rows:
            district_name = _clean_signal_text(item.get("district") or item.get("ward") or "khu vực")
            district_count = int(item.get("count") or 0)
            district_avg_price = item.get("avg_price_vnd")
            district_avg_area = item.get("avg_area_m2")
            district_ppm = item.get("avg_price_per_m2_vnd")

            if district_avg_price is not None and avg_price is not None:
                rel_price = float(district_avg_price) / float(avg_price) if float(avg_price or 0) else 1.0
            else:
                rel_price = 1.0

            if district_count >= 80:
                insight = "nguồn cung dày nhất trong nhóm này"
            elif rel_price < 0.9:
                insight = "mặt bằng mềm hơn, dễ mở rộng diện tích"
            elif rel_price > 1.1:
                insight = "giá cao hơn mặt bằng chung, hợp nhu cầu vị trí"
            elif district_avg_area is not None and float(district_avg_area) >= 75:
                insight = "diện tích trung bình rộng hơn, phù hợp mua ở"
            else:
                insight = "cân bằng giữa giá và độ dày lựa chọn"

            price_text = _fmt_ty(district_avg_price) or "chưa đủ dữ liệu"
            area_text = f"{float(district_avg_area):.0f} m²" if district_avg_area is not None else "chưa rõ diện tích"
            ppm_text = _fmt_trieu_per_m2(district_ppm) or "chưa rõ giá/m²"
            lines.append(
                f"- {district_name}: {insight}; khoảng {district_count} listing, giá TB {price_text}, {area_text}, {ppm_text}."
            )

    if reference_price_vnd is not None and avg_price is not None and float(avg_price) > 0:
        delta_pct = ((float(reference_price_vnd) - float(avg_price)) / float(avg_price)) * 100.0
        if abs(delta_pct) <= 2.0:
            lines.append("Mức giá bạn nêu đang **xấp xỉ mặt bằng** khu vực.")
        elif delta_pct > 0:
            lines.append(
                f"Mức **{reference_price_vnd / 1_000_000_000:.2f} tỷ** đang **cao hơn** mặt bằng khoảng **{abs(delta_pct):.1f}%**."
            )
        else:
            lines.append(
                f"Mức **{reference_price_vnd / 1_000_000_000:.2f} tỷ** đang **thấp hơn** mặt bằng khoảng **{abs(delta_pct):.1f}%**."
            )

    if reference_ppm_vnd is not None and avg_price_per_m2 is not None and float(avg_price_per_m2) > 0:
        delta_pct = ((float(reference_ppm_vnd) - float(avg_price_per_m2)) / float(avg_price_per_m2)) * 100.0
        if abs(delta_pct) <= 2.0:
            lines.append("Mức giá bạn nêu đang **xấp xỉ mặt bằng** khu vực.")
        elif delta_pct > 0:
            lines.append(f"Mức **{reference_ppm_vnd / 1_000_000:.2f} triệu/m²** đang **cao hơn** mặt bằng khoảng **{abs(delta_pct):.1f}%**.")
        else:
            lines.append(f"Mức **{reference_ppm_vnd / 1_000_000:.2f} triệu/m²** đang **thấp hơn** mặt bằng khoảng **{abs(delta_pct):.1f}%**.")

    # Guard against accidental raw array text.
    text = "\n".join(line for line in lines if str(line).strip())
    return text.replace("[]", "")


def run_analytics_listings(advisor: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analytics tool for aggregate queries on listings.
    
    Supports:
    - Count queries: "bao nhieu listing o [location]"
    - Average price: "gia trung binh o [location]"
    - Price range: "gia tu X den Y o [location]"
    - Min/Max price: "gia re nhat / dat nhat o [location]"
    - District ranking: "Top 3 quận"
    - Price/m2 average: "Giá/m2 trung bình o [location]"
    - Property type breakdown: "Phân bố theo loại"
    """
    metrics = ExecutionMetrics()
    
    try:
        metrics.add_step("parse_args")
        query = str(args.get("query") or "").strip()
        
        if not query:
            metrics.set_error("empty_query")
            return {
                "tool": "analytics_listings",
                "status": "invalid_input",
                "found": False,
                "message": "Empty query",
                "results": {},
                "execution": metrics.finalize(),
            }
        
        # Check for ambiguous/unclear queries that need clarification
        from utils.vn_normalizer import normalize_query_pipeline
        
        qn = normalize_query_pipeline(str(query or "").lower())
        clarification_prompt = AgentListingDataAccess._detect_clarification_needed(qn)
        
        if clarification_prompt:
            logger.info("clarification_needed query=%s prompt=%s", query, clarification_prompt)
            return {
                "tool": "analytics_listings",
                "status": "need_clarification",
                "found": False,
                "message": clarification_prompt,
                "clarification_question": clarification_prompt,
                "results": {},
                "execution": metrics.finalize(),
            }
        
        metrics.add_step("initialize_dal")
        dal = AgentListingDataAccess(config_path=args.get("config_path"))
        
        logger.info("analytics_query query_len=%d", len(query))
        
        metrics.add_step("execute_analytics")
        stats = dal.get_listing_statistics(query=query)
        
        metrics.add_step("format_response")
        results = {
            "total_count": stats.get("total_count", 0),
            "count": stats.get("count", 0),
            "district": stats.get("district"),
            "ward": stats.get("ward"),
            "avg_price_vnd": stats.get("avg_price_vnd"),
            "avg_price_per_m2_vnd": stats.get("avg_price_per_m2_vnd"),
            "min_price_vnd": stats.get("min_price_vnd"),
            "max_price_vnd": stats.get("max_price_vnd"),
            "avg_area_m2": stats.get("avg_area_m2"),
            "metric_type": stats.get("metric_type", "default"),
            "area_distribution_requested": bool(stats.get("area_distribution_requested")),
            "ranking_scope": stats.get("ranking_scope", "district"),
            "ranking_metric": stats.get("ranking_metric", "count"),
            "ranking_min_count": stats.get("ranking_min_count"),
            "districts": stats.get("districts") or [],
            "wards": stats.get("wards") or [],
            "ward_breakdown": stats.get("ward_breakdown") or [],
            "district_breakdown": stats.get("district_breakdown") or [],
            "max_price_per_m2_vnd": stats.get("max_price_per_m2_vnd"),
            "max_price_per_m2_listing": stats.get("max_price_per_m2_listing"),
            "filters_applied": stats.get("filters_applied", {}),
            "location_context": stats.get("location_context", ""),
            "insights": stats.get("insights") or {},
            # For district_ranking metric type
            "ranking_districts": [],
            "limit": stats.get("limit"),
            # For property_type_grouping metric type
            "breakdown": stats.get("breakdown") or [],
            "breakdown_metric": stats.get("breakdown_metric", "count"),
            # Errors
            "error": stats.get("error"),
        }

        metric_type = str(results.get("metric_type") or "default")

        # Only expose ranking_districts when rows are district-like objects.
        if metric_type in {"district_ranking", "ward_ranking", "district_compare_avg_price"}:
            ranking_rows = stats.get("districts") or []
            if isinstance(ranking_rows, list):
                results["ranking_districts"] = [row for row in ranking_rows if isinstance(row, dict)]

        if metric_type == "property_type_grouping" and _is_property_type_count_intent(query):
            results["breakdown_metric"] = "count"
            breakdown_rows = results.get("breakdown") or []
            if breakdown_rows:
                results["breakdown"] = sorted(
                    breakdown_rows,
                    key=lambda item: int(item.get("count") or 0),
                    reverse=True,
                )

        ratio_understanding = None
        if metric_type == "property_type_grouping":
            ratio_understanding = _detect_property_type_ratio_understanding(query)
            if ratio_understanding:
                results["breakdown_metric"] = "count"
                results["query_understanding"] = ratio_understanding
        
        # Build natural language summary
        count = int(results.get("total_count") or results.get("count") or 0)
        avg_price = results.get("avg_price_vnd")
        avg_price_per_m2 = results.get("avg_price_per_m2_vnd")
        avg_area = results.get("avg_area_m2")
        district_breakdown = results.get("district_breakdown") or []
        max_ppm = results.get("max_price_per_m2_vnd")
        max_ppm_listing = results.get("max_price_per_m2_listing") or {}
        location = stats.get("location_context", "")
        
        summary = ""
        
        # Handle new metric types
        if metric_type in {"district_ranking", "ward_ranking"}:
            # For "Top 3 quận" type queries
            ranking_districts = results.get("ranking_districts") or []
            if ranking_districts:
                ranking_scope = str(results.get("ranking_scope") or "district")
                ranking_metric = str(results.get("ranking_metric") or "count")
                ranking_min_count = results.get("ranking_min_count")
                area_label = "wards" if ranking_scope == "ward" or metric_type == "ward_ranking" else "districts"
                district_lines = []
                for i, d in enumerate(ranking_districts, 1):
                    district_lines.append(f"{i}. {d.get('district')}: {d.get('count', 0)} listing (avg {d.get('avg_price_vnd', 0)/1e9:.2f} ty)")
                if ranking_metric == "avg_price_vnd":
                    summary = f"Top {area_label} ranked by average price:\n" + "\n".join(district_lines)
                    if ranking_min_count is not None:
                        summary += f"\n(Chi tinh khu vuc co it nhat {int(ranking_min_count)} listing de tranh outlier.)"
                else:
                    summary = f"Top {area_label} ranked by listing count:\n" + "\n".join(district_lines)
            else:
                summary = "No area ranking found"
        elif metric_type == "avg_price_per_m2":
            # For "Giá/m2 trung bình" type queries  
            if results.get("error"):
                summary = results["error"]
            else:
                district = results.get("district", "")
                avg_ppm = results.get("avg_price_per_m2_vnd")
                if avg_ppm is not None:
                    ppm_million = avg_ppm / 1_000_000
                    summary = f"Gia/m2 trung binh o {district} la khoang {ppm_million:.2f} trieu/m2 ({count} listing)"
                else:
                    summary = f"Khong co du lieu tinh gia/m2 o {district}"
        elif metric_type == "district_compare_avg_price":
            comparison_rows = results.get("ranking_districts") or []
            if comparison_rows:
                lines = []
                for item in comparison_rows[:8]:
                    district_name = str(item.get("district") or "Unknown")
                    avg_val = item.get("avg_price_vnd")
                    district_count = int(item.get("count") or 0)
                    if avg_val is None:
                        continue
                    lines.append(f"- {district_name}: {float(avg_val) / 1_000_000_000:.2f} ty ({district_count} listing)")
                if lines:
                    summary = "So sanh gia trung binh giua cac khu vuc:\n" + "\n".join(lines)
                else:
                    summary = "Khong du du lieu de so sanh gia trung binh giua cac khu vuc da neu."
            else:
                summary = results.get("error") or "Khong co du lieu de so sanh gia trung binh giua cac khu vuc da neu."
        elif metric_type == "property_type_grouping":
            # For "Phân bố theo loại" type queries
            breakdown = results.get("breakdown") or []
            breakdown_metric = str(results.get("breakdown_metric") or "count")
            if breakdown:
                top_rows = breakdown[:5]
                type_lines = []
                if breakdown_metric == "count":
                    query_norm = _normalize_vi_text(query)
                    total_type_count = sum(int(item.get("count") or 0) for item in breakdown) or 0
                    for item in top_rows:
                        ptype = item.get("property_type", "Unknown")
                        pcount = int(item.get("count") or 0)
                        share = (pcount * 100.0 / total_type_count) if total_type_count else 0.0
                        type_lines.append(f"{ptype}: {pcount} listing ({share:.1f}%)")

                    asks_apartment_vs_townhouse = (
                        ("can ho" in query_norm or "chung cu" in query_norm)
                        and "nha pho" in query_norm
                        and any(token in query_norm for token in ["ti le", "ty le", "so voi", "bao nhieu"])
                    )
                    ratio_request = ratio_understanding or _detect_property_type_ratio_understanding(query)
                    if ratio_request:
                        asks_apartment_vs_townhouse = True

                    if asks_apartment_vs_townhouse:
                        compare_values = (ratio_request or {}).get("compare_values") or ["Chung cư", "Nhà phố"]
                        left = str(compare_values[0]) if compare_values else "Chung cư"
                        right = str(compare_values[1]) if len(compare_values) > 1 else "Nhà phố"

                        left_count = 0
                        right_count = 0
                        for item in breakdown:
                            pcount = int(item.get("count") or 0)
                            ptype = item.get("property_type")
                            if _property_type_matches_group(ptype, left):
                                left_count += pcount
                            if _property_type_matches_group(ptype, right):
                                right_count += pcount

                        pair_total = left_count + right_count
                        left_pair_pct = (left_count * 100.0 / pair_total) if pair_total else 0.0
                        right_pair_pct = (right_count * 100.0 / pair_total) if pair_total else 0.0

                        if right_count > 0:
                            ratio = left_count / right_count
                            ratio_text = f"{ratio:.2f}:1"
                        elif left_count > 0:
                            ratio_text = "∞:1"
                        else:
                            ratio_text = "0:0"

                        summary = (
                            f"Ty le {left} so voi {right}:\n"
                            f"- {left}: {left_count} listing ({left_pair_pct:.1f}%)\n"
                            f"- {right}: {right_count} listing ({right_pair_pct:.1f}%)\n"
                            f"- Ty le {left}:{right} = {ratio_text}"
                        )
                    else:
                        summary = "Phan bo so luong listing theo loai bat dong san:\n" + "\n".join(type_lines)
                elif breakdown_metric == "avg_price_per_m2_vnd":
                    for item in top_rows:
                        ptype = item.get("property_type", "Unknown")
                        pcount = int(item.get("count") or 0)
                        ppm = item.get("avg_price_per_m2_vnd")
                        if ppm is None:
                            continue
                        type_lines.append(f"{ptype}: {ppm / 1_000_000:.2f} trieu/m2 ({pcount} listing)")
                    summary = "Gia/m2 trung binh theo loai bat dong san:\n" + "\n".join(type_lines)
                elif breakdown_metric == "avg_area_m2":
                    for item in top_rows:
                        ptype = item.get("property_type", "Unknown")
                        pcount = int(item.get("count") or 0)
                        avg_area_type = item.get("avg_area_m2")
                        if avg_area_type is None:
                            continue
                        type_lines.append(f"{ptype}: {avg_area_type:.1f} m2 ({pcount} listing)")
                    summary = "Dien tich trung binh theo loai bat dong san:\n" + "\n".join(type_lines)
                else:
                    for item in top_rows:
                        ptype = item.get("property_type", "Unknown")
                        pcount = int(item.get("count") or 0)
                        pavg = item.get("avg_price_vnd")
                        if pavg is None:
                            continue
                        type_lines.append(f"{ptype}: {pavg / 1_000_000_000:.2f} ty ({pcount} listing)")
                    summary = "Gia trung binh theo loai bat dong san:\n" + "\n".join(type_lines)

                if not type_lines:
                    summary = "Da tong hop theo loai bat dong san. Xem bieu do de so sanh chi tiet."
            else:
                summary = "Khong co du lieu"
        elif count == 0:
            filters_applied = results.get("filters_applied") or {}
            requested_tags = []
            if bool(filters_applied.get("near_metro")):
                requested_tags.append("gan metro")
            if bool(filters_applied.get("wants_gym")):
                requested_tags.append("co gym")
            if bool(filters_applied.get("wants_pool")):
                requested_tags.append("co ho boi")
            if bool(filters_applied.get("family_friendly")):
                requested_tags.append("phu hop gia dinh")

            if requested_tags:
                criteria_text = " va ".join(requested_tags)
                summary = (
                    f"Khong tim thay listing {criteria_text} trong {location or 'pham vi hien tai'}. "
                    "Ban co the mo rong khu vuc hoac bo sung ngan sach de toi tim them goi y phu hop."
                )
            else:
                summary = f"Khong tim thay listing trong {location}" if location else "Khong tim thay listing phu hop"
        else:
            if metric_type == "avg_area_m2" and avg_area is not None:
                summary_parts = [f"Dien tich trung binh la {avg_area:.2f} m2"]
                if location:
                    summary_parts.append(f"tai {location}")
                summary_parts.append(f"(dua tren {count} listing)")
                summary = " ".join(summary_parts)
            elif metric_type == "max_price_per_m2" and max_ppm is not None:
                ppm_million = max_ppm / 1_000_000
                title = str(max_ppm_listing.get("title") or "listing")
                district = str(max_ppm_listing.get("district") or "").strip()
                district_part = f" o {district}" if district else ""
                summary = (
                    f"Listing co gia/m2 cao nhat la {title}{district_part}, "
                    f"khoang {ppm_million:.2f} trieu/m2."
                )
            else:
                summary_parts = [f"Co {count} listing"]
                if location:
                    summary_parts.append(f"o {location}")

                if district_breakdown:
                    pairs = [
                        f"{int(item.get('count') or 0)} o {str(item.get('district') or '').strip()}"
                        for item in district_breakdown[:4]
                        if str(item.get('district') or '').strip()
                    ]
                    if pairs:
                        summary_parts.append("(" + "; ".join(pairs) + ")")

                if avg_price is not None:
                    avg_price_billions = avg_price / 1_000_000_000
                    summary_parts.append(f"voi gia trung binh {avg_price_billions:.2f} ty")

                if avg_price_per_m2 is not None:
                    ppm_million = avg_price_per_m2 / 1_000_000
                    summary_parts.append(f"gia/m2 trung binh {ppm_million:.2f} trieu")

                summary = " ".join(summary_parts)
        
        found = count > 0
        status = "ok" if found else "not_found"

        if found and metric_type == "default":
            summary = _build_consultative_analytics_response(query=query, results=results)
        
        logger.info("analytics_result count=%d status=%s location=%s", count, status, location)
        
        return {
            "tool": "analytics_listings",
            "status": status,
            "found": found,
            "count": count,
            "query": query,
            "results": results,
            "summary": summary,
            "message": summary,
            "retrieval_stats": {
                "retrieval_mode": "analytics",
                "total_count": count,
                "filters_applied": results.get("filters_applied", {}),
            },
            "execution": metrics.finalize(),
        }
    except Exception as exc:
        metrics.set_error(type(exc).__name__)
        logger.exception("analytics_listings failed query=%s error=%s", args.get("query", "?"), exc)
        return {
            "tool": "analytics_listings",
            "status": "tool_error",
            "found": False,
            "count": 0,
            "query": str(args.get("query") or "").strip(),
            "results": {},
            "message": f"Analytics failed: {exc}",
            "retrieval_stats": {},
            "execution": metrics.finalize(),
        }
