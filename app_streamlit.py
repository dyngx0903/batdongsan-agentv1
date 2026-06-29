from __future__ import annotations

import json
import unicodedata
import uuid
from typing import Any, Dict

from datetime import datetime

import streamlit as st

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import plotly.express as px
except Exception:
    px = None

from batdongsan_ai_chatbot import BatdongsanAIChatbot
from ui.theme import inject_global_styles


st.set_page_config(page_title="Batdongsan AI Agent", page_icon="🏠", layout="wide")

ALLOWED_TOOLS = [
    "search_listings",
    "explain_listing",
    "similar_listings",
    "compare_listings",
    "suggest_area",
    "analytics_listings",
    "respond_to_user",
]

if "chatbot" not in st.session_state:
    st.session_state.chatbot = BatdongsanAIChatbot()
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "prefill_prompt" not in st.session_state:
    st.session_state.prefill_prompt = ""
if "ui_preset" not in st.session_state:
    st.session_state.ui_preset = "Basic"
if "last_applied_preset" not in st.session_state:
    st.session_state.last_applied_preset = ""
if "selected_tool" not in st.session_state:
    st.session_state.selected_tool = "auto"
if "enable_llm_router_fallback" not in st.session_state:
    st.session_state.enable_llm_router_fallback = True
if "enable_llm_response_composer" not in st.session_state:
    st.session_state.enable_llm_response_composer = True
if "enable_llm_input_preprocessor" not in st.session_state:
    st.session_state.enable_llm_input_preprocessor = True
if "enable_llm_input_primary" not in st.session_state:
    st.session_state.enable_llm_input_primary = True
if "enable_best_effort_response" not in st.session_state:
    st.session_state.enable_best_effort_response = True
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []
if "conversation_ended" not in st.session_state:
    st.session_state.conversation_ended = False


def _render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Batdongsan AI Agent</h1>
            <p>Tro ly tim kiem, so sanh va phan tich bat dong san theo nhu cau thuc te.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _apply_ui_preset(preset: str) -> None:
    if preset == st.session_state.last_applied_preset:
        return

    if preset not in {"Basic", "Debug"}:
        preset = "Basic"

    if preset == "Basic":
        st.session_state.selected_tool = "auto"
        st.session_state.enable_llm_router_fallback = True
        st.session_state.enable_llm_response_composer = True
        st.session_state.enable_llm_input_preprocessor = True
        st.session_state.enable_llm_input_primary = True
        st.session_state.enable_best_effort_response = True
        st.session_state.debug_mode = False
    elif preset == "Debug":
        st.session_state.selected_tool = "auto"
        st.session_state.enable_llm_router_fallback = True
        st.session_state.enable_llm_response_composer = True
        st.session_state.enable_llm_input_preprocessor = True
        st.session_state.enable_llm_input_primary = True
        st.session_state.enable_best_effort_response = True
        st.session_state.debug_mode = True

    st.session_state.last_applied_preset = preset


def _latest_assistant_raw() -> Dict[str, Any]:
    for msg in reversed(st.session_state.messages):
        if msg.get("role") == "assistant" and isinstance(msg.get("raw"), dict):
            return msg["raw"]
    return {}


def _render_agent_status(raw: Dict[str, Any]) -> None:
    process_sequence = raw.get("process_sequence") or []
    if not process_sequence:
        return

    st.markdown("#### Agent status")
    for step in process_sequence:
        status = str(step.get("status") or "").lower()
        icon = "🟢" if status == "ok" else "🟡" if status in {"need_clarification"} else "🔴"
        label = str(step.get("step") or "step")
        tool = str(step.get("tool") or "")
        latency_ms = step.get("latency_ms")
        latency_text = f" | {latency_ms} ms" if latency_ms is not None else ""
        st.caption(f"{icon} {label} -> {tool}{latency_text}")


def _render_memory_panel(raw: Dict[str, Any]) -> None:
    metadata = raw.get("metadata") or {}
    normalized_slots = metadata.get("normalized_slots") or {}
    slots = normalized_slots.get("slots") or {}
    profile = normalized_slots.get("user_profile") or {}
    parsed_filters = metadata.get("parsed_filters") or {}

    memory_rows = []
    for key, value in slots.items():
        if value is not None and str(value).strip() != "":
            memory_rows.append({"field": key, "value": str(value)})

    for key, value in profile.items():
        if value is not None and str(value).strip() != "":
            memory_rows.append({"field": f"profile.{key}", "value": str(value)})

    for key, value in parsed_filters.items():
        if value is not None and str(value).strip() != "":
            memory_rows.append({"field": f"filter.{key}", "value": str(value)})

    st.markdown("#### Memory panel")
    if not memory_rows:
        st.caption("Chua co du lieu nho ngu canh trong phien hien tai.")
        return

    if pd is not None:
        st.dataframe(pd.DataFrame(memory_rows), use_container_width=True, hide_index=True)
    else:
        st.table(memory_rows)


def _render_analytics_charts(raw: Dict[str, Any], key_prefix: str = "analytics") -> None:
    metadata = raw.get("metadata") or {}
    last_result = metadata.get("last_tool_result") or {}
    if str(last_result.get("tool") or "") != "analytics_listings":
        return

    last_status = str(last_result.get("status") or "").strip().lower()
    if last_status in {"need_clarification", "not_found", "tool_error", "invalid_input", "internal_error"}:
        return

    results = last_result.get("results") or {}
    def _dict_rows(rows: Any) -> list[Dict[str, Any]]:
        if not isinstance(rows, list):
            return []
        return [item for item in rows if isinstance(item, dict)]

    ward_breakdown = _dict_rows(results.get("ward_breakdown") or results.get("wards") or [])
    district_breakdown = _dict_rows(results.get("district_breakdown") or [])
    ranking_districts = _dict_rows(results.get("ranking_districts") or [])
    breakdown = _dict_rows(results.get("breakdown") or [])
    breakdown_metric = str(results.get("breakdown_metric") or "count")
    query_understanding = results.get("query_understanding") or {}
    metric_type = str(results.get("metric_type") or "")
    area_distribution_requested = bool(results.get("area_distribution_requested"))
    is_ranking_query = metric_type in {"district_ranking", "ward_ranking"}
    is_area_distribution = False
    area_axis_title = "Khu vuc"

    def _infer_area_axis_title(labels: list[str]) -> str:
        normalized = [str(lbl or "").strip().lower() for lbl in labels if str(lbl or "").strip()]
        if not normalized:
            return "Khu vuc"
        if any("tp" in v or "thanh pho" in v or "city" in v for v in normalized):
            return "Thanh pho"
        if any("phuong" in v or "xa" in v or "ward" in v or "commune" in v for v in normalized):
            return "Phuong/Xa"
        if any("quan" in v or "huyen" in v or "district" in v for v in normalized):
            return "Quan/Huyen"
        return "Khu vuc"

    total_count = int(results.get("total_count") or results.get("count") or 0)
    avg_price_vnd = results.get("avg_price_vnd")
    avg_price_per_m2_vnd = results.get("avg_price_per_m2_vnd")
    avg_area_m2 = results.get("avg_area_m2")
    min_price_vnd = results.get("min_price_vnd")
    max_price_vnd = results.get("max_price_vnd")

    def _fmt_vnd_short(value: Any) -> str:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return "N/A"
        if val >= 1_000_000_000:
            return f"{val / 1_000_000_000:.2f} ty"
        if val >= 1_000_000:
            return f"{val / 1_000_000:.0f} tr"
        return f"{val:.0f} VND"

    has_area_breakdown = area_distribution_requested and bool(ward_breakdown or district_breakdown)
    has_chart_mode = bool(has_area_breakdown or ranking_districts or (metric_type == "property_type_grouping" and breakdown))
    debug_mode = bool(st.session_state.get("debug_mode", False))
    if not has_chart_mode:
        # KPI strip for quick executive-level scan
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Tong listing", f"{total_count:,}")
        kpi_cols[1].metric("Gia TB", _fmt_vnd_short(avg_price_vnd))
        kpi_cols[2].metric("Gia/m2 TB", _fmt_vnd_short(avg_price_per_m2_vnd))
        kpi_cols[3].metric("Dien tich TB", f"{float(avg_area_m2):.1f} m2" if avg_area_m2 is not None else "N/A")

        if debug_mode and (min_price_vnd is not None or max_price_vnd is not None):
            st.caption(f"Khoang gia quan sat: {_fmt_vnd_short(min_price_vnd)} - {_fmt_vnd_short(max_price_vnd)}")

    chart_rows = []
    chart_title = "Phan bo listing"
    if area_distribution_requested and ward_breakdown:
        chart_rows = [
            {
                "label": str(item.get("ward") or "Unknown"),
                "count": int(item.get("count") or 0),
                "avg_price_per_m2_vnd": item.get("avg_price_per_m2_vnd"),
                "avg_area_m2": item.get("avg_area_m2"),
            }
            for item in ward_breakdown
        ]
        chart_title = "Phan bo theo phuong/xa"
        is_area_distribution = True
        area_axis_title = "Phuong/Xa"
    elif area_distribution_requested and district_breakdown:
        chart_rows = [
            {
                "label": str(item.get("district") or "Unknown"),
                "count": int(item.get("count") or 0),
                "avg_price_per_m2_vnd": item.get("avg_price_per_m2_vnd"),
                "avg_area_m2": item.get("avg_area_m2"),
            }
            for item in district_breakdown
        ]
        chart_title = "Phan bo theo quan/huyen"
        is_area_distribution = True
        area_axis_title = _infer_area_axis_title([row.get("label", "") for row in chart_rows])
    elif ranking_districts:
        chart_rows = [
            {
                "label": str(item.get("district") or "Unknown"),
                "count": int(item.get("count") or 0),
                "avg_price_per_m2_vnd": item.get("avg_price_per_m2_vnd"),
                "avg_area_m2": item.get("avg_area_m2"),
            }
            for item in ranking_districts
        ]
        chart_title = "Top quan/huyen theo so luong tin"
        is_area_distribution = True
        area_axis_title = _infer_area_axis_title([row.get("label", "") for row in chart_rows])
    elif breakdown:
        chart_rows = [
            {
                "label": str(item.get("property_type") or "Unknown"),
                "count": int(item.get("count") or 0),
                "avg_price_vnd": item.get("avg_price_vnd"),
                "avg_price_per_m2_vnd": item.get("avg_price_per_m2_vnd"),
                "avg_area_m2": item.get("avg_area_m2"),
            }
            for item in breakdown
        ]
        chart_title = "Co cau theo loai bat dong san"

    if chart_rows and pd is not None:
        df = pd.DataFrame(chart_rows)

        is_property_ratio_query = (
            metric_type == "property_type_grouping"
            and breakdown_metric == "count"
            and str(query_understanding.get("intent") or "").strip().lower() == "analytics_ratio"
            and str(query_understanding.get("dimension") or "").strip().lower() == "property_type"
        )

        def _norm_text(value: Any) -> str:
            raw_text = str(value or "").strip().lower().replace("đ", "d")
            decomposed = unicodedata.normalize("NFKD", raw_text)
            return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).strip()

        if px is not None and is_property_ratio_query:
            compare_values = query_understanding.get("compare_values") or []

            def _matches_group(property_label: Any, canonical: str) -> bool:
                plabel = _norm_text(property_label)
                ctext = _norm_text(canonical)
                if ctext in {"chung cu", "can ho"}:
                    return any(token in plabel for token in ["chung cu", "can ho", "apartment", "condo"])
                if ctext == "nha pho":
                    return any(token in plabel for token in ["nha pho", "townhouse", "shophouse", "nha pho thuong mai"])
                return ctext in plabel

            pie_rows = []
            used_indices: set[int] = set()
            for canonical in compare_values:
                count_val = 0
                for idx, item in enumerate(chart_rows):
                    if _matches_group(item.get("label"), str(canonical)):
                        count_val += int(item.get("count") or 0)
                        used_indices.add(idx)
                pie_rows.append({"label": str(canonical), "count": count_val})

            # Fallback: if mapping produced no usable values, use top two labels by count.
            if not pie_rows or all(int(item.get("count") or 0) == 0 for item in pie_rows):
                top_two = sorted(chart_rows, key=lambda item: int(item.get("count") or 0), reverse=True)[:2]
                pie_rows = [
                    {"label": str(item.get("label") or "Unknown"), "count": int(item.get("count") or 0)}
                    for item in top_two
                ]

            pie_df = pd.DataFrame(pie_rows)
            pie_df = pie_df[pie_df["count"] > 0]
            if not pie_df.empty:
                fig_ratio_pie = px.pie(
                    pie_df,
                    values="count",
                    names="label",
                    hole=0.45,
                    color_discrete_sequence=["#1E5B99", "#0B3C78", "#2F76B2", "#6BAED6"],
                )
                fig_ratio_pie.update_traces(textposition="inside", textinfo="percent+label")
                fig_ratio_pie.update_layout(margin=dict(l=16, r=16, t=18, b=8), height=360)
                st.plotly_chart(fig_ratio_pie, use_container_width=True, key=f"{key_prefix}_ratio_pair_pie")
            return

        if px is not None and is_area_distribution:
            area_df = df.sort_values("count", ascending=False).copy()
            area_df["avg_price_per_m2_trieu"] = area_df["avg_price_per_m2_vnd"].apply(
                lambda v: (float(v) / 1_000_000) if v is not None else None
            )
            area_df["avg_area_m2"] = area_df["avg_area_m2"].apply(
                lambda v: float(v) if v is not None else None
            )

            metric_df = area_df.melt(
                id_vars=["label"],
                value_vars=["count", "avg_price_per_m2_trieu", "avg_area_m2"],
                var_name="metric",
                value_name="value",
            )
            metric_df = metric_df[metric_df["value"].notna()]
            metric_name_map = {
                "count": "So luong",
                "avg_price_per_m2_trieu": "Gia/m² TB (trieu)",
                "avg_area_m2": "Dien tich TB (m²)",
            }
            metric_df["metric_label"] = metric_df["metric"].map(metric_name_map)

            fig = px.bar(
                metric_df,
                x="label",
                y="value",
                color="metric_label",
                barmode="group",
                text="value",
                color_discrete_map={
                    "So luong": "#0B3C78",
                    "Gia/m² TB (trieu)": "#1E5B99",
                    "Dien tich TB (m²)": "#2F76B2",
                },
                labels={
                    "label": area_axis_title,
                    "value": "Gia tri",
                    "metric_label": "Chi so",
                },
            )
            fig.update_layout(
                height=440,
                margin=dict(l=16, r=16, t=18, b=8),
                xaxis_title=area_axis_title,
                yaxis_title="Gia tri",
                legend_title_text="Chi so",
            )
            fig.update_traces(
                texttemplate="%{y:.1f}",
                textposition="outside",
                cliponaxis=False,
                marker_line_color="#102D4A",
                marker_line_width=0.8,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_area_grouped_bar")
        elif px is not None and metric_type == "property_type_grouping":
            metric_map = {
                "count": {
                    "value_col": "count",
                    "axis_title": "So luong listing",
                    "chart_title": "Phan bo so luong listing theo loai bat dong san",
                    "convert": lambda v: float(v),
                    "format": lambda v: f"{int(round(v))} listing",
                    "hover": "So luong",
                },
                "avg_price_vnd": {
                    "value_col": "avg_price_vnd",
                    "axis_title": "Gia trung binh (ty VND)",
                    "chart_title": "Gia trung binh theo loai bat dong san",
                    "convert": lambda v: float(v) / 1_000_000_000,
                    "format": lambda v: f"{v:.2f} ty",
                    "hover": "Gia TB",
                },
                "avg_price_per_m2_vnd": {
                    "value_col": "avg_price_per_m2_vnd",
                    "axis_title": "Gia/m² trung binh (trieu)",
                    "chart_title": "Gia/m² trung binh theo loai bat dong san",
                    "convert": lambda v: float(v) / 1_000_000,
                    "format": lambda v: f"{v:.2f} tr/m²",
                    "hover": "Gia/m² TB",
                },
                "avg_area_m2": {
                    "value_col": "avg_area_m2",
                    "axis_title": "Dien tich trung binh (m²)",
                    "chart_title": "Dien tich trung binh theo loai bat dong san",
                    "convert": lambda v: float(v),
                    "format": lambda v: f"{v:.1f} m²",
                    "hover": "Dien tich TB",
                },
            }
            selected = metric_map.get(breakdown_metric, metric_map["count"])
            value_col = selected["value_col"]

            metric_df = df[["label", value_col]].copy()
            metric_df = metric_df[metric_df[value_col].notna()]
            metric_df["value"] = metric_df[value_col].apply(selected["convert"])
            metric_df["value_text"] = metric_df["value"].apply(selected["format"])
            metric_df = metric_df.sort_values("value", ascending=True)

            if not metric_df.empty:
                fig = px.bar(
                    metric_df,
                    x="value",
                    y="label",
                    orientation="h",
                    color_discrete_sequence=["#1E5B99"],
                    text="value_text",
                    labels={
                        "label": "Loai hinh",
                        "value": selected["axis_title"],
                    },
                )
                fig.update_layout(
                    height=max(320, 48 * len(metric_df)),
                    margin=dict(l=16, r=16, t=18, b=8),
                    xaxis_title=selected["axis_title"],
                    yaxis_title="",
                    showlegend=False,
                )
                fig.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                    marker_line_color="#102D4A",
                    marker_line_width=0.8,
                    hovertemplate=f"<b>%{{y}}</b><br>{selected['hover']}: %{{text}}<extra></extra>",
                )
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_property_metric_bar")
        elif px is not None:
            fig = px.bar(
                df.sort_values("count", ascending=True),
                x="count",
                y="label",
                orientation="h",
                color="count",
                color_continuous_scale="Blues",
                text="count",
            )
            fig.update_layout(
                height=max(320, 48 * len(df)),
                margin=dict(l=16, r=16, t=18, b=8),
                coloraxis_showscale=False,
                xaxis_title="So luong",
                yaxis_title="",
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_fallback_count_bar")
        else:
            st.bar_chart(df.set_index("label"))

        if breakdown and px is not None and metric_type != "property_type_grouping":
            pie_df = pd.DataFrame(
                [
                    {
                        "label": str(item.get("property_type") or "Unknown"),
                        "count": int(item.get("count") or 0),
                    }
                    for item in breakdown
                ]
            )
            if not pie_df.empty:
                fig_pie = px.pie(pie_df, values="count", names="label", hole=0.45)
                fig_pie.update_layout(margin=dict(l=16, r=16, t=18, b=8), height=360)
                st.plotly_chart(fig_pie, use_container_width=True, key=f"{key_prefix}_type_share_pie")

    elif chart_rows:
        if st.session_state.get("debug_mode", False):
            st.caption("Da co du lieu phan bo nhung khong du dieu kien de ve chart.")
    else:
        if st.session_state.get("debug_mode", False):
            st.caption("Khong co du lieu phan bo de ve chart cho truy van nay.")

    filters_applied = results.get("filters_applied") or {}
    if debug_mode and filters_applied:
        with st.expander("Filters applied"):
            st.json(filters_applied)


def _normalize_text(text: str) -> str:
    raw = str(text or "").strip().lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFKD", raw)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).strip()


def _is_end_conversation_command(text: str) -> bool:
    normalized = _normalize_text(text)
    return normalized in {"ket thuc", "kết thúc", "end", "stop", "done", "finish"}


def _render_end_conversation_feedback(raw: Dict[str, Any]) -> None:
    if not isinstance(raw, dict) or not raw:
        return

    st.markdown("### Feedback cuoi hoi thoai")
    st.caption("Danh gia tong the chat nay de cai thien chat luong agent.")

    col1, col2 = st.columns(2)
    key_up = f"fb_up_end_{st.session_state.session_id}"
    key_down = f"fb_down_end_{st.session_state.session_id}"

    if col1.button("Huu ich", key=key_up, use_container_width=True):
        st.session_state.feedback_log.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": st.session_state.session_id,
                "rating": "positive",
                "summary": (raw.get("final_response") or {}).get("summary", ""),
                "tool": (raw.get("metadata") or {}).get("selected_tool", ""),
            }
        )
        st.success("Da ghi nhan feedback tong the: tich cuc.")

    if col2.button("Chua dung", key=key_down, use_container_width=True):
        st.session_state.feedback_log.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": st.session_state.session_id,
                "rating": "negative",
                "summary": (raw.get("final_response") or {}).get("summary", ""),
                "tool": (raw.get("metadata") or {}).get("selected_tool", ""),
            }
        )
        st.warning("Da ghi nhan feedback tong the: can cai thien.")


def _render_top_options(top_options: list[Dict[str, Any]], *, key_prefix: str = "top", show_verbose: bool = True) -> None:
    def _has_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return False
            normalized = text.lower()
            if normalized in {"[]", "{}", "null", "none", "n/a", "na", "khong", "không"}:
                return False
            return True
        if isinstance(value, (list, tuple, set, dict)):
            return bool(value)
        return True

    def _to_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            chunks = [str(v).strip() for v in value.values() if str(v).strip()]
            return ", ".join(chunks)
        if isinstance(value, (list, tuple, set)):
            chunks = [str(v).strip() for v in value if str(v).strip()]
            return ", ".join(chunks)
        text = str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    chunks = [str(v).strip() for v in parsed if str(v).strip()]
                    if chunks:
                        return ", ".join(chunks)
            except Exception:
                pass
        return text

    def _render_listing_detail_sections(item: Dict[str, Any], *, section_key: str) -> None:
        core_fields = [
            ("Giá", item.get("price_text")),
            ("Diện tích", item.get("area_text")),
            ("Loại hình", item.get("property_type")),
            ("Dự án", item.get("project")),
            ("Quận", item.get("district")),
            ("Phường", item.get("ward")),
            ("Đường", item.get("street")),
            ("Pháp lý", item.get("legal_status")),
            ("Nội thất", item.get("interior")),
            ("Hướng", item.get("direction")),
            ("Kết cấu", item.get("structure")),
        ]

        enrichment_fields = [
            ("Vị trí", item.get("location_quality")),
            ("Khu vực lân cận", item.get("neighborhood_quality")),
            ("Tầm nhìn", item.get("view")),
            ("Tiện ích", item.get("amenities_building")),
            ("Tiện ích khu vực", item.get("amenities_area")),
            ("Gần landmarks", item.get("nearby_landmarks")),
            ("Gần giao thông", item.get("nearby_transport")),
            ("Gần đường", item.get("nearby_roads")),
            ("Phù hợp cho", item.get("suitable_for")),
            ("Lối vào", item.get("access")),
        ]

        filtered_core = [(label, _to_text(value)) for label, value in core_fields if _has_value(value)]
        filtered_enrichment = [(label, _to_text(value)) for label, value in enrichment_fields if _has_value(value)]

        if filtered_core:
            st.markdown("**Thông tin chính**")
            for label, value in filtered_core:
                if label in {"Giá", "Diện tích"} and value:
                    st.markdown(f"- **{label}**: {value}")
                else:
                    st.markdown(f"- {label}: {value}")

        if filtered_enrichment:
            st.markdown("**Điểm nổi bật**")
            for label, value in filtered_enrichment:
                st.markdown(f"- {label}: {value}")

        listing_key = str(item.get("listing_ref") or item.get("listing_id") or "unknown").strip()
        listing_key = listing_key.replace("/", "_").replace(" ", "_")

        url = str(item.get("url") or "").strip()
        if url:
            if st.button("🔗 Xem trực tiếp", key=f"{section_key}_view_direct_{listing_key}"):
                st.markdown(f"[Mở trang chi tiết]({url})")

        st.markdown("**Tiếp theo**")
        district = str(item.get("district") or "").strip()
        price = str(item.get("price_text") or "").strip()
        area = str(item.get("area_text") or "").strip()
        suggestions = []
        if district:
            suggestions.append(f"So sánh căn này với 2 căn cùng tầm giá ở {district}")
        if area and "m²" in area:
            suggestions.append(f"Tìm các căn tương tự nhưng diện tích lớn hơn")
        if price and district:
            suggestions.append(f"Phân tích xem mức giá {price} có cao hơn mặt bằng {district} hay không")
        if not suggestions:
            suggestions = [
                "Tìm các căn tương tự ở cùng dự án",
                "So sánh với các căn khác cùng tầm giá",
                "Phân tích xu hướng giá ở khu vực này",
            ]
        for suggestion_idx, suggestion in enumerate(suggestions[:3], start=1):
            if st.button(suggestion, key=f"{section_key}_suggest_{suggestion_idx}_{listing_key}"):
                st.session_state.prefill_prompt = suggestion
                st.rerun()

    visible_items = top_options
    if not visible_items:
        return

    has_full_detail = any(str(item.get("detail_view") or "").strip().lower() == "full" for item in visible_items)
    if show_verbose or not has_full_detail:
        st.markdown("#### Đề xuất")
    for idx, item in enumerate(visible_items, start=1):
        is_full_detail = str(item.get("detail_view") or "").strip().lower() == "full"
        title = item.get("title") or item.get("district") or item.get("winner_ref") or f"Option {idx}"
        if not is_full_detail:
            score = item.get("fit_score")
            badge = f" | fit_score: {score}" if score is not None else ""
            st.markdown(f"**{idx}. {title}{badge}**")
        elif show_verbose:
            st.markdown(f"**Chi tiết listing: {title}**")

        info_parts = []
        if item.get("price_text"):
            info_parts.append(f"Gia: {item['price_text']}")
        if item.get("area_text"):
            info_parts.append(f"Dien tich: {item['area_text']}")
        if item.get("district"):
            info_parts.append(f"Khu vuc: {item['district']}")
        if item.get("project"):
            info_parts.append(f"Du an: {item['project']}")
        if info_parts and (show_verbose or not is_full_detail):
            st.markdown(f"**{' | '.join(info_parts)}**")

        if item.get("summary") and (show_verbose or not is_full_detail):
            st.write(item.get("summary"))
        if item.get("winner_reason") and (show_verbose or not is_full_detail):
            st.caption(f"Ly do noi bat: {item.get('winner_reason')}")

        # Show richer detail block when available (especially for explain_listing).
        if is_full_detail and show_verbose:
            detail_parts = []
            if item.get("transaction_type"):
                detail_parts.append(f"Giao dich: {item.get('transaction_type')}")
            if item.get("property_type"):
                detail_parts.append(f"Loai hinh: {item.get('property_type')}")
            if item.get("project"):
                detail_parts.append(f"Du an: {item.get('project')}")
            if item.get("ward"):
                detail_parts.append(f"Phuong: {item.get('ward')}")
            if item.get("street"):
                detail_parts.append(f"Duong: {item.get('street')}")
            if item.get("legal_status"):
                detail_parts.append(f"Phap ly: {item.get('legal_status')}")
            if item.get("floors") is not None:
                detail_parts.append(f"So tang: {item.get('floors')}")
            if item.get("frontage_width_m") is not None:
                detail_parts.append(f"Mat tien: {item.get('frontage_width_m')} m")
            if item.get("road_access_width_m") is not None:
                detail_parts.append(f"Hem/duong vao: {item.get('road_access_width_m')} m")
            if item.get("direction"):
                detail_parts.append(f"Huong: {item.get('direction')}")
            if detail_parts:
                st.caption(" | ".join(detail_parts))

        if is_full_detail:
            _render_listing_detail_sections(item, section_key=f"{key_prefix}_{idx}")

        if not is_full_detail:
            utilities = item.get("utilities")
            if isinstance(utilities, list) and utilities:
                st.caption("Tien ich: " + ", ".join(str(x) for x in utilities[:8]))

            enrichment_matches = item.get("enrichment_matches")
            if isinstance(enrichment_matches, list) and enrichment_matches:
                st.caption("Enrichment match: " + ", ".join(str(x) for x in enrichment_matches[:6]))

        listing_ref = str(item.get("listing_ref") or "").strip()
        if not listing_ref:
            source = str(item.get("source") or "").strip()
            listing_id = str(item.get("listing_id") or "").strip()
            if source and listing_id:
                listing_ref = f"{source}/{listing_id}"

        if listing_ref and not is_full_detail:
            detail_prompt = f"Xem chi tiet listing {listing_ref}"
            if st.button("Xem chi tiet", key=f"{key_prefix}_detail_{idx}_{listing_ref}", use_container_width=False):
                st.session_state.prefill_prompt = detail_prompt
                st.rerun()

        st.divider()


def _render_sections(final_response: Dict[str, Any], *, key_prefix: str = "top", raw: Dict[str, Any] | None = None) -> None:
    selected_tool = ""
    if isinstance(raw, dict):
        metadata = raw.get("metadata") or {}
        selected_tool = str(metadata.get("selected_tool") or "").strip().lower()
        if not selected_tool:
            last_tool_result = metadata.get("last_tool_result") or {}
            selected_tool = str(last_tool_result.get("tool") or "").strip().lower()

    top_options = final_response.get("top_options") or []
    has_full_detail = any(str(item.get("detail_view") or "").strip().lower() == "full" for item in top_options)
    show_verbose = bool(st.session_state.debug_mode)
    summary = final_response.get("summary") or "Khong co noi dung phan hoi"
    if show_verbose or not has_full_detail:
        st.write(summary)

    if top_options and selected_tool != "compare_listings":
        _render_top_options(top_options, key_prefix=key_prefix, show_verbose=show_verbose)

    reasons = final_response.get("reasons") or []
    cautions = final_response.get("cautions") or []
    if st.session_state.debug_mode and selected_tool != "compare_listings":
        if reasons:
            st.markdown("#### Ly do")
            for reason in reasons:
                st.markdown(f"- {reason}")

        if cautions:
            st.markdown("#### Luu y")
            for caution in cautions:
                st.markdown(f"- {caution}")

    questions = final_response.get("next_questions") or []
    if questions:
        st.markdown("#### Can bo sung")
        for question in questions:
            st.markdown(f"- {question}")

    next_step = final_response.get("next_step")
    if next_step and selected_tool != "compare_listings":
        st.markdown("#### Buoc tiep theo")
        st.info(str(next_step))


inject_global_styles()
_render_hero()

left_col, right_col = st.columns([2.1, 1], gap="large")

with left_col:
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="panel">
                <p class="panel-title">Bắt đầu nhanh</p>
                <p class="panel-sub">Nhập nhu cầu của bạn vào ô chat bên dưới, hoặc chọn một câu hỏi mẫu trong sidebar.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right_col:
    st.markdown(
        """
        <div class="quick-tip">
            Mẹo: mở <b>Advanced agent settings</b> để force tool, bắt debug JSON, hoặc bật/tắt các chế độ LLM.
        </div>
        """,
        unsafe_allow_html=True,
    )
    latest_raw = _latest_assistant_raw()
    _render_agent_status(latest_raw)

with st.sidebar:
    if st.session_state.debug_mode:
        st.subheader("Session")
        st.caption("Session id")
        st.code(st.session_state.session_id)

    st.subheader("Preset")
    preset_options = ["Basic", "Debug"]
    if st.session_state.ui_preset not in preset_options:
        st.session_state.ui_preset = "Basic"
    st.session_state.ui_preset = st.selectbox(
        "User mode",
        options=preset_options,
        index=preset_options.index(st.session_state.ui_preset),
        help="Preset giao dien/cau hinh: Basic hoac Debug.",
    )
    _apply_ui_preset(st.session_state.ui_preset)

    top_k = st.slider("Top K results", min_value=3, max_value=20, value=5, step=1)

    if st.session_state.debug_mode:
        st.markdown("---")
        _render_memory_panel(_latest_assistant_raw())

    st.markdown("---")
    st.subheader("Prompt mau")
    sample_prompts = [
        "Tìm nhà riêng tư dưới 3 tỷ ở quận 7, gần metro, phù hợp cho gia đình có con nhỏ",
        "Cần tìm căn hộ 2 phòng ngủ, giá dưới 4 tỷ, ở khu vực trung tâm thành phố",
        "Gợi ý khu vực phù hợp cho gia đình 4 người, có cha mẹ già",
    ]
    for idx, text in enumerate(sample_prompts, start=1):
        if st.button(f"Dung mau {idx}", use_container_width=True):
            st.session_state.prefill_prompt = text

    with st.expander("Advanced agent settings", expanded=False):
        selected_tool = st.selectbox(
            "Tool routing",
            options=["auto"] + ALLOWED_TOOLS,
            index=( ["auto"] + ALLOWED_TOOLS ).index(st.session_state.selected_tool)
            if st.session_state.selected_tool in (["auto"] + ALLOWED_TOOLS)
            else 0,
            key="selected_tool",
            help="Chọn auto để runtime tự route, hoặc force chọn tool cụ thể (dùng cho testing/debug)",
        )
        enable_llm_router_fallback = st.checkbox(
            "Enable LLM router fallback",
            value=st.session_state.enable_llm_router_fallback,
            key="enable_llm_router_fallback",
            help="Dùng LLM khi router không chắc chắn",
        )
        enable_llm_response_composer = st.checkbox(
            "Enable LLM response composer",
            value=st.session_state.enable_llm_response_composer,
            key="enable_llm_response_composer",
            help="Cho phép LLM tổng hợp câu trả lời.",
        )
        enable_llm_input_preprocessor = st.checkbox(
            "Enable LLM input preprocessor",
            value=st.session_state.enable_llm_input_preprocessor,
            key="enable_llm_input_preprocessor",
            help="Trich xuat slot/intent tu input nguoi dung truoc khi goi tool.",
        )
        enable_llm_input_primary = st.checkbox(
            "Allow LLM input as primary router",
            value=st.session_state.enable_llm_input_primary,
            key="enable_llm_input_primary",
            help="Dùng LLM lấy intent chính thay cho rule-based router. L",
        )
        enable_best_effort_response = st.checkbox(
            "Enable best effort response",
            value=st.session_state.enable_best_effort_response,
            key="enable_best_effort_response",
            help="Trả về gợi ý ngay cả khi không có kết quả chính xác",
        )
        st.session_state.debug_mode = st.checkbox(
            "Debug mode (show raw runtime JSON)",
            value=st.session_state.debug_mode,
        )

    if st.session_state.feedback_log:
        st.markdown("---")
        st.caption(f"Feedback da ghi nhan: {len(st.session_state.feedback_log)}")

    if st.button("New session"):
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.messages = []
        st.session_state.prefill_prompt = ""
        st.session_state.conversation_ended = False
        st.rerun()

if "top_k_last" not in st.session_state:
    st.session_state.top_k_last = top_k

for msg_index, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        raw = msg.get("raw")
        final_response = (raw or {}).get("final_response") if isinstance(raw, dict) else None
        if msg["role"] == "assistant" and isinstance(final_response, dict) and final_response:
            _render_sections(final_response, key_prefix=f"history_{msg_index}", raw=raw)
            _render_analytics_charts(raw, key_prefix=f"history_{msg_index}_analytics")
        else:
            st.markdown(msg["content"])

        if raw and st.session_state.debug_mode:
            with st.expander("Raw JSON"):
                st.code(json.dumps(raw, ensure_ascii=False, indent=2), language="json")

prompt = st.chat_input("Hãy nhập yêu cầu ở đây: ví dụ căn hộ dưới 4 tỷ ở quận 2, gần metro")
if not prompt and st.session_state.prefill_prompt:
    prompt = st.session_state.prefill_prompt
    st.session_state.prefill_prompt = ""

if prompt:
    if _is_end_conversation_command(prompt):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation_ended = True
        with st.chat_message("assistant"):
            st.info("Da ket thuc hoi thoai. Vui long gui feedback tong the ben duoi.")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Da ket thuc hoi thoai. Vui long gui feedback tong the ben duoi.",
            }
        )
    else:
        st.session_state.conversation_ended = False
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Dang xu ly..."):
                result = {}
                summary = "Khong co noi dung phan hoi"

                try:
                    metadata = {
                        "top_k": top_k,
                        "debug_mode": st.session_state.debug_mode,
                        "enable_llm_router_fallback": enable_llm_router_fallback,
                        "enable_llm_response_composer": enable_llm_response_composer,
                        "enable_llm_input_preprocessor": enable_llm_input_preprocessor,
                        "enable_llm_input_primary": enable_llm_input_primary,
                        "enable_best_effort_response": enable_best_effort_response,
                    }
                    if selected_tool != "auto":
                        metadata["selected_tool"] = selected_tool

                    result = st.session_state.chatbot.chat(
                        user_message=prompt,
                        session_id=st.session_state.session_id,
                        metadata=metadata,
                    )

                    raw_final_response = result.get("final_response")
                    if isinstance(raw_final_response, dict):
                        final_response = raw_final_response
                    elif isinstance(raw_final_response, str):
                        final_response = {"summary": raw_final_response}
                    else:
                        final_response = {}

                    summary = final_response.get("summary") or "Khong co noi dung phan hoi"
                    _render_sections(final_response, key_prefix="live", raw=result)
                    _render_analytics_charts(result, key_prefix="live_analytics")

                    meta = result.get("metadata") or {}
                    tool_history = meta.get("selected_tool_history") or []
                    if st.session_state.debug_mode and tool_history:
                        st.caption(
                            "ReAct trace: "
                            + " -> ".join(str(tool) for tool in tool_history)
                            + f" | turns={meta.get('react_turns', len(tool_history))}"
                        )
                except Exception as exc:
                    summary = f"He thong gap loi khi xu ly: {exc}"
                    st.error(summary)
                    result = {
                        "state": "failed",
                        "branch": "ui_runtime_error",
                        "final_response": {"summary": summary},
                        "metadata": {},
                    }

                if st.session_state.debug_mode:
                    with st.expander("Chi tiet runtime"):
                        st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")

        st.session_state.messages.append({"role": "assistant", "content": summary, "raw": result})

if st.session_state.conversation_ended:
    end_raw = _latest_assistant_raw()
    _render_end_conversation_feedback(end_raw)
