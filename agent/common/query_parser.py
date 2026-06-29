from __future__ import annotations

import re
from typing import Any, Dict
from dataclasses import dataclass, field

from utils.alias_registry import normalize_to_canonical
from utils.vn_normalizer import normalize_query_pipeline


@dataclass
class QueryFilters:
    transaction_type: str | None = None
    property_type: str | None = None
    city: str | None = None
    district: str | None = None
    street: str | None = None
    max_price_vnd: int | None = None
    min_price_vnd: int | None = None
    min_area_m2: float | None = None
    max_area_m2: float | None = None
    min_bedrooms: int | None = None
    ward: str | None = None
    project: str | None = None
    min_bathrooms: int | None = None
    legal_status: str | None = None
    direction: str | None = None
    price_direction: str | None = None
    min_floors: int | None = None
    min_frontage_width_m: float | None = None
    min_road_access_width_m: float | None = None


@dataclass
class SoftPreferences:
    near_metro: bool = False
    near_school: bool = False
    quiet_area: bool = False
    family_friendly: bool = False
    many_amenities: bool = False
    wants_gym: bool = False
    wants_pool: bool = False
    near_entertainment: bool = False
    view: bool = False
    nearby_transport: bool = False
    nearby_landmarks: bool = False
    nearby_roads: bool = False


@dataclass
class UserProfile:
    family_size: int | None = None
    has_children: bool = False
    has_elderly: bool = False
    commuting_destination: str | None = None


@dataclass
class ParsedQuery:
    hard_filters: QueryFilters
    soft_preferences: SoftPreferences = field(default_factory=SoftPreferences)
    user_profile: UserProfile = field(default_factory=UserProfile)
    use_case: str = "general_search"
    matched_signals: list[str] = field(default_factory=list)
    missing_required_slots: list[str] = field(default_factory=list)
    clarification_question: str = ""
    parser_confidence: float = 0.0
    normalized_query: str = ""
    query: str = ""
    schema_version: str = "query_understanding.v2-lite"

_BUDGET_UNIT = (
    r"(?:"
    r"ty|tỷ|ti|tỉ|"
    r"trieu|triệu|tr|"
    r"billion|million"
    r"vnd|vnđ|dong|đồng|đ"
    r")"
)
_BUDGET_RANGE_WITH_UNIT_PATTERN = re.compile(
    r"(?:tu|from|khoang|khoảng|tam|tầm)?\s*(\d+(?:[\.,]\d+)?)\s*(?:-|den|đến|to|toi|tới)\s*(\d+(?:[\.,]\d+)?)\s*(" + _BUDGET_UNIT + r")\b",
    re.IGNORECASE,
)
_BUDGET_RANGE_MIXED_UNIT_PATTERN = re.compile(
    rf"(?:tu|from|khoang|khoảng|tam|tầm)?\s*"
    rf"(\d+(?:[\.,]\d+)?)\s*({_BUDGET_UNIT})\s*"
    rf"(?:-|den|đến|to|toi|tới)\s*"
    rf"(\d+(?:[\.,]\d+)?)\s*({_BUDGET_UNIT})\b",
    re.IGNORECASE,
)
# _BUDGET_MAX_PATTERN = re.compile(
#     r"(?:duoi|dưới|toi da|tối đa|khong qua|không quá|khong vuot|không vượt|under|<=)\s*(\d+(?:[\.,]\d+)?)\s*(ty|ti|trieu|triệu|tr|billion|million)\b",
#     re.IGNORECASE,
# )
_BUDGET_MAX_PATTERN = re.compile(
    rf"(?:duoi|dưới|toi\s*da|tối\s*đa|khong\s*qua|không\s*quá|"
    rf"khong\s*vuot|không\s*vượt|under|<=)\s*"
    rf"(\d+(?:[\.,]\d+)?)\s*({_BUDGET_UNIT})\b",
    re.IGNORECASE,
)

_BUDGET_MIN_PATTERN = re.compile(
    rf"(?:tren|trên|tu|từ|toi\s*thieu|tối\s*thiểu|at\s*least|>=)\s*"
    rf"(\d+(?:[\.,]\d+)?)\s*({_BUDGET_UNIT})\b",
    re.IGNORECASE,
)
_BUDGET_CONTEXT_SINGLE_PATTERN = re.compile(
    rf"(?:ngan\s*sach|ngân\s*sách|von|vốn|budget|tai\s*chinh|"
    rf"tài\s*chính|tam|tầm|khoang|khoảng)\s*"
    rf"(?:la|cỡ|co|khoang)?\s*"
    rf"(\d+(?:[\.,]\d+)?)\s*({_BUDGET_UNIT})\b",
    re.IGNORECASE,
)

_BUDGET_FALLBACK_SINGLE_PATTERN = re.compile(
    rf"\b(\d+(?:[\.,]\d+)?)\s*({_BUDGET_UNIT})\b",
    re.IGNORECASE,
)
_AREA_UNIT = r"(?:m2|m²|met\s*vuong|mét\s*vuông)"
_AREA_PATTERN = re.compile(
    rf"(\d+(?:[\.,]\d+)?)\s*{_AREA_UNIT}\b",
    re.IGNORECASE,
)
_AREA_RANGE_PATTERN = re.compile(
    rf"(?:tu|từ|from|khoang|khoảng)?\s*"
    rf"(\d+(?:[\.,]\d+)?)\s*"
    rf"(?:-|den|đến|to|toi|tới)\s*"
    rf"(\d+(?:[\.,]\d+)?)\s*{_AREA_UNIT}\b",
    re.IGNORECASE,
)
# _AREA_MAX_PATTERN = re.compile(
#     r"(?:duoi|dưới|toi\s*da|tối\s*đa|khong\s*qua|không\s*quá|under|<=)\s*(\d+(?:[\.,]\d+)?)\s*(?:m2|m²|met\s*vuong)?\b",
#     re.IGNORECASE,
# )
# _AREA_MIN_PATTERN = re.compile(
#     r"(?:tren|trên|tu|từ|toi\s*thieu|tối\s*thiểu|at\s*least|>=)\s*(\d+(?:[\.,]\d+)?)\s*(?:m2|m²|met\s*vuong)?\b",
#     re.IGNORECASE,
# )
_AREA_MAX_PATTERN = re.compile(
    rf"(?:duoi|dưới|toi\s*da|tối\s*đa|khong\s*qua|không\s*quá|under|<=)\s*"
    rf"(\d+(?:[\.,]\d+)?)\s*{_AREA_UNIT}\b",
    re.IGNORECASE,
)
_AREA_MIN_PATTERN = re.compile(
    rf"(?:tren|trên|tu|từ|toi\s*thieu|tối\s*thiểu|at\s*least|>=)\s*"
    rf"(\d+(?:[\.,]\d+)?)\s*{_AREA_UNIT}\b",
    re.IGNORECASE,
)

_BEDROOM_PATTERN = re.compile(r"(\d+)\s*(?:pn|phong\s*ngu)", re.IGNORECASE)
_BATHROOM_PATTERN = re.compile(r"(\d+)\s*(?:wc|phong\s*tam|ve\s*sinh|vs)", re.IGNORECASE)
_FLOOR_PATTERN = re.compile(r"(\d+)\s*(?:tang|lau)", re.IGNORECASE)
_WIDTH_PATTERN = re.compile(r"(\d+(?:[\.,]\d+)?)\s*m(?:et)?", re.IGNORECASE)

_SEARCH_CLARIFICATION_SLOT_LABELS = {
    "location": "khu vực ưu tiên",
    "budget": "ngân sách khoảng bao nhiêu",
    "transaction_type": "bạn muốn mua hay thuê",
    "property_type": "loại bất động sản",
}
_SEARCH_CLARIFICATION_THRESHOLD = 1


def _norm(text: str) -> str:
    return normalize_query_pipeline(text)


def _extract_property_type(qn: str) -> str | None:
    canonical = normalize_to_canonical("property_type", qn)
    if canonical:
        return canonical
    if re.search(r"\b(?:can\s*ho|chung\s*cu|apartment)\b", qn):
        return "Chung cư"
    if re.search(r"\b(?:nha\s*pho|townhouse|shophouse|shop\s*house|lien\s*ke)\b", qn):
        return "Nhà phố"
    if re.search(r"\b(?:dat\s*nen|lo\s*dat|dat)\b", qn):
        return "Đất"
    if re.search(r"\b(?:biet\s*thu|villa|nha\s*rieng|nha)\b", qn):
        return "Nhà riêng"
    return None


def _extract_transaction_type(qn: str) -> str | None:
    canonical = normalize_to_canonical("transaction_type", qn)
    if canonical:
        return canonical
    # Keep buy priority to avoid overriding "mua ... cho thue" to rent.
    if re.search(r"\bmua\b", qn) or re.search(r"\bban\b", qn) or re.search(r"\b(?:buy|sell)\b", qn):
        return "Bán"
    if re.search(r"\b(?:cho\s*thue|thue|rent)\b", qn):
        return "Cho thuê"
    return None


def _extract_city(qn: str) -> str | None:
    canonical = normalize_to_canonical("city", qn)
    if canonical:
        return canonical
    if any(k in qn for k in ["hcm", "ho chi minh", "sai gon", "tp hcm"]):
        return "Hồ Chí Minh"
    if "ha noi" in qn:
        return "Hà Nội"
    if "da nang" in qn:
        return "Đà Nẵng"
    return None


def _extract_district(qn: str) -> str | None:
    canonical = normalize_to_canonical("district", qn)
    if canonical:
        return canonical
    if re.search(r"\bthu\s*duc\b", qn):
        return "Thủ Đức"

    m = re.search(r"\b(?:quan|q\.?|district)\s*(\d{1,2})\b", qn)
    if m:
        return f"Quận {m.group(1)}"

    hm = re.search(r"\b(?:huyen|district)\s+([a-z\s]{2,40})\b", qn)
    if hm:
        tail = re.sub(r"\s+", " ", hm.group(1)).strip()
        county_named = {
            "cu chi": "Huyện Củ Chi",
            "hoc mon": "Huyện Hóc Môn",
            "nha be": "Huyện Nhà Bè",
            "can gio": "Huyện Cần Giờ",
            "binh chanh": "Huyện Bình Chánh",
        }
        if tail in county_named:
            return county_named[tail]

    named = {
        "binh thanh": "Bình Thạnh",
        "go vap": "Gò Vấp",
        "tan binh": "Tân Bình",
        "tan phu": "Tân Phú",
        "binh tan": "Bình Tân",
        "phu nhuan": "Phú Nhuận",
        "hai chau": "Hải Châu",
        "cu chi": "Huyện Củ Chi",
        "hoc mon": "Huyện Hóc Môn",
        "nha be": "Huyện Nhà Bè",
        "can gio": "Huyện Cần Giờ",
        "binh chanh": "Huyện Bình Chánh",
    }
    for token, canonical in named.items():
        if re.search(rf"\b{re.escape(token)}\b", qn):
            return canonical

    return None


def _extract_ward(qn: str) -> str | None:
    canonical = normalize_to_canonical("ward", qn)
    if canonical:
        return canonical
    m = re.search(r"\bp\.?\s*(\d{1,2})\b", qn)
    if m:
        return f"Phường {m.group(1)}"
    return None


def _extract_project(qn: str) -> str | None:
    m = re.search(r"\b(?:du an|khu do thi|vinhomes|masteri|sala)\s+([^,.;]{3,60})", qn)
    if not m:
        return None
    return m.group(0).strip()


def _extract_legal_status(qn: str) -> str | None:
    if "so hong" in qn:
        return "so hong"
    if "so do" in qn:
        return "so do"
    if "hdmb" in qn or "hop dong mua ban" in qn:
        return "hdmb"
    return None


def _extract_direction(qn: str) -> str | None:
    mapping = {
        "dong nam": "dong nam",
        "dong bac": "dong bac",
        "tay nam": "tay nam",
        "tay bac": "tay bac",
        "dong": "đông",
        "tay": "tây",
        "nam": "nam",
        "bac": "bắc",
    }
    if not re.search(r"\bhuong\b", qn):
        return None
    for key, value in mapping.items():
        if re.search(rf"\b{re.escape(key)}\b", qn):
            return value
    return None


def _extract_price_direction(qn: str) -> str | None:
    text = _norm(qn)
    if not text:
        return None

    cheaper_patterns = [
        r"\bre hon\b",
        r"\bthap hon\b",
        r"\bdap hon\b",
        r"\bgia thap hon\b",
        r"\blower\b",
        r"\bcheaper\b",
    ]
    expensive_patterns = [
        r"\bcao hon\b",
        r"\bdat hon\b",
        r"\bgia cao hon\b",
        r"\bhigher\b",
        r"\bmore expensive\b",
    ]

    if any(re.search(pattern, text) for pattern in cheaper_patterns):
        return "cheaper"
    if any(re.search(pattern, text) for pattern in expensive_patterns):
        return "expensive"
    return None


def _money_unit_multiplier(unit: str) -> int:
    normalized = _norm(unit)
    if normalized in {"ty", "ti", "billion"}:
        return 1_000_000_000
    if normalized in {"trieu", "tr", "million"}:
        return 1_000_000
    return 1_000_000_000


def _money_to_vnd(value_text: str, unit_text: str) -> int | None:
    try:
        amount = float(str(value_text).replace(",", ".").strip())
    except (TypeError, ValueError):
        return None
    multiplier = _money_unit_multiplier(unit_text)
    return int(amount * multiplier)


def _extract_budget_bounds(qn: str) -> tuple[int | None, int | None]:
    # 1) Most explicit: range with two units, e.g. "2 ty den 3500 trieu".
    mixed_match = _BUDGET_RANGE_MIXED_UNIT_PATTERN.search(qn)
    if mixed_match:
        min_vnd = _money_to_vnd(mixed_match.group(1), mixed_match.group(2))
        max_vnd = _money_to_vnd(mixed_match.group(3), mixed_match.group(4))
        if min_vnd is not None and max_vnd is not None:
            if min_vnd > max_vnd:
                min_vnd, max_vnd = max_vnd, min_vnd
            return min_vnd, max_vnd

    # 2) Range sharing the same unit, e.g. "3-5 ty".
    range_match = _BUDGET_RANGE_WITH_UNIT_PATTERN.search(qn)
    if range_match:
        min_vnd = _money_to_vnd(range_match.group(1), range_match.group(3))
        max_vnd = _money_to_vnd(range_match.group(2), range_match.group(3))
        if min_vnd is not None and max_vnd is not None:
            if min_vnd > max_vnd:
                min_vnd, max_vnd = max_vnd, min_vnd
            return min_vnd, max_vnd

    # 3) Explicit max / min constraints.
    min_vnd: int | None = None
    max_vnd: int | None = None

    max_match = _BUDGET_MAX_PATTERN.search(qn)
    if max_match:
        max_vnd = _money_to_vnd(max_match.group(1), max_match.group(2))

    min_match = _BUDGET_MIN_PATTERN.search(qn)
    if min_match:
        min_vnd = _money_to_vnd(min_match.group(1), min_match.group(2))

    if min_vnd is not None or max_vnd is not None:
        if min_vnd is not None and max_vnd is not None and min_vnd > max_vnd:
            min_vnd, max_vnd = max_vnd, min_vnd
        return min_vnd, max_vnd

    # 4) Budget-context single value, e.g. "ngan sach 4 ty" => treat as max budget.
    context_single = _BUDGET_CONTEXT_SINGLE_PATTERN.search(qn)
    if context_single:
        single_vnd = _money_to_vnd(context_single.group(1), context_single.group(2))
        return None, single_vnd

    # 5) Fallback first money value in sentence, conservative as max budget.
    fallback_single = _BUDGET_FALLBACK_SINGLE_PATTERN.search(qn)
    if fallback_single:
        single_vnd = _money_to_vnd(fallback_single.group(1), fallback_single.group(2))
        return None, single_vnd

    return None, None


def _is_reference_price_comparison_query(qn: str) -> bool:
    text = str(qn or "").strip().lower()
    if not text:
        return False

    has_market_context = any(token in text for token in ["mat bang", "gia thi truong", "gia khu vuc"])
    has_comparison_signal = any(
        token in text
        for token in ["cao hon", "thap hon", "dat hon", "re hon", "so voi", "so sanh", "hop ly khong"]
    )
    has_money_signal = bool(_BUDGET_FALLBACK_SINGLE_PATTERN.search(text))

    if not (has_market_context and has_comparison_signal and has_money_signal):
        return False

    # Keep explicit budget intents untouched (e.g. "ngan sach duoi 5 ty").
    has_explicit_budget_intent = any(
        token in text
        for token in [
            "ngan sach",
            "budget",
            "duoi",
            "toi da",
            "khong qua",
            "khong vuot",
            "under",
            "<=",
            "toi thieu",
            ">=",
        ]
    )
    return not has_explicit_budget_intent


def _extract_area_bounds(qn: str) -> tuple[float | None, float | None]:
    range_match = _AREA_RANGE_PATTERN.search(qn)
    if range_match:
        try:
            min_area = float(range_match.group(1).replace(",", "."))
            max_area = float(range_match.group(2).replace(",", "."))
        except (TypeError, ValueError):
            min_area, max_area = None, None
        if min_area is not None and max_area is not None:
            if min_area > max_area:
                min_area, max_area = max_area, min_area
            return min_area, max_area

    min_area: float | None = None
    max_area: float | None = None

    max_match = _AREA_MAX_PATTERN.search(qn)
    if max_match:
        try:
            max_area = float(max_match.group(1).replace(",", "."))
        except (TypeError, ValueError):
            max_area = None

    min_match = _AREA_MIN_PATTERN.search(qn)
    if min_match:
        try:
            min_area = float(min_match.group(1).replace(",", "."))
        except (TypeError, ValueError):
            min_area = None

    if min_area is not None or max_area is not None:
        if min_area is not None and max_area is not None and min_area > max_area:
            min_area, max_area = max_area, min_area
        return min_area, max_area

    area_match = _AREA_PATTERN.search(qn)
    if area_match:
        try:
            return float(area_match.group(1).replace(",", ".")), None
        except (TypeError, ValueError):
            return None, None

    return None, None


def _detect_soft_preferences(qn: str) -> SoftPreferences:
    return SoftPreferences(
        near_metro=bool(re.search(r"\b(metro|mrt|lrt|ga|tau dien|tram)\b", qn)),
        near_school=bool(re.search(r"\b(truong|school|mam non|hoc vien|tieu hoc|trung hoc|dai hoc)\b", qn)),
        quiet_area=bool(re.search(r"\b(yen tinh|it on|an tinh|quiet)\b", qn)),
        family_friendly=bool(re.search(r"\b(gia dinh|family|tre em|con nho)\b", qn)),
        #many_amenities=bool(re.search(r"\b(nhieu tien ich|day du tien ich|amenities)\b", qn)),
        wants_gym=bool(re.search(r"\b(gym|fitness|phong tap|the thao|sports)\b", qn)),
        wants_pool=bool(re.search(r"\b(ho boi|be boi|pool|swimming)\b", qn)),
        near_entertainment=bool(re.search(r"\b(vui choi|giai tri|entretaiment|entertainment|trung tam thuong mai|trung tam|rap chieu|rap chieu phim|rap|phố đi bộ|pho di bo|phố đi bộ|cafe|nha hang|nha hang|khu vui chơi|cong vien|công viên|bar|nightlife)\b", qn)),
        view=bool(re.search(r"\b(view|thoang|thoang mat|mat thoang|song view|park view|city view)\b", qn)),
        nearby_transport=bool(re.search(r"\b(gan ga|ga xe|tram xe|ben xe|xe bus|xe buyt|bus|tram)\b", qn)),
        nearby_landmarks=bool(re.search(r"\b(gan cho|gan truong|gan benh vien|gan cong vien|gan sieu thi|gan trung tam thuong mai|gan landmark|gan noi lam viec)\b", qn)),
        nearby_roads=bool(re.search(r"\b(gan duong lon|duong lon|mat tien duong|gan duong|gan pho|road access|near road)\b", qn)),
    )


def _extract_user_profile(qn: str) -> UserProfile:
    family_size = None
    m = re.search(r"\b(?:gia dinh|nha)\s*(\d+)\s*(?:nguoi|thanh vien)\b", qn)
    if m:
        try:
            family_size = int(m.group(1))
        except ValueError:
            family_size = None

    has_children = bool(re.search(r"\b(con nho|tre em| tre nho)\b", qn))
    has_elderly = bool(re.search(r"\b(nguoi gia|ong ba|cao tuoi|lon tuoi)\b", qn))

    commuting_destination = None
    commute_match = re.search(r"\b(?:di lam|lam viec|commute|di hoc)\s*(?:o|tai)?\s*([^,.;]{3,40})", qn)
    if commute_match:
        commuting_destination = commute_match.group(1).strip()

    return UserProfile(
        family_size=family_size,
        has_children=has_children,
        has_elderly=has_elderly,
        commuting_destination=commuting_destination,
    )


def _has_location_signal(hard: QueryFilters) -> bool:
    return any(
        bool(str(value or "").strip())
        for value in (hard.city, hard.district, hard.ward, hard.street, hard.project)
    )


def _has_use_case_signal(soft: SoftPreferences, user: UserProfile) -> bool:
    return any(
        [
            soft.near_metro,
            soft.near_school,
            soft.quiet_area,
            soft.family_friendly,
            soft.many_amenities,
            soft.wants_gym,
            soft.wants_pool,
            soft.view,
            soft.nearby_transport,
            soft.nearby_landmarks,
            soft.nearby_roads,
            user.has_children,
            user.has_elderly,
            bool(str(user.commuting_destination or "").strip()),
        ]
    )


def _infer_missing_required_slots(
    *,
    hard: QueryFilters,
    soft: SoftPreferences,
    user: UserProfile,
    use_case: str,
) -> list[str]:
    missing: list[str] = []

    # For market/consultative flows, avoid hard-slot clarification.
    if use_case in {"market_overview", "suggest_area", "compare_listings", "explain_listing"}:
        return missing

    if not _has_location_signal(hard):
        missing.append("location")

    has_budget = hard.min_price_vnd is not None or hard.max_price_vnd is not None
    has_transaction = hard.transaction_type is not None
    has_property = hard.property_type is not None

    # Require transaction/property/budget only when user intent is still very broad.
    if not has_transaction and not has_property and not has_budget:
        missing.extend(["transaction_type", "property_type", "budget"])

    return missing


def _build_clarification_question(missing_slots: list[str]) -> str:
    if not missing_slots:
        return ""

    priority = ["location", "budget", "transaction_type", "property_type"]
    labels = [
        _SEARCH_CLARIFICATION_SLOT_LABELS.get(slot, slot.replace("_", " "))
        for slot in priority
        if slot in missing_slots and slot in _SEARCH_CLARIFICATION_SLOT_LABELS
    ]
    if not labels:
        return ""

    top_labels = labels[:2]
    if len(top_labels) == 1:
        return f"Bạn có thể cho mình biết {top_labels[0]} không?"
    if len(top_labels) == 2:
        return f"Bạn có thể cho mình biết {top_labels[0]} và {top_labels[1]} không?"
    return f"Bạn có thể cho mình biết {top_labels[0]} không?"


def infer_search_clarification(parsed: ParsedQuery) -> tuple[list[str], str]:
    missing_slots = _infer_missing_required_slots(
        hard=parsed.hard_filters,
        soft=parsed.soft_preferences,
        user=parsed.user_profile,
        use_case=parsed.use_case,
    )
    clarification_question = ""
    if len(missing_slots) >= _SEARCH_CLARIFICATION_THRESHOLD:
        clarification_question = _build_clarification_question(missing_slots)
    return missing_slots, clarification_question


def _infer_use_case(qn: str) -> str:
    market_overview_tokens = [
        "gia thi truong",
        "mat bang gia",
        "gia khu vuc",
        "gia trung binh",
        "xu huong gia",
        "tiem nang tang gia",
        "chua biet gia",
        "khong biet gia",
        "gia nhu nao",
        "gia sao",
    ]
    if any(token in qn for token in market_overview_tokens):
        return "market_overview"

    if "mat bang" in qn and any(token in qn for token in ["gia", "thi truong", "cao hon", "thap hon", "so voi"]):
        return "market_overview"

    if re.search(r"\b(so sanh|compare|vs)\b", qn):
        return "compare_listings"
    if re.search(r"\b(giai thich|phan tich|explain)\b", qn):
        return "explain_listing"
    center_location_tokens = [
        "gan trung tam",
        "gần trung tâm",
        "trung tam",
        "trung tâm",
        "sat trung tam",
        "sát trung tâm",
        "cbd",
    ]
    has_center_intent = any(token in qn for token in center_location_tokens)
    has_area_context = any(
        token in qn
        for token in [
            "ty",
            "ti",
            "trieu",
            "can ho",
            "chung cu",
            "nha pho",
            "dat nen",
            "nha rieng",
            "bat dong san",
        ]
    )
    if has_center_intent and has_area_context:
        return "suggest_area"
    if re.search(r"\b(khu nao|khu vuc nao|nhung khu vuc nao|o khu nao|o khu vuc nao|nen mua o dau|suggest area|khu vuc phu hop|khu nao phu hop)\b", qn) and any(
        token in qn
        for token in [
            "ty",
            "ti",
            "trieu",
            "can ho",
            "chung cu",
            "nha pho",
            "dat nen",
            "nha rieng",
            "bat dong san",
        ]
    ):
        return "suggest_area"
    return "general_search"


def parse_user_query(query: str) -> ParsedQuery:
    text = str(query or "").strip()
    qn = normalize_query_pipeline(text)
    hard = QueryFilters()

    hard.property_type = _extract_property_type(qn)

    hard.transaction_type = _extract_transaction_type(qn)

    hard.city = _extract_city(qn)

    hard.district = _extract_district(qn)
    hard.ward = _extract_ward(qn)
    hard.project = _extract_project(qn)

    min_budget_vnd, max_budget_vnd = _extract_budget_bounds(qn)
    if _is_reference_price_comparison_query(qn):
        # Numbers like "5,86 ty ~87,72 trieu/m2" are reference values to compare
        # against market baseline, not filtering budgets.
        min_budget_vnd, max_budget_vnd = None, None
    hard.min_price_vnd = min_budget_vnd
    hard.max_price_vnd = max_budget_vnd

    min_area_m2, max_area_m2 = _extract_area_bounds(qn)
    hard.min_area_m2 = min_area_m2
    hard.max_area_m2 = max_area_m2

    bedroom_match = _BEDROOM_PATTERN.search(qn)
    if bedroom_match:
        try:
            hard.min_bedrooms = int(bedroom_match.group(1))
        except ValueError:
            hard.min_bedrooms = None

    bathroom_match = _BATHROOM_PATTERN.search(qn)
    if bathroom_match:
        try:
            hard.min_bathrooms = int(bathroom_match.group(1))
        except ValueError:
            hard.min_bathrooms = None

    floor_match = _FLOOR_PATTERN.search(qn)
    if floor_match:
        try:
            hard.min_floors = int(floor_match.group(1))
        except ValueError:
            hard.min_floors = None

    hard.legal_status = _extract_legal_status(qn)
    hard.direction = _extract_direction(qn)
    hard.price_direction = _extract_price_direction(qn)

    if "mat tien" in qn:
        width_match = _WIDTH_PATTERN.search(qn)
        if width_match:
            try:
                hard.min_frontage_width_m = float(width_match.group(1).replace(",", "."))
            except ValueError:
                hard.min_frontage_width_m = None

    if "duong" in qn:
        width_match = _WIDTH_PATTERN.search(qn)
        if width_match:
            try:
                hard.min_road_access_width_m = float(width_match.group(1).replace(",", "."))
            except ValueError:
                hard.min_road_access_width_m = None

    soft = _detect_soft_preferences(qn)
    user = _extract_user_profile(qn)
    matched_signals: list[str] = []
    if hard.max_price_vnd is not None:
        matched_signals.append("budget")
    if hard.district or hard.city:
        matched_signals.append("location")
    if hard.min_bedrooms is not None:
        matched_signals.append("bedrooms")
    if soft.family_friendly or user.has_children or user.has_elderly:
        matched_signals.append("family")

    use_case = _infer_use_case(qn)
    parsed_preview = ParsedQuery(
        hard_filters=hard,
        soft_preferences=soft,
        user_profile=user,
        use_case=use_case,
        matched_signals=matched_signals,
        normalized_query=qn,
        query=text,
        schema_version="query_understanding.v3-canonical",
    )
    missing_required_slots, clarification_question = infer_search_clarification(parsed_preview)

    confidence = 0.35 + (0.15 * len(matched_signals))
    if len(missing_required_slots) >= _SEARCH_CLARIFICATION_THRESHOLD:
        confidence = max(0.2, confidence - 0.1 * len(missing_required_slots))
    confidence = min(0.95, confidence)

    return ParsedQuery(
        hard_filters=hard,
        soft_preferences=soft,
        user_profile=user,
        use_case=use_case,
        matched_signals=matched_signals,
        missing_required_slots=missing_required_slots,
        clarification_question=clarification_question,
        parser_confidence=confidence,
        normalized_query=qn,
        query=text,
        schema_version="query_understanding.v3-canonical",
    )


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ".").strip())
    except (TypeError, ValueError):
        return None


def canonicalize_llm_slots(raw_slots: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Canonicalize LLM-understood slots using deterministic parser rules.
    Returns keys aligned with QueryFilters fields for retrieval filters.
    """
    slots = raw_slots if isinstance(raw_slots, dict) else {}
    canonical: Dict[str, Any] = {}

    property_raw = _norm(str(slots.get("property_type") or ""))
    city_raw = _norm(str(slots.get("city") or ""))
    district_raw = _norm(str(slots.get("district") or ""))
    ward_raw = _norm(str(slots.get("ward") or ""))
    tx_raw = _norm(str(slots.get("transaction_type") or ""))
    legal_raw = _norm(str(slots.get("legal_status") or ""))
    direction_raw = _norm(str(slots.get("direction") or ""))
    price_direction_raw = _norm(str(slots.get("price_direction") or ""))

    property_type = _extract_property_type(property_raw) if property_raw else None
    city = _extract_city(city_raw) if city_raw else None
    district = _extract_district(district_raw) if district_raw else None
    ward = _extract_ward(ward_raw) if ward_raw else None
    transaction_type = _extract_transaction_type(tx_raw) if tx_raw else None
    legal_status = _extract_legal_status(legal_raw) if legal_raw else None

    direction = None
    if direction_raw:
        direction = _extract_direction(f"huong {direction_raw}")

    price_direction = None
    if price_direction_raw:
        price_direction = _extract_price_direction(price_direction_raw)

    budget_max_vnd = slots.get("budget_max_vnd", slots.get("max_price_vnd"))
    min_area_m2 = slots.get("min_area_m2")
    max_area_m2 = slots.get("max_area_m2")
    min_bedrooms = slots.get("min_bedrooms")
    min_bathrooms = slots.get("min_bathrooms")
    project = str(slots.get("project") or "").strip() or None

    if transaction_type:
        canonical["transaction_type"] = transaction_type
    if property_type:
        canonical["property_type"] = property_type
    if city:
        canonical["city"] = city
    if district:
        canonical["district"] = district
    if ward:
        canonical["ward"] = ward
    if project:
        canonical["project"] = project
    if legal_status:
        canonical["legal_status"] = legal_status
    if direction:
        canonical["direction"] = direction
    if price_direction:
        canonical["price_direction"] = price_direction

    max_price_vnd = _to_int(budget_max_vnd)
    if max_price_vnd is not None:
        canonical["max_price_vnd"] = max_price_vnd

    area_value = _to_float(min_area_m2)
    if area_value is not None:
        canonical["min_area_m2"] = area_value

    max_area_value = _to_float(max_area_m2)
    if max_area_value is not None:
        canonical["max_area_m2"] = max_area_value

    bedrooms_value = _to_int(min_bedrooms)
    if bedrooms_value is not None:
        canonical["min_bedrooms"] = bedrooms_value

    bathrooms_value = _to_int(min_bathrooms)
    if bathrooms_value is not None:
        canonical["min_bathrooms"] = bathrooms_value

    return canonical