from __future__ import annotations

import re
from typing import Dict, List, Optional

from .vn_normalizer import normalize_text

PROPERTY_TYPE_ALIASES: Dict[str, List[str]] = {
    "Chung cư": ["chung cu", "can ho", "apartment", "can ho dich vu", "chung cu mini"],
    "Nhà phố": ["nha pho", "townhouse", "shophouse", "shop house", "nha lien ke"],
    "Nhà riêng": ["nha rieng", "villa", "biet thu", "house"],
    "Đất": ["dat", "dat nen", "lo dat", "land"],
}

TRANSACTION_TYPE_ALIASES: Dict[str, List[str]] = {
    "bán": ["ban", "mua", "buy", "sell"],
    "thuê": ["thue", "cho thue", "rent", "rental"],
}

CITY_ALIASES: Dict[str, List[str]] = {
    "Hồ Chí Minh": ["hcm", "tphcm", "tp hcm", "tp. hcm", "ho chi minh", "sai gon"],
    "Hà Nội": ["ha noi", "hn", "hanoi"],
    "Đà Nẵng": ["da nang", "dn", "danang"],
}

_DISTRICT_ALIASES: Dict[str, List[str]] = {
    "Thủ Đức": ["thu duc", "tp thu duc"],
    "Quận 1": ["quan 1", "q1", "q 1"],
    "Quận 2": ["quan 2", "q2", "q 2"],
    "Quận 3": ["quan 3", "q3", "q 3"],
    "Quận 4": ["quan 4", "q4", "q 4"],
    "Quận 5": ["quan 5", "q5", "q 5"],
    "Quận 6": ["quan 6", "q6", "q 6"],
    "Quận 7": ["quan 7", "q7", "q 7"],
    "Quận 8": ["quan 8", "q8", "q 8"],
    "Quận 9": ["quan 9", "q9", "q 9"],
    "Quận 10": ["quan 10", "q10", "q 10"],
    "Quận 11": ["quan 11", "q11", "q 11"],
    "Quận 12": ["quan 12", "q12", "q 12"],
    "Bình Thạnh": ["binh thanh"],
    "Gò Vấp": ["go vap"],
    "Tân Bình": ["tan binh"],
    "Tân Phú": ["tan phu"],
    "Bình Tân": ["binh tan"],
    "Phú Nhuận": ["phu nhuan"],
    "Hải Châu": ["hai chau"],
}

_WARD_ALIASES: Dict[str, List[str]] = {
    f"Phường {idx}": [
        f"phuong {idx}",
        f"p{idx}",
        f"p {idx}",
        f"p.{idx}",
        f"ward {idx}",
    ]
    for idx in range(1, 31)
}

_FIELD_MAP: Dict[str, Dict[str, List[str]]] = {
    "property_type": PROPERTY_TYPE_ALIASES,
    "transaction_type": TRANSACTION_TYPE_ALIASES,
    "city": CITY_ALIASES,
    "district": _DISTRICT_ALIASES,
    "ward": _WARD_ALIASES,
}

_CANONICAL_OVERRIDES: Dict[str, Dict[str, str]] = {
    "transaction_type": {
        "bán": "Bán",
        "thuê": "Cho thuê",
    },
}


def _materialize_aliases(canonical_value: str, aliases: List[str]) -> List[str]:
    out: List[str] = []
    for candidate in [canonical_value, *aliases, normalize_text(canonical_value)]:
        normalized_candidate = normalize_text(candidate)
        if normalized_candidate and normalized_candidate not in out:
            out.append(normalized_candidate)
    return out


def get_aliases(field: str, canonical_value: str) -> List[str]:
    field_name = str(field or "").strip()
    mapping = _FIELD_MAP.get(field_name)
    if not mapping or not canonical_value:
        return []

    if field_name == "ward":
        number_match = re.search(r"(\d{1,2})", normalize_text(canonical_value))
        if number_match:
            canonical_key = f"Phường {int(number_match.group(1))}"
            aliases = mapping.get(canonical_key, [])
            return _materialize_aliases(canonical_key, aliases)

    for canonical, aliases in mapping.items():
        if normalize_text(canonical) == normalize_text(canonical_value):
            return _materialize_aliases(canonical, aliases)
    return [normalize_text(canonical_value)]


def normalize_to_canonical(field: str, text: str) -> Optional[str]:
    field_name = str(field or "").strip()
    mapping = _FIELD_MAP.get(field_name)
    if not mapping:
        return None

    normalized_text = normalize_text(text)
    if not normalized_text:
        return None

    for canonical, aliases in mapping.items():
        alias_candidates = _materialize_aliases(canonical, aliases)
        for candidate in alias_candidates:
            if normalized_text == candidate:
                return _CANONICAL_OVERRIDES.get(field_name, {}).get(canonical, canonical)
            if f" {candidate} " in f" {normalized_text} ":
                return _CANONICAL_OVERRIDES.get(field_name, {}).get(canonical, canonical)

    if field_name == "ward":
        number_match = re.search(r"\b(?:phuong|p\.?|ward)\s*(\d{1,2})\b", normalized_text)
        if number_match:
            return f"Phường {int(number_match.group(1))}"
    return None
