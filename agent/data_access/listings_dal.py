from __future__ import annotations

import math
import re
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psycopg

from agent.common import get_logger, load_config, parse_user_query
from agent.retrieval import RetrievalDbGateway, RetrievalService
from agent.retrieval.retrieval_service import QueryFilters as RetrievalQueryFilters
from utils.vn_normalizer import normalize_query_pipeline


@dataclass
class SearchResult:
    items: List[Dict[str, Any]]
    retrieval_stats: Dict[str, Any]


@dataclass
class ExplainResult:
    found: bool
    listing: Dict[str, Any]
    message: str
    analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompareResult:
    found: bool
    listing_a: Dict[str, Any]
    listing_b: Dict[str, Any]
    recommendation: Dict[str, Any]
    message: str


@dataclass
class SuggestAreaResult:
    need_clarification: bool
    missing_fields: List[str]
    area_recommendations: List[Dict[str, Any]]
    summary: str
    next_clarification_prompt: str | None = None
    suggested_next_tool: str | None = None
    next_user_action: str | None = None


class AgentListingDataAccess:


    SOURCE_CANDIDATES: Sequence[str] = ("agent_listing_search_v1", "listings")
    # CHANGED (2026-06-27): Remove inventory_score weight - don't rank by listing count
    # Rankings now focus on: location, persona, budget fit, property type only
    _DEFAULT_AREA_WEIGHTS: Dict[str, float] = {
        "location_score": 0.37,      # 0.35 -> 0.37 (reallocate from inventory 0.05)
        "persona_score": 0.26,       # 0.25 -> 0.26 (reallocate from inventory 0.05)
        "budget_fit": 0.21,          # 0.20 -> 0.21 (reallocate from inventory 0.05)
        "property_type_fit": 0.16,   # 0.15 -> 0.16 (reallocate from inventory 0.05)
        "inventory_score": 0.00,     # WAS 0.05 -> NOW 0 (remove inventory bias from ranking)
    }
    _CENTER_PRIORITY_AREA_WEIGHTS: Dict[str, float] = {
        "location_score": 0.47,      # 0.45 -> 0.47
        "persona_score": 0.26,       # 0.25 -> 0.26
        "budget_fit": 0.16,          # 0.15 -> 0.16
        "property_type_fit": 0.11,   # 0.10 -> 0.11
        "inventory_score": 0.00,     # WAS 0.05 -> NOW 0
    }
    _COMMUTE_PRIORITY_AREA_WEIGHTS: Dict[str, float] = {
        "location_score": 0.42,      # 0.40 -> 0.42
        "persona_score": 0.21,       # 0.20 -> 0.21
        "budget_fit": 0.21,          # 0.20 -> 0.21
        "property_type_fit": 0.16,   # 0.15 -> 0.16
        "inventory_score": 0.00,     # WAS 0.05 -> NOW 0
    }
    _HCM_CENTRALITY_MAP: Dict[str, float] = {
        "quan 1": 1.00,
        "quan 3": 0.98,
        "thu thiem": 0.93,
        "an khanh": 0.88,
        "thao dien": 0.88,
        "binh thanh": 0.90,
        "phu nhuan": 0.86,
        "quan 4": 0.84,
        "quan 10": 0.78,
        "tan binh": 0.75,
        "quan 5": 0.73,
        "quan 2": 0.82,
        "quan 7": 0.56,
        "thu duc": 0.52,
        "quan 11": 0.64,
        "go vap": 0.52,
        "tan phu": 0.48,
        "binh tan": 0.38,
        "quan 9": 0.30,
        "nha be": 0.18,
        "hoc mon": 0.14,
        "cu chi": 0.10,
        "can gio": 0.05,
    }
    _HANOI_CENTRALITY_MAP: Dict[str, float] = {
        "hoan kiem": 1.00,
        "ba dinh": 0.95,
        "hai ba trung": 0.92,
        "dong da": 0.88,
        "tay ho": 0.82,
        "cau giay": 0.80,
        "thanh xuan": 0.72,
        "nam tu liem": 0.68,
        "bac tu liem": 0.58,
        "long bien": 0.55,
        "hoang mai": 0.53,
        "ha dong": 0.42,
    }
    _BINH_DUONG_CENTRALITY_MAP: Dict[str, float] = {
        "thu dau mot": 1.00,
        "thanh pho thu dau mot": 1.00,
        "thuan an": 0.92,
        "thanh pho thuan an": 0.92,
        "di an": 0.90,
        "thanh pho di an": 0.90,
        "ben cat": 0.72,
        "thanh pho ben cat": 0.72,
        "tan uyen": 0.64,
        "thanh pho tan uyen": 0.64,
        "bau bang": 0.42,
        "huyen bau bang": 0.42,
        "bac tan uyen": 0.38,
        "huyen bac tan uyen": 0.38,
        "dau tieng": 0.36,
        "huyen dau tieng": 0.36,
        "phu giao": 0.34,
        "huyen phu giao": 0.34,
    }
    _SEARCH_CENTER_PRIORITY_BOOST_WEIGHT: float = 0.12
    _CONTRACT_COLUMNS: Sequence[str] = (
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

    _OPTIONAL_DETAIL_COLUMNS: Sequence[str] = (
        "street",
        "structure",
        "interior",
        "access",
        "location_quality",
        "neighborhood_quality",
        "view",
        "suitable_for",
        "amenities_building",
        "amenities_area",
        "nearby_landmarks",
        "nearby_transport",
        "nearby_roads",
    )

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.logger = get_logger("agent_listing_dal")
        self.config_path = self._resolve_config_path(config_path)
        self.db_config = self._load_db_config(self.config_path)
        self.retrieval_service = RetrievalService(db_gateway=RetrievalDbGateway(config_path=self.config_path))
        self._table_columns_cache: Dict[str, set[str]] = {}

    def _resolve_config_path(self, config_path: Optional[str]) -> str:
        if config_path:
            return str(config_path)
        return str(Path(__file__).resolve().parents[2] / "CONFIG" / "global.yaml")

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
    def _tokenize(text: str) -> List[str]:
        return [tok for tok in re.findall(r"\w+", (text or "").lower()) if len(tok) > 1][:8]

    @staticmethod
    def _expand_property_type_aliases(property_type: str | None) -> List[str]:
        key = str(property_type or "").strip().lower()
        if not key:
            return []

        aliases: Dict[str, List[str]] = {
            "can ho": ["can ho", "căn hộ", "chung cư", "chung cu", "apartment"],
            "nha pho": ["nha pho", "nhà phố", "shophouse", "shop house", "townhouse", "liền kề", "lien ke"],
            "dat": ["dat", "đất", "đất nền", "dat nen", "đất nền dự án", "dat nen du an"],
            "nha": ["nha", "nhà", "nhà riêng", "nha rieng", "biệt thự", "biet thu", "villa"],
        }
        return aliases.get(key, [key])

    @staticmethod
    def _expand_transaction_type_aliases(transaction_type: str | None) -> List[str]:
        key = str(transaction_type or "").strip().lower()
        if not key:
            return []
        aliases: Dict[str, List[str]] = {
            "ban": ["ban", "bán", "mua bán", "mua ban"],
            "thue": ["thue", "thuê", "cho thuê", "cho thue"],
        }
        return aliases.get(key, [key])

    @staticmethod
    def _expand_district_aliases(district: str | None) -> List[str]:
        raw_value = str(district or "").strip().lower()
        value = normalize_query_pipeline(raw_value)
        if not value:
            return []

        m = re.search(r"\bquan\s*(\d{1,2})\b", value)
        if m:
            num = m.group(1)
            return [f"quan {num}", f"quận {num}", f"q{num}", f"q {num}"]

        tp = re.search(r"\b(?:thanh pho|tp)\s+([a-z\s]{2,40})\b", value)
        if tp:
            city_name = re.sub(r"\s+", " ", tp.group(1)).strip()
            bd_city_named = {
                "thu dau mot": ["thu dau mot", "thủ dầu một", "thanh pho thu dau mot", "thành phố thủ dầu một", "tp thu dau mot", "tp thủ dầu một"],
                "thuan an": ["thuan an", "thuận an", "thanh pho thuan an", "thành phố thuận an", "tp thuan an", "tp thuận an"],
                "di an": ["di an", "dĩ an", "thanh pho di an", "thành phố dĩ an", "tp di an", "tp dĩ an"],
                "tan uyen": ["tan uyen", "tân uyên", "thanh pho tan uyen", "thành phố tân uyên", "tp tan uyen", "tp tân uyên"],
                "ben cat": ["ben cat", "bến cát", "thanh pho ben cat", "thành phố bến cát", "tp ben cat", "tp bến cát"],
            }
            if city_name in bd_city_named:
                return bd_city_named[city_name]

        hm = re.search(r"\bhuyen\s+([a-z\s]{2,40})\b", value)
        if hm:
            county_name = re.sub(r"\s+", " ", hm.group(1)).strip()
            bd_county_named = {
                "bau bang": ["bau bang", "bàu bàng", "huyen bau bang", "huyện bàu bàng"],
                "dau tieng": ["dau tieng", "dầu tiếng", "huyen dau tieng", "huyện dầu tiếng"],
                "phu giao": ["phu giao", "phú giáo", "huyen phu giao", "huyện phú giáo"],
                "bac tan uyen": ["bac tan uyen", "bắc tân uyên", "huyen bac tan uyen", "huyện bắc tân uyên"],
            }
            if county_name in bd_county_named:
                return bd_county_named[county_name]
            if county_name:
                return [
                    f"huyen {county_name}",
                    f"huyện {county_name}",
                    county_name,
                ]

        if value == "thu duc":
            return ["thu duc", "thủ đức", "tp thu duc", "tp thủ đức"]

        named = {
            "binh thanh": ["binh thanh", "bình thạnh"],
            "go vap": ["go vap", "gò vấp"],
            "tan binh": ["tan binh", "tân bình"],
            "tan phu": ["tan phu", "tân phú"],
            "binh tan": ["binh tan", "bình tân"],
            "phu nhuan": ["phu nhuan", "phú nhuận"],
            "hai chau": ["hai chau", "hải châu"],
            "cu chi": ["cu chi", "củ chi", "huyen cu chi", "huyện củ chi"],
            "hoc mon": ["hoc mon", "hóc môn", "huyen hoc mon", "huyện hóc môn"],
            "nha be": ["nha be", "nhà bè", "huyen nha be", "huyện nhà bè"],
            "can gio": ["can gio", "cần giờ", "huyen can gio", "huyện cần giờ"],
            "binh chanh": ["binh chanh", "bình chánh", "huyen binh chanh", "huyện bình chánh"],
            "thu dau mot": ["thu dau mot", "thủ dầu một", "thanh pho thu dau mot", "thành phố thủ dầu một", "tp thu dau mot", "tp thủ dầu một"],
            "thuan an": ["thuan an", "thuận an", "thanh pho thuan an", "thành phố thuận an", "tp thuan an", "tp thuận an"],
            "di an": ["di an", "dĩ an", "thanh pho di an", "thành phố dĩ an", "tp di an", "tp dĩ an"],
            "tan uyen": ["tan uyen", "tân uyên", "thanh pho tan uyen", "thành phố tân uyên", "tp tan uyen", "tp tân uyên"],
            "ben cat": ["ben cat", "bến cát", "thanh pho ben cat", "thành phố bến cát", "tp ben cat", "tp bến cát"],
            "bau bang": ["bau bang", "bàu bàng", "huyen bau bang", "huyện bàu bàng"],
            "dau tieng": ["dau tieng", "dầu tiếng", "huyen dau tieng", "huyện dầu tiếng"],
            "phu giao": ["phu giao", "phú giáo", "huyen phu giao", "huyện phú giáo"],
            "bac tan uyen": ["bac tan uyen", "bắc tân uyên", "huyen bac tan uyen", "huyện bắc tân uyên"],
        }
        return named.get(value, [value])

    @staticmethod
    def _extract_districts_from_normalized_query(qn: str) -> List[str]:
        text = str(qn or "").strip().lower()
        if not text:
            return []

        districts: List[str] = []

        for num in re.findall(r"\b(?:quan|q\.?|district)\s*(\d{1,2})\b", text):
            candidate = f"Quận {num}"
            if candidate not in districts:
                districts.append(candidate)

        for county_name in re.findall(r"\bhuyen\s+([a-z\s]{2,40})\b", text):
            cleaned = re.sub(r"\s+", " ", county_name).strip()
            county_named = {
                "cu chi": "Huyện Củ Chi",
                "hoc mon": "Huyện Hóc Môn",
                "nha be": "Huyện Nhà Bè",
                "can gio": "Huyện Cần Giờ",
                "binh chanh": "Huyện Bình Chánh",
                "bau bang": "Huyện Bàu Bàng",
                "dau tieng": "Huyện Dầu Tiếng",
                "phu giao": "Huyện Phú Giáo",
                "bac tan uyen": "Huyện Bắc Tân Uyên",
            }
            canonical = county_named.get(cleaned)
            if canonical and canonical not in districts:
                districts.append(canonical)

        for city_name in re.findall(r"\b(?:thanh pho|tp)\s+([a-z\s]{2,40})\b", text):
            cleaned = re.sub(r"\s+", " ", city_name).strip()
            city_named = {
                "thu dau mot": "Thành phố Thủ Dầu Một",
                "thuan an": "Thành phố Thuận An",
                "di an": "Thành phố Dĩ An",
                "tan uyen": "Thành phố Tân Uyên",
                "ben cat": "Thành phố Bến Cát",
            }
            canonical = city_named.get(cleaned)
            if canonical and canonical not in districts:
                districts.append(canonical)


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
            "thu dau mot": "Thành phố Thủ Dầu Một",
            "thuan an": "Thành phố Thuận An",
            "di an": "Thành phố Dĩ An",
            "tan uyen": "Thành phố Tân Uyên",
            "ben cat": "Thành phố Bến Cát",
            "bau bang": "Huyện Bàu Bàng",
            "dau tieng": "Huyện Dầu Tiếng",
            "phu giao": "Huyện Phú Giáo",
            "bac tan uyen": "Huyện Bắc Tân Uyên",
        }
        for token, canonical in named.items():
            if re.search(rf"\b{re.escape(token)}\b", text) and canonical not in districts:
                districts.append(canonical)

        return districts

    @staticmethod
    def _expand_city_aliases(city: str) -> List[str]:
        value = str(city or "").strip().lower()
        if not value:
            return []

        canonical = value.replace("tp.", "tp").replace("thanh pho", "tp").strip()
        mapping = {
            "ha noi": ["ha noi", "hanoi", "hà nội", "tp ha noi", "tp hà nội"],
            "hanoi": ["ha noi", "hanoi", "hà nội", "tp ha noi", "tp hà nội"],
            "hà nội": ["ha noi", "hanoi", "hà nội", "tp ha noi", "tp hà nội"],
            "tp ha noi": ["ha noi", "hanoi", "hà nội", "tp ha noi", "tp hà nội"],
            "ho chi minh": [
                "ho chi minh",
                "hcm",
                "hcmc",
                "tp ho chi minh",
                "tp. ho chi minh",
                "thành phố hồ chí minh",
                "hồ chí minh",
                "sai gon",
                "sài gòn",
            ],
            "hcm": [
                "ho chi minh",
                "hcm",
                "hcmc",
                "tp ho chi minh",
                "tp. ho chi minh",
                "thành phố hồ chí minh",
                "hồ chí minh",
                "sai gon",
                "sài gòn",
            ],
            "da nang": ["da nang", "đà nẵng", "tp da nang", "tp đà nẵng"],
            "đà nẵng": ["da nang", "đà nẵng", "tp da nang", "tp đà nẵng"],
            "tp da nang": ["da nang", "đà nẵng", "tp da nang", "tp đà nẵng"],
            "binh duong": ["binh duong", "bình dương", "tinh binh duong", "tỉnh bình dương", "tp binh duong", "tp. binh duong"],
            "tinh binh duong": ["binh duong", "bình dương", "tinh binh duong", "tỉnh bình dương", "tp binh duong", "tp. binh duong"],
            "tp binh duong": ["binh duong", "bình dương", "tinh binh duong", "tỉnh bình dương", "tp binh duong", "tp. binh duong"],
        }
        return mapping.get(canonical, [city])

    @staticmethod
    def _detect_analytics_metric_from_query(qn: str) -> str:
        text = str(qn or "").strip().lower()
        if not text:
            return "default"

        # Compare average price between areas, e.g. "so sanh gia trung binh giua quan 7 va binh thanh".
        if any(token in text for token in ["so sanh", "vs", "giua"]) and any(
            token in text for token in ["gia trung binh", "trung binh", "average"]
        ):
            return "district_compare_avg_price"

        type_grouping_hints = [
            "theo dang listing",
            "theo loai listing",
            "theo dang tin",
            "theo loai tin",
            "theo loai",
            "theo dang",
            "property type",
            "loai hinh",
        ]
        if any(hint in text for hint in type_grouping_hints) and any(token in text for token in ["gia", "trung binh", "listing", "tin"]):
            return "property_type_grouping"

        # Check for price/m2 average (Issue 8) - require explicit m2 marker.
        if any(token in text for token in ["m2", "/m2", "tren m2", "moi m2"]) and any(
            token in text for token in ["gia", "chi phi", "tien"]
        ):
            if "trung binh" in text or "average" in text:
                return "avg_price_per_m2"
        
        # Check for max price/m2 (current logic)
        if any(token in text for token in ["gia/m2", "gia tren m2", "gia moi m2", "gia m2"]) and "trung" not in text:
            return "max_price_per_m2"
        
        # Check for ranking/top-K queries (Issues 4 & 5)
        # NOTE: "bao nhieu" (how many) is a count query, not a ranking query.
        ranking_intent = any(token in text for token in ["top", "hang", "nhieu nhat", "dat nhat", "re nhat", "cao nhat"])
        if re.search(r"\bnhieu\b.{0,30}\bnhat\b", text):
            ranking_intent = True
        if not ("bao nhieu" in text) and ranking_intent:
            if any(token in text for token in ["phuong", "xa", "ward", "commune"]):
                return "ward_ranking"
            if any(token in text for token in ["quan", "district", "khu vuc", "noi nao", "khu nao"]):
                return "district_ranking"
        
        # Check for property type breakdown (Issue 12)
        if any(token in text for token in ["phan bo", "loai bat dong san"]) and "loai" in text:
            return "property_type_grouping"

        ratio_tokens = ["so sanh", "ratio", "ty le", "ti le", "so voi", "phan tram", "giua", "vs"]
        property_type_groups = {
            "can_ho": ["can ho", "chung cu", "apartment"],
            "nha_pho": ["nha pho", "townhouse", "shophouse"],
            "dat": ["dat", "dat nen"],
            "nha_rieng": ["nha rieng", "biet thu", "villa", "nha"],
        }
        detected_groups = 0
        for aliases in property_type_groups.values():
            if any(re.search(rf"\b{re.escape(alias)}\b", text) for alias in aliases):
                detected_groups += 1
        if any(token in text for token in ratio_tokens) and detected_groups >= 2:
            return "property_type_grouping"

        if "can ho" in text and "nha pho" in text and (
            "so sanh" in text
            or "ratio" in text
            or "ty le" in text
            or "ti le" in text
            or "so voi" in text
            or "phan tram" in text
        ):
            return "property_type_grouping"
        
        # Check for average area
        if "dien tich trung binh" in text or "dien tich trung" in text:
            return "avg_area_m2"
        
        return "default"

    @staticmethod
    def _detect_ranking_metric_from_query(qn: str) -> tuple[str, str]:
        text = str(qn or "").strip().lower()
        if not text:
            return ("count", "desc")

        mentions_price = any(token in text for token in ["gia", "gia ban", "muc gia"])
        mentions_avg = any(token in text for token in ["trung binh", "tb", "average"])
        asks_highest = any(token in text for token in ["cao nhat", "dat nhat", "max"])
        asks_lowest = any(token in text for token in ["thap nhat", "re nhat", "min"])

        if mentions_price and mentions_avg and (asks_highest or asks_lowest):
            return ("avg_price_vnd", "asc" if asks_lowest else "desc")

        return ("count", "desc")

    @staticmethod
    def _append_soft_preference_filters(where: List[str], params: List[Any], parsed: Any) -> None:
        soft = getattr(parsed, "soft_preferences", None)
        if soft is None:
            return

        if bool(getattr(soft, "near_metro", False)):
            where.append(
                "(" 
                "search_document ILIKE %s OR "
                "nearby_transport ILIKE %s OR "
                "nearby_landmarks ILIKE %s"
                ")"
            )
            params.extend(["%metro%", "%metro%", "%metro%"])

        if bool(getattr(soft, "wants_gym", False)):
            where.append(
                "(" 
                "search_document ILIKE %s OR "
                "search_document ILIKE %s OR "
                "search_document ILIKE %s OR "
                "amenities_building ILIKE %s OR "
                "amenities_building ILIKE %s OR "
                "amenities_area ILIKE %s OR "
                "amenities_area ILIKE %s"
                ")"
            )
            params.extend(["%gym%", "%fitness%", "%phong tap%", "%gym%", "%fitness%", "%gym%", "%fitness%"])

        if bool(getattr(soft, "wants_pool", False)):
            where.append(
                "(" 
                "search_document ILIKE %s OR "
                "search_document ILIKE %s OR "
                "search_document ILIKE %s OR "
                "amenities_building ILIKE %s OR "
                "amenities_building ILIKE %s OR "
                "amenities_area ILIKE %s OR "
                "amenities_area ILIKE %s OR "
                "amenities_area ILIKE %s"
                ")"
            )
            params.extend(["%ho boi%", "%be boi%", "%pool%", "%ho boi%", "%pool%", "%ho boi%", "%be boi%", "%pool%"])

        if bool(getattr(soft, "near_entertainment", False)):
            where.append(
                "(" 
                "search_document ILIKE %s OR "
                "amenities_area ILIKE %s OR "
                "amenities_building ILIKE %s OR "
                "nearby_landmarks ILIKE %s"
                ")"
            )
            params.extend(["%vui choi%", "%vui choi%", "%vui choi%", "%quận%", ])

        if bool(getattr(soft, "family_friendly", False)):
            where.append("(suitable_for ILIKE %s OR search_document ILIKE %s)")
            params.extend(["%gia dinh%", "%gia dinh%"])

    @staticmethod
    def _query_requests_district_view(qn: str) -> bool:
        text = str(qn or "").strip().lower()
        if not text:
            return False
        distribution_triggers = ["theo", "phan bo", "xep hang", "top", "so sanh"]
        district_terms = ["quan", "huyen", "district", "khu vuc"]

        if any(phrase in text for phrase in ["theo quan", "theo huyen", "cac quan", "cac huyen"]):
            return True
        if any(phrase in text for phrase in ["quan nao", "huyen nao", "khu vuc nao"]):
            return True
        return any(trigger in text for trigger in distribution_triggers) and any(
            term in text for term in district_terms
        )

    @staticmethod
    def _query_requests_ward_view(qn: str) -> bool:
        text = str(qn or "").strip().lower()
        if not text:
            return False
        distribution_triggers = ["theo", "phan bo", "xep hang", "top", "so sanh"]
        ward_terms = ["phuong", "xa", "ward", "commune"]

        if any(phrase in text for phrase in ["theo phuong", "theo xa", "cac phuong", "cac xa"]):
            return True
        if any(phrase in text for phrase in ["phuong nao", "xa nao"]):
            return True
        return any(trigger in text for trigger in distribution_triggers) and any(
            term in text for term in ward_terms
        )

    def _build_filters(self, query: str) -> Tuple[List[str], List[Any], Dict[str, Any]]:
        parsed = parse_user_query(query)
        hard = parsed.hard_filters

        where = ["search_document IS NOT NULL", "BTRIM(search_document) <> ''"]
        params: List[Any] = []

        if hard.transaction_type:
            aliases = self._expand_transaction_type_aliases(hard.transaction_type)
            placeholders = ", ".join(["%s"] * len(aliases))
            where.append(f"transaction_type ILIKE ANY(ARRAY[{placeholders}])")
            params.extend([f"%{alias}%" for alias in aliases])
        if hard.property_type:
            aliases = self._expand_property_type_aliases(hard.property_type)
            placeholders = ", ".join(["%s"] * len(aliases))
            where.append(f"property_type ILIKE ANY(ARRAY[{placeholders}])")
            params.extend([f"%{alias}%" for alias in aliases])
        if hard.city:
            where.append("city ILIKE %s")
            params.append(f"%{hard.city}%")
        if hard.district:
            aliases = self._expand_district_aliases(hard.district)
            placeholders = ", ".join(["%s"] * len(aliases))
            where.append(f"district ILIKE ANY(ARRAY[{placeholders}])")
            params.extend([f"%{alias}%" for alias in aliases])
        if hard.max_price_vnd is not None:
            where.append("price_value_vnd <= %s")
            params.append(int(hard.max_price_vnd))
        if hard.min_price_vnd is not None:
            where.append("price_value_vnd >= %s")
            params.append(int(hard.min_price_vnd))
        if hard.min_area_m2 is not None:
            where.append("area_m2 >= %s")
            params.append(float(hard.min_area_m2))
        if hard.min_bedrooms is not None:
            where.append("bedrooms IS NOT NULL AND bedrooms >= %s")
            params.append(int(hard.min_bedrooms))

        applied_filters = {
            "transaction_type": hard.transaction_type,
            "property_type": hard.property_type,
            "city": hard.city,
            "district": hard.district,
            "max_price_vnd": hard.max_price_vnd,
            "min_price_vnd": hard.min_price_vnd,
            "min_area_m2": hard.min_area_m2,
            "min_bedrooms": hard.min_bedrooms,
        }
        return where, params, applied_filters

    def _build_search_sql(self, table_name: str, query: str, top_k: int) -> Tuple[str, List[Any], Dict[str, Any]]:
        where, where_params, applied_filters = self._build_filters(query)
        query_like = f"%{query.strip()}%"
        tokens = self._tokenize(query)

        token_scoring = " + ".join(["CASE WHEN search_document ILIKE %s THEN 1 ELSE 0 END" for _ in tokens])
        if not token_scoring:
            token_scoring = "0"

        sql = f"""
            SELECT
                source,
                listing_id,
                title,
                url,
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
                bedrooms,
                bathrooms,
                floors,
                frontage_width_m,
                road_access_width_m,
                legal_status,
                direction,
                CASE WHEN search_document ILIKE %s THEN 1.0 ELSE 0.0 END AS lexical_score,
                0.0 AS semantic_score,
                (
                    CASE WHEN search_document ILIKE %s THEN 1 ELSE 0 END
                    + {token_scoring}
                )::float AS final_score,
                'db_contract_lexical' AS matched_by
            FROM {table_name}
            WHERE {' AND '.join(where)}
            ORDER BY final_score DESC, listing_id DESC
            LIMIT %s
        """

        params: List[Any] = [query_like, query_like]
        params.extend([f"%{tok}%" for tok in tokens])
        params.extend(where_params)
        params.append(top_k)
        return sql, params, applied_filters

    @classmethod
    def _select_contract_columns(cls) -> str:
        return ",\n                ".join(cls._CONTRACT_COLUMNS)

    def _table_columns(self, conn: psycopg.Connection, table_name: str) -> set[str]:
        cached = self._table_columns_cache.get(table_name)
        if cached is not None:
            return cached

        sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema() AND table_name = %s
        """
        with conn.cursor() as cur:
            cur.execute(sql, [table_name])
            cols = {str(row[0]) for row in (cur.fetchall() or []) if row and row[0]}
        self._table_columns_cache[table_name] = cols
        return cols

    def _select_columns_for_table(self, conn: psycopg.Connection, table_name: str) -> List[str]:
        available_cols = self._table_columns(conn, table_name)
        preferred = list(self._CONTRACT_COLUMNS) + list(self._OPTIONAL_DETAIL_COLUMNS)
        selected = [col for col in preferred if col in available_cols]

        # Ensure minimum identity columns exist before querying the row.
        if "source" not in selected or "listing_id" not in selected:
            return []
        return selected

    def _search_source_candidates(self, conn: psycopg.Connection) -> List[str]:
        available: List[str] = []
        with conn.cursor() as cur:
            for table_name in self.SOURCE_CANDIDATES:
                try:
                    cur.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
                    available.append(table_name)
                except psycopg.errors.UndefinedTable:
                    conn.rollback()
                    self.logger.info("db_contract_table_missing source=%s", table_name)
        return available

    def _search_from_table(self, conn: psycopg.Connection, table_name: str, query: str, top_k: int) -> SearchResult:
        sql, params, applied_filters = self._build_search_sql(table_name, query=query, top_k=top_k)
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            columns = [d[0] for d in cur.description] if cur.description else []

        items: List[Dict[str, Any]] = []
        for row in rows:
            rec = dict(zip(columns, row))
            rec["score"] = float(rec.get("final_score") or 0.0)
            items.append(rec)

        return SearchResult(
            items=items,
            retrieval_stats={
                "retrieval_mode": "db_contract_lexical",
                "fallback_reason": "none",
                "requested_top_k": top_k,
                "returned_count": len(items),
                "matched_signals": [],
                "applied_filters": applied_filters,
                "contract_source": table_name,
            },
        )

    def _get_listing_from_table(
        self,
        conn: psycopg.Connection,
        table_name: str,
        source: str,
        listing_id: str,
    ) -> Dict[str, Any] | None:
        selected_columns = self._select_columns_for_table(conn, table_name)
        if not selected_columns:
            return None

        column_sql = ",\n                ".join(selected_columns)
        sql = f"""
            SELECT
                {column_sql}
            FROM {table_name}
            WHERE source = %s AND listing_id::text = %s
            LIMIT 1
        """
        with conn.cursor() as cur:
            cur.execute(sql, [source, str(listing_id)])
            row = cur.fetchone()
            cols = [d[0] for d in cur.description] if cur.description else []
        if not row:
            return None
        return dict(zip(cols, row))

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_text(value: Any) -> str:
        return str(value or "").strip().lower()

    def _field_contains_any(self, values: List[Any], keywords: set[str]) -> bool:
        haystacks = [self._normalize_text(v) for v in values if v is not None]
        return any(any(keyword in hay for keyword in keywords) for hay in haystacks)

    def _compute_compare_fit_score(self, listing: Dict[str, Any], query: str | None) -> tuple[float, List[str]]:
        parsed = parse_user_query(query or "")
        hard = parsed.hard_filters
        soft = parsed.soft_preferences
        score = 0.0
        reasons: List[str] = []

        budget_cap = hard.max_price_vnd
        price_vnd = self._safe_float(listing.get("price_value_vnd"))
        if budget_cap is not None and price_vnd is not None:
            if price_vnd <= budget_cap:
                score += 2.0
                reasons.append("Nằm trong ngân sách")
            else:
                score -= 1.5
                reasons.append("Vượt ngân sách")

        min_area = hard.min_area_m2
        area_m2 = self._safe_float(listing.get("area_m2"))
        if min_area is not None and area_m2 is not None:
            if area_m2 >= min_area:
                score += 1.0
                reasons.append("Đạt tiêu chí diện tích")
            else:
                score -= 0.8
                reasons.append("Diện tích nhỏ hơn mong muốn")

        min_bedrooms = hard.min_bedrooms
        bedrooms = self._safe_int(listing.get("bedrooms"))
        if min_bedrooms is not None and bedrooms is not None:
            if bedrooms >= min_bedrooms:
                score += 1.0
                reasons.append("Đủ số phòng ngủ")
            else:
                score -= 1.0
                reasons.append("Thiếu phòng ngủ")

        search_doc = self._normalize_text(listing.get("search_document"))
        tokens = [search_doc, listing.get("title"), listing.get("district"), listing.get("project")]

        if soft.near_metro and self._field_contains_any(tokens, {"metro", "ga", "tram", "tau"}):
            score += 0.8
            reasons.append("Có lợi thế di chuyển")
        if soft.near_school and self._field_contains_any(tokens, {"truong", "school", "mam non", "hoc vien"}):
            score += 0.8
            reasons.append("Gần trường học")
        if soft.family_friendly and self._field_contains_any(tokens, {"gia dinh", "family", "yen tinh"}):
            score += 0.8
            reasons.append("Phù hợp nhu cầu gia đình")
        if bool(getattr(soft, "near_entertainment", False)) and self._field_contains_any(tokens, {"vui choi", "giai tri", "trung tam", "rap", "cafe", "nha hang", "khu vui choi", "cong vien"}):
            score += 0.8
            reasons.append("Gần nơi vui chơi/giải trí")

        if not reasons:
            reasons.append("Điểm số dựa trên dữ liệu cấu trúc hiện có")
        return score, reasons

    @staticmethod
    def _priority_weights(priority: List[str] | None) -> Dict[str, float]:
        dims = [str(item or "").strip().lower() for item in (priority or []) if str(item or "").strip()]
        if not dims:
            dims = ["location", "price", "size"]

        unique_dims: List[str] = []
        for dim in dims:
            if dim in {"location", "price", "size"} and dim not in unique_dims:
                unique_dims.append(dim)
        if not unique_dims:
            unique_dims = ["location", "price", "size"]

        total = sum(range(1, len(unique_dims) + 1))
        weights: Dict[str, float] = {}
        for idx, dim in enumerate(unique_dims):
            rank_score = len(unique_dims) - idx
            weights[dim] = rank_score / total

        # Ensure all expected keys exist.
        for dim in ["location", "price", "size"]:
            weights.setdefault(dim, 0.0)
        return weights

    def _compute_profile_fit_score(
        self,
        listing: Dict[str, Any],
        user_profile: Dict[str, Any],
    ) -> tuple[float, Dict[str, Any], List[str]]:
        budget_vnd = self._safe_float(user_profile.get("budget_vnd") or user_profile.get("budget") or user_profile.get("max_budget_vnd"))
        bedrooms_needed = self._safe_int(user_profile.get("bedrooms_needed") or user_profile.get("min_bedrooms"))
        location_preference = self._normalize_text(user_profile.get("location_preference") or user_profile.get("district"))
        commuting_destination = self._normalize_text(user_profile.get("commuting_destination") or user_profile.get("work_location"))
        weights = self._priority_weights(user_profile.get("priority") if isinstance(user_profile.get("priority"), list) else None)

        price_fit = 0.5
        price_vnd = self._safe_float(listing.get("price_value_vnd"))
        if budget_vnd is not None and price_vnd is not None and budget_vnd > 0:
            if price_vnd <= budget_vnd:
                price_fit = 1.0
            else:
                over_ratio = (price_vnd / budget_vnd) - 1.0
                price_fit = max(0.0, round(1.0 - over_ratio, 3))

        size_fit = 0.5
        bedrooms = self._safe_int(listing.get("bedrooms"))
        if bedrooms_needed is not None and bedrooms is not None and bedrooms_needed > 0:
            if bedrooms >= bedrooms_needed:
                size_fit = 1.0
            elif bedrooms == bedrooms_needed - 1:
                size_fit = 0.6
            else:
                size_fit = 0.2

        district_norm = self._normalize_text(listing.get("district"))
        location_fit = 0.5
        if location_preference:
            location_fit = 1.0 if location_preference in district_norm else 0.4

        commute_minutes = None
        if commuting_destination:
            commute_minutes = self._estimate_commute_minutes(commuting_destination, district_norm)
            if commute_minutes is not None:
                if commute_minutes <= 20:
                    location_fit = min(1.0, location_fit + 0.3)
                elif commute_minutes <= 30:
                    location_fit = min(1.0, location_fit + 0.2)
                elif commute_minutes <= 40:
                    location_fit = min(1.0, location_fit + 0.1)

        overall = (
            location_fit * weights.get("location", 0.0)
            + price_fit * weights.get("price", 0.0)
            + size_fit * weights.get("size", 0.0)
        )

        reasons: List[str] = []
        if price_fit >= 0.9:
            reasons.append("Gia phu hop ngan sach")
        elif price_fit <= 0.4:
            reasons.append("Gia vuot ngan sach")
        if location_fit >= 0.9:
            reasons.append("Vi tri phu hop uu tien")
        elif location_fit <= 0.4:
            reasons.append("Vi tri chua phu hop uu tien")
        if size_fit >= 0.9:
            reasons.append("So phong ngu dap ung nhu cau")
        elif size_fit <= 0.4:
            reasons.append("So phong ngu chua dap ung")
        if commute_minutes is not None:
            reasons.append(f"Uoc tinh di chuyen {commute_minutes} phut")

        breakdown = {
            "price_fit": round(price_fit, 3),
            "location_fit": round(location_fit, 3),
            "size_fit": round(size_fit, 3),
            "priority_weights": {k: round(v, 3) for k, v in weights.items()},
            "commute_minutes": commute_minutes,
            "overall_score": round(overall, 3),
        }
        return round(overall, 3), breakdown, reasons

    def _build_area_candidates(
        self,
        destination_norm: str,
        family_signal: bool,
        has_children: bool,
        has_elderly: bool,
        budget_vnd: int,
        property_type: str | None = None,
        city: str | None = None,
        query_text: str = "",
    ) -> List[Dict[str, Any]]:
        inventory_candidates = self._build_area_candidates_from_inventory(
            city=city,
            budget_vnd=budget_vnd,
            property_type=property_type,
            limit=8,
            query_text=query_text,
            destination_norm=destination_norm,
        )
        if inventory_candidates:
            out: List[Dict[str, Any]] = []
            for row in inventory_candidates:
                district = str(row.get("district") or "").strip()
                if not district:
                    continue

                listing_count = int(row.get("listing_count") or 0)
                listing_in_budget_count = int(row.get("listing_in_budget_count") or 0)
                budget_coverage = float(listing_in_budget_count / listing_count) if listing_count > 0 else 0.0
                property_match_count = int(row.get("property_match_count") or 0)
                property_match = float(property_match_count / listing_count) if listing_count > 0 and property_type else 1.0
                reasons: List[str] = []
                reasons.append(f"{int(round(budget_coverage * 100))}% listing trong ngan sach")
                reasons.append(f"{listing_count} listing tong the")
                if property_type:
                    reasons.append(f"{property_match_count} listing khop loai hinh {property_type}")
                distinct_types = int(row.get("distinct_property_type_count") or 0)
                if distinct_types > 1:
                    reasons.append(f"Co {distinct_types} loai hinh de so sanh")

                if budget_vnd <= 0:
                    budget_note = "Can bo sung ngan sach de xep hang chinh xac hon"
                elif budget_coverage >= 0.7:
                    budget_note = "Coverage tot trong khung ngan sach"
                elif budget_coverage >= 0.4:
                    budget_note = "Coverage trung binh, can loc them"
                else:
                    budget_note = "Coverage con thap trong khung ngan sach"

                out.append(
                    {
                        "district": district,
                        "listing_count": listing_count,
                        "listing_in_budget_count": listing_in_budget_count,
                        "budget_coverage": round(budget_coverage, 3),
                        "property_match_count": property_match_count,
                        "property_match": round(property_match, 3),
                        "distinct_property_type_count": int(row.get("distinct_property_type_count") or 0),
                        "median_price_vnd": row.get("median_price_vnd"),
                        "reasons": reasons[:3],
                        "budget_note": budget_note,
                    }
                )

            if out:
                return out

        city_norm = normalize_query_pipeline(str(city or ""))
        tier_low = budget_vnd <= 3_000_000_000
        tier_mid = 3_000_000_000 < budget_vnd <= 5_000_000_000

        if city_norm in {"ha noi", "hanoi"}:
            if tier_low:
                base = ["Nam Tu Liem", "Ha Dong", "Long Bien"]
            elif tier_mid:
                base = ["Cau Giay", "Thanh Xuan", "Nam Tu Liem"]
            else:
                base = ["Tay Ho", "Ba Dinh", "Cau Giay"]
        elif city_norm in {"da nang"}:
            if tier_low:
                base = ["Lien Chieu", "Cam Le", "Ngu Hanh Son"]
            elif tier_mid:
                base = ["Hai Chau", "Thanh Khe", "Ngu Hanh Son"]
            else:
                base = ["Son Tra", "Hai Chau", "Ngu Hanh Son"]
        elif city_norm in {"binh duong", "tinh binh duong"}:
            if tier_low:
                base = ["Thu Dau Mot", "Di An", "Thuan An"]
            elif tier_mid:
                base = ["Thuan An", "Di An", "Ben Cat"]
            else:
                base = ["Thu Dau Mot", "Tan Uyen", "Ben Cat"]
        elif any(k in destination_norm for k in ["quan 1", "q1", "district 1"]):
            if tier_low:
                base = ["Thu Duc", "Go Vap", "Binh Tan"]
            elif tier_mid:
                base = ["Binh Thanh", "Phu Nhuan", "Quan 7"]
            else:
                base = ["Binh Thanh", "Quan 2", "Quan 7"]
        elif any(k in destination_norm for k in ["thu duc", "quan 2", "q2", "district 2"]):
            if tier_low:
                base = ["Thu Duc", "Quan 9", "Binh Tan"]
            elif tier_mid:
                base = ["Thu Duc", "Binh Thanh", "Quan 7"]
            else:
                base = ["Quan 2", "Thu Duc", "Binh Thanh"]
        else:
            if tier_low:
                base = ["Thu Duc", "Binh Tan", "Go Vap"]
            elif tier_mid:
                base = ["Binh Thanh", "Phu Nhuan", "Quan 7"]
            else:
                base = ["Quan 2", "Quan 7", "Phu Nhuan"]

        out: List[Dict[str, Any]] = []
        for district in base:
            reasons: List[str] = ["Co nhieu khu vuc co nguon cung va mat bang gia de so sanh"]
            if budget_vnd > 0:
                reasons.append("Dau hieu phu hop voi khung ngan sach hien tai")
            if tier_low:
                budget_note = "Khung duoi 3 ty"
            elif tier_mid:
                budget_note = "Khung 3-5 ty"
            else:
                budget_note = "Khung tren 5 ty"

            out.append({"district": district, "reasons": reasons[:3], "budget_note": budget_note})
        return out

    @staticmethod
    def _estimate_commute_minutes(destination_norm: str, district: str) -> int | None:
        district_norm = str(district or "").strip().lower()
        if not destination_norm:
            return None

        if any(k in destination_norm for k in ["quan 1", "q1", "district 1"]):
            return {
                "binh thanh": 18,
                "phu nhuan": 20,
                "quan 7": 28,
                "thu duc": 35,
                "go vap": 32,
                "binh tan": 42,
                "quan 2": 24,
                "quan 9": 45,
            }.get(district_norm)

        if any(k in destination_norm for k in ["thu duc", "quan 2", "q2", "district 2"]):
            return {
                "thu duc": 18,
                "quan 2": 20,
                "binh thanh": 24,
                "quan 7": 30,
                "go vap": 38,
                "binh tan": 48,
                "quan 9": 22,
                "phu nhuan": 35,
            }.get(district_norm)

        return {
            "thu duc": 30,
            "quan 2": 30,
            "binh thanh": 28,
            "quan 7": 32,
            "go vap": 34,
            "binh tan": 40,
            "quan 9": 35,
            "phu nhuan": 30,
        }.get(district_norm)

    @staticmethod
    def _estimate_listing_count(district: str, budget_vnd: int) -> int:
        # Heuristic until we bind to real district-level inventory aggregation.
        base_map = {
            "thu duc": 42,
            "go vap": 30,
            "binh tan": 26,
            "binh thanh": 34,
            "phu nhuan": 22,
            "quan 7": 29,
            "quan 2": 31,
            "quan 9": 24,
        }
        base = base_map.get(str(district or "").strip().lower(), 20)
        if budget_vnd <= 0:
            return max(8, int(base * 0.5))
        if budget_vnd <= 3_000_000_000:
            return max(6, int(base * 0.75))
        if budget_vnd <= 5_000_000_000:
            return max(8, int(base * 1.0))
        return max(10, int(base * 1.15))

    @staticmethod
    def _clamp_unit(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @classmethod
    def _fit_budget_ratio(cls, budget_vnd: int, median_price_vnd: int | None, budget_coverage: float) -> float:
        del median_price_vnd
        if budget_vnd <= 0:
            return 0.0
        return round(cls._clamp_unit(budget_coverage), 3)

    @classmethod
    def _fit_inventory_ratio(cls, total_listing_count: int, max_listing_count: int | None = None) -> float:
        if total_listing_count <= 0:
            return 0.0
        if not max_listing_count or max_listing_count <= 0:
            return round(cls._clamp_unit(float(total_listing_count) / 100.0), 3)
        if max_listing_count <= 1:
            return 1.0
        return round(cls._clamp_unit(math.log1p(float(total_listing_count)) / math.log1p(float(max_listing_count))), 3)

    @classmethod
    def _fit_property_match_ratio(cls, total_listing_count: int, property_match_count: int | None, property_type: str | None) -> float:
        if not property_type:
            return 1.0
        if total_listing_count <= 0:
            return 0.0
        return round(cls._clamp_unit(float(property_match_count or 0) / float(total_listing_count)), 3)

    @classmethod
    def _location_centrality_score(cls, district: str, city: str | None = None) -> float:
        district_key = normalize_query_pipeline(str(district or "")).strip().lower()
        city_key = normalize_query_pipeline(str(city or "")).strip().lower()
        if not district_key:
            return 0.5

        candidates: List[Dict[str, float]] = []
        if city_key in {"ha noi", "hanoi"}:
            candidates.append(cls._HANOI_CENTRALITY_MAP)
        elif city_key in {"ho chi minh", "hcm", "sai gon", "saigon", "tp hcm"}:
            candidates.append(cls._HCM_CENTRALITY_MAP)
        elif city_key in {"binh duong", "tinh binh duong"}:
            candidates.append(cls._BINH_DUONG_CENTRALITY_MAP)
        else:
            candidates.extend([cls._HCM_CENTRALITY_MAP, cls._HANOI_CENTRALITY_MAP, cls._BINH_DUONG_CENTRALITY_MAP])

        best = None
        for mapping in candidates:
            for key, value in mapping.items():
                if re.search(rf"(?<![0-9a-z]){re.escape(key)}(?![0-9a-z])", district_key):
                    best = max(float(best), float(value)) if best is not None else float(value)
        return round(cls._clamp_unit(best if best is not None else 0.5), 3)

    @classmethod
    def _location_score(cls, district: str, city: str | None, destination_norm: str) -> float:
        centrality_score = cls._location_centrality_score(district=district, city=city)
        commute_minutes = cls._estimate_commute_minutes(destination_norm, district)
        if commute_minutes is None:
            return round(cls._clamp_unit(centrality_score), 3)
        commute_score = cls._clamp_unit(1.0 - (float(commute_minutes) / 45.0))
        return round(cls._clamp_unit((centrality_score * 0.7) + (commute_score * 0.3)), 3)

    @classmethod
    def _soft_preference_signals_for_item(cls, record: Dict[str, Any], soft: Any) -> List[str]:
        if soft is None:
            return []

        search_doc = cls._normalize_text(record.get("search_document"))
        view_text = cls._normalize_text(record.get("view"))
        amenities_building = cls._normalize_text(record.get("amenities_building"))
        amenities_area = cls._normalize_text(record.get("amenities_area"))
        nearby_transport = cls._normalize_text(record.get("nearby_transport"))
        nearby_landmarks = cls._normalize_text(record.get("nearby_landmarks"))
        nearby_roads = cls._normalize_text(record.get("nearby_roads"))

        def _has_any(*keywords: str) -> bool:
            texts = (search_doc, view_text, amenities_building, amenities_area, nearby_transport, nearby_landmarks, nearby_roads)
            return any(keyword and any(keyword in text for text in texts) for keyword in keywords)

        matched: List[str] = []
        if bool(getattr(soft, "near_metro", False)) and _has_any("metro", "mrt", "lrt", "ga", "tram", "tau dien"):
            matched.append("near_metro")
        if bool(getattr(soft, "near_school", False)) and _has_any("truong", "school", "mam non", "hoc vien", "tieu hoc", "trung hoc", "dai hoc"):
            matched.append("near_school")
        if bool(getattr(soft, "quiet_area", False)) and _has_any("yen tinh", "it on", "an tinh", "quiet"):
            matched.append("quiet_area")
        if bool(getattr(soft, "family_friendly", False)) and _has_any("gia dinh", "family", "tre em", "con nho"):
            matched.append("family_friendly")
        if bool(getattr(soft, "wants_gym", False)) and _has_any("gym", "fitness", "phong tap", "the thao", "sports"):
            matched.append("wants_gym")
        if bool(getattr(soft, "wants_pool", False)) and _has_any("ho boi", "be boi", "pool", "swimming"):
            matched.append("wants_pool")
        if bool(getattr(soft, "near_entertainment", False)) and _has_any("vui choi", "giai tri", "entertainment", "trung tam thuong mai", "trung tam", "rap chieu", "rap", "pho di bo", "cafe", "nha hang", "cong vien", "bar", "nightlife"):
            matched.append("near_entertainment")
        if bool(getattr(soft, "view", False)) and _has_any("view", "thoang", "thoang mat", "mat thoang", "song view", "park view", "city view"):
            matched.append("view")
        if bool(getattr(soft, "nearby_transport", False)) and _has_any("ga xe", "tram xe", "ben xe", "xe bus", "xe buyt", "bus", "tram"):
            matched.append("nearby_transport")
        if bool(getattr(soft, "nearby_landmarks", False)) and _has_any("gan cho", "gan truong", "gan benh vien", "gan cong vien", "gan sieu thi", "gan trung tam thuong mai", "gan landmark", "gan noi lam viec", "cong vien"):
            matched.append("nearby_landmarks")
        if bool(getattr(soft, "nearby_roads", False)) and _has_any("gan duong lon", "duong lon", "mat tien duong", "gan duong", "gan pho", "road access", "near road"):
            matched.append("nearby_roads")

        return matched

    @staticmethod
    def _soft_preference_boost(matched_soft_preferences: List[str]) -> float:
        if not matched_soft_preferences:
            return 0.0
        return round(min(0.20, 0.06 * len(matched_soft_preferences)), 3)

    @classmethod
    def _persona_score(
        cls,
        query_text: str,
        budget_vnd: int,
        property_type: str | None,
        family_signal: bool,
        location_score: float,
        property_type_fit: float,
    ) -> tuple[float, List[str]]:
        normalized_query = normalize_query_pipeline(str(query_text or "")).strip().lower()
        score = 0.45
        reasons: List[str] = []

        center_intent_terms = [
            "gan trung tam",
            "gần trung tâm",
            "trung tam",
            "trung tâm",
            "sat trung tam",
            "sát trung tâm",
            "cbd",
            "khu lam viec",
            "khu làm việc",
        ]
        if any(term in normalized_query for term in center_intent_terms):
            score = max(score, 0.70 + (location_score * 0.20))
            reasons.append("Ưu tiên gần trung tâm")

        if any(term in normalized_query for term in ["an cu", "an cư", "o that", "ở thật", "chat luong song", "chất lượng sống"]):
            score += 0.08
            reasons.append("Hợp nhu cầu ở thực và chất lượng sống")

        if any(term in normalized_query for term in ["giu gia", "giữ giá", "thanh khoan", "thanh khoản", "sinh loi", "sinh lời"]):
            score += 0.08
            reasons.append("Có lợi cho giữ giá và thanh khoản")

        if budget_vnd >= 10_000_000_000:
            score += 0.08
            reasons.append("Ngân sách đủ rộng để ưu tiên vị trí, chất lượng sống và thanh khoản")
        elif budget_vnd >= 5_000_000_000:
            score += 0.04

        property_type_norm = normalize_query_pipeline(str(property_type or "")).strip().lower()
        if property_type_norm and any(alias in property_type_norm for alias in ["can ho", "chung cu", "apartment"]):
            score += 0.06 if property_type_fit >= 0.5 else 0.02
            reasons.append("Đúng phân khúc căn hộ")

        if family_signal:
            score += 0.04
            reasons.append("Phù hợp nhu cầu ở thực")

        score += min(0.10, location_score * 0.10)
        score = cls._clamp_unit(score)
        if not reasons:
            reasons.append("Phù hợp hồ sơ nhu cầu hiện tại")
        return round(score, 3), reasons

    @classmethod
    def _area_score(
        cls,
        budget_fit: float,
        property_type_fit: float,
        inventory_score: float,
        location_score: float,
        persona_score: float,
        weights: Dict[str, float],
        sample_confidence: float = 1.0,
    ) -> float:
        score = (
            + cls._clamp_unit(location_score) * weights.get("location_score", 0.0)
            + cls._clamp_unit(persona_score) * weights.get("persona_score", 0.0)
            + cls._clamp_unit(budget_fit) * weights.get("budget_fit", 0.0)
            + cls._clamp_unit(property_type_fit) * weights.get("property_type_fit", 0.0)
            + cls._clamp_unit(inventory_score) * weights.get("inventory_score", 0.0)
        )
        score *= cls._clamp_unit(sample_confidence)
        return round(cls._clamp_unit(score), 3)

    @classmethod
    def _area_score_weights(cls, query_text: str, family_signal: bool, destination_norm: str, near_metro: bool) -> Dict[str, float]:
        # CHANGED (2026-06-27): Remove inventory bias - set inventory_score to 0
        # Focus on core criteria: location, persona, budget fit, property type
        normalized_query = normalize_query_pipeline(str(query_text or "")).strip().lower()
        center_priority_terms = [
            "gan trung tam",
            "gần trung tâm",
            "trung tam",
            "trung tâm",
            "cbd",
            "thu thiem",
            "khu lam viec",
            "khu làm việc",
            "truc giao thong chinh",
            "trục giao thông chính",
        ]
        commute_priority = bool(destination_norm) or near_metro or any(term in normalized_query for term in ["metro", "ga", "tram", "tau dien", "tàu điện"])
        if any(term in normalized_query for term in center_priority_terms):
            return dict(cls._CENTER_PRIORITY_AREA_WEIGHTS)
        if commute_priority:
            return dict(cls._COMMUTE_PRIORITY_AREA_WEIGHTS)
        if family_signal:
            return dict(cls._DEFAULT_AREA_WEIGHTS)
        return dict(cls._DEFAULT_AREA_WEIGHTS)

    def _build_area_candidates_from_inventory(
        self,
        city: str | None,
        budget_vnd: int,
        property_type: str | None = None,
        limit: int = 8,
        query_text: str = "",
        destination_norm: str = "",
        near_metro: bool = False,
    ) -> List[Dict[str, Any]]:
        row_limit = max(3, min(20, int(limit)))
        fetch_limit = max(40, row_limit * 8)
        city_aliases = self._expand_city_aliases(city or "") if city else []
        property_type_aliases = self._expand_property_type_aliases(property_type) if property_type else []
        property_type_params = [f"%{alias}%" for alias in property_type_aliases]
        normalized_query = normalize_query_pipeline(str(query_text or "")).strip().lower()
        center_intent = any(
            term in normalized_query
            for term in ["gan trung tam", "gần trung tâm", "trung tam", "trung tâm", "sat trung tam", "sát trung tâm", "cbd"]
        )
        commute_priority = bool(destination_norm) or near_metro or any(term in normalized_query for term in ["metro", "ga", "tram", "tau dien", "tàu điện"])

        with self._connect() as conn:
            for table_name in self.SOURCE_CANDIDATES:
                where = ["district IS NOT NULL", "district <> ''"]
                params: List[Any] = []

                if city_aliases:
                    placeholders = ", ".join(["%s"] * len(city_aliases))
                    where.append(f"(LOWER(COALESCE(city, '')) LIKE ANY (ARRAY[{placeholders}]))")
                    params.extend([f"%{str(alias).lower()}%" for alias in city_aliases])

                if budget_vnd > 0:
                    where.append("COALESCE(price_value_vnd, 0) > 0")
                budget_clause = "COUNT(*)::int AS listing_in_budget_count"
                if budget_vnd > 0:
                    budget_clause = "COUNT(*) FILTER (WHERE price_value_vnd IS NOT NULL AND price_value_vnd > 0 AND price_value_vnd <= %s)::int AS listing_in_budget_count"
                    params.append(int(budget_vnd))

                property_clause = "COUNT(*)::int AS property_match_count"
                if property_type_params:
                    placeholders = ", ".join(["%s"] * len(property_type_params))
                    property_clause = (
                        f"COUNT(*) FILTER (WHERE LOWER(COALESCE(property_type, '')) ILIKE ANY(ARRAY[{placeholders}]))::int AS property_match_count"
                    )
                    params.extend(property_type_params)

                sql = f"""
                    SELECT district, COUNT(*)::int AS listing_count
                    FROM {table_name}
                    WHERE {' AND '.join(where)}
                    GROUP BY district
                    ORDER BY listing_count DESC
                    LIMIT %s
                """

                sql = f"""
                    SELECT
                        district,
                        COUNT(*)::int AS total_listing_count,
                        {budget_clause},
                        {property_clause},
                        COUNT(DISTINCT COALESCE(NULLIF(BTRIM(property_type), ''), 'Khac'))::int AS distinct_property_type_count,
                        percentile_cont(0.5) WITHIN GROUP (ORDER BY price_value_vnd) FILTER (WHERE price_value_vnd IS NOT NULL AND price_value_vnd > 0)::BIGINT AS median_price_vnd
                    FROM {table_name}
                    WHERE {' AND '.join(where)}
                    GROUP BY district
                    ORDER BY total_listing_count DESC
                    LIMIT %s
                """

                try:
                    with conn.cursor() as cur:
                        cur.execute(sql, [*params, fetch_limit])
                        rows = cur.fetchall()
                except psycopg.errors.UndefinedTable:
                    conn.rollback()
                    continue
                except Exception as exc:
                    conn.rollback()
                    self.logger.warning("suggest_area_inventory_query_failed table=%s error=%s", table_name, exc)
                    continue

                out: List[Dict[str, Any]] = []
                for row in rows:
                    district, listing_count, listing_in_budget_count, property_match_count, distinct_property_type_count, median_price_vnd = row
                    district_text = str(district or "").strip()
                    if not district_text:
                        continue
                    centrality_score = self._location_centrality_score(district=district_text, city=city)
                    if center_intent and centrality_score < 0.65:
                        continue
                    inventory_density = min(1.0, float(listing_count or 0) / float(max(fetch_limit, 1)))
                    if center_intent:
                        candidate_priority = (centrality_score * 0.82) + (inventory_density * 0.05)
                    elif commute_priority:
                        candidate_priority = (centrality_score * 0.45) + (inventory_density * 0.20)
                    else:
                        candidate_priority = (centrality_score * 0.30) + (inventory_density * 0.25)
                    if budget_vnd > 0:
                        candidate_priority += min(0.08, float(listing_in_budget_count or 0) / float(max(listing_count or 1, 1)) * 0.08)
                    if property_type_aliases:
                        candidate_priority += min(0.08, float(property_match_count or 0) / float(max(listing_count or 1, 1)) * 0.08)
                    out.append(
                        {
                            "district": district_text,
                            "listing_count": int(listing_count or 0),
                            "listing_in_budget_count": int(listing_in_budget_count or 0),
                            "property_match_count": int(property_match_count or 0),
                            "distinct_property_type_count": int(distinct_property_type_count or 0),
                            "median_price_vnd": int(median_price_vnd or 0) if median_price_vnd is not None else None,
                            "centrality_score": round(centrality_score, 3),
                            "candidate_priority": round(candidate_priority, 3),
                        }
                    )

                if out:
                    out.sort(key=lambda item: (-float(item.get("candidate_priority") or 0.0), -float(item.get("centrality_score") or 0.0), -int(item.get("listing_count") or 0)))
                    return out[:row_limit]

        return []

    def _rank_area_candidates(
        self,
        base_candidates: List[Dict[str, Any]],
        query_text: str,
        destination_norm: str,
        budget_vnd: int,
        family_signal: bool,
        property_type: str | None,
        city: str | None,
        near_metro: bool,
    ) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        weights = self._area_score_weights(
            query_text=query_text,
            family_signal=family_signal,
            destination_norm=destination_norm,
            near_metro=near_metro,
        )
        max_listing_count = max((int(candidate.get("listing_count") or candidate.get("listing_count_hint") or 0) for candidate in base_candidates), default=0)

        for candidate in base_candidates:
            district = str(candidate.get("district") or "").strip()
            if not district:
                continue

            listing_count = max(0, int(candidate.get("listing_count") or candidate.get("listing_count_hint") or 0))
            if listing_count <= 0:
                continue
            listing_in_budget_count = max(0, int(candidate.get("listing_in_budget_count") or 0))
            distinct_property_type_count = candidate.get("distinct_property_type_count")
            median_price_vnd = candidate.get("median_price_vnd")
            property_match_count = candidate.get("property_match_count")

            budget_coverage = round((listing_in_budget_count / listing_count), 3) if listing_count > 0 else 0.0
            budget_fit = self._fit_budget_ratio(budget_vnd, median_price_vnd if isinstance(median_price_vnd, int) else None, budget_coverage)
            inventory_score = self._fit_inventory_ratio(listing_count, max_listing_count)
            property_type_hint = property_type or str(candidate.get("property_type") or "").strip() or None
            property_type_fit = self._fit_property_match_ratio(listing_count, property_match_count if isinstance(property_match_count, int) else None, property_type_hint)
            if property_type_hint and property_type_fit <= 0.0:
                continue
            location_score = self._location_score(district=district, city=city, destination_norm=destination_norm)
            persona_score, persona_reasons = self._persona_score(
                query_text=query_text,
                budget_vnd=budget_vnd,
                property_type=property_type_hint,
                family_signal=family_signal,
                location_score=location_score,
                property_type_fit=property_type_fit,
            )
            sample_confidence = 1.0
            if listing_count < 5:
                sample_confidence = 0.55
            elif listing_count < 10:
                sample_confidence = 0.75
            elif listing_count < 20:
                sample_confidence = 0.9
            score = self._area_score(
                budget_fit=budget_fit,
                property_type_fit=property_type_fit,
                inventory_score=inventory_score,
                location_score=location_score,
                persona_score=persona_score,
                weights=weights,
                sample_confidence=sample_confidence,
            )

            matching_reasons: List[str] = []
            non_matching_reasons: List[str] = []

            if location_score >= 0.9:
                matching_reasons.append("Rất gần trung tâm hoặc kết nối đi lại thuận tiện")
            elif location_score >= 0.75:
                matching_reasons.append("Vị trí thuận tiện, phù hợp ưu tiên di chuyển")
            else:
                non_matching_reasons.append("Xa trung tâm hơn so với các lựa chọn ưu tiên")

            matching_reasons.extend(persona_reasons[:2])

            budget_coverage_pct = int(round(budget_coverage * 100)) if listing_count > 0 else 0
            if budget_vnd > 0:
                if budget_vnd >= 10_000_000_000:
                    matching_reasons.append("Ngân sách đủ rộng để ưu tiên vị trí, chất lượng sống và thanh khoản")
                else:
                    # CHANGED (2026-06-27): Show proper coverage format (X/Y) instead of confusing "54/0"
                    if listing_in_budget_count > 0 and listing_count > 0:
                        matching_reasons.append(f"{budget_coverage_pct}% nguồn cung phù hợp ngân sách ({listing_in_budget_count}/{listing_count} listing)")
                    elif listing_count > 0:
                        matching_reasons.append(f"Chỉ {budget_coverage_pct}% listing phù hợp ngân sách")
                    else:
                        non_matching_reasons.append("Chưa có dữ liệu listing để đo mức độ phù hợp ngân sách")
            else:
                non_matching_reasons.append("Chưa có khung ngân sách để đo mức độ phù hợp")

            if median_price_vnd is not None and budget_vnd > 0:
                # CHANGED (2026-06-27): Remove duplicate reason - avoid saying both "100% fit" and "median in budget"
                if float(median_price_vnd) <= float(budget_vnd):
                    # Only add if not already said in budget_fit message above
                    if budget_coverage_pct < 100:
                        matching_reasons.append("Mặt bằng giá trung vị còn nằm trong khung chi trả")
                else:
                    non_matching_reasons.append("Mặt bằng giá cao hơn ngân sách")

            # CHANGED (2026-06-27): Don't use inventory as ranking criteria - move to trade-off/note section
            if listing_count >= 30:
                pass  # Don't boost score - just note in tradeoffs
            elif listing_count >= 15:
                pass  # Similar - it's a note, not a ranking boost
            elif listing_count > 0:
                # Warn about low sample size affecting confidence
                if listing_count < 5:
                    non_matching_reasons.append(f"⚠️ Nguồn cung mỏng ({listing_count} listing) - cần thận trọng với độ tin cậy")
            else:
                non_matching_reasons.append("Chưa có dữ liệu listing để đánh giá khu vực")

            if distinct_property_type_count is not None:
                if int(distinct_property_type_count or 0) >= 3:
                    matching_reasons.append("Có đa dạng loại hình để so sánh")
                elif int(distinct_property_type_count or 0) <= 1:
                    non_matching_reasons.append("Loại hình trong khu vực còn tập trung")

            if property_type_hint:
                if property_type_fit >= 0.6:
                    matching_reasons.append(f"Khớp loại hình {property_type_hint}")
                else:
                    non_matching_reasons.append(f"Loại hình {property_type_hint} còn ít trong khu vực")

            inventory_score_text = round(inventory_score, 3)

            ranked.append(
                {
                    "district": district,
                    "score": score,
                    "persona_score": persona_score,
                    "budget_fit": budget_fit,
                    "property_type_fit": property_type_fit,
                    "inventory_score": inventory_score_text,
                    "location_score": location_score,
                    "inventory_fit": inventory_score_text,
                    "property_match": property_type_fit,
                    "listing_count": listing_count,
                    "listing_in_budget_count": listing_in_budget_count,
                    "budget_coverage": budget_coverage,
                    "median_price_vnd": median_price_vnd,
                    "distinct_property_type_count": distinct_property_type_count,
                    "sample_confidence": round(sample_confidence, 3),
                    "matching_reasons": matching_reasons[:4],
                    "non_matching_reasons": non_matching_reasons[:2],
                }
            )

        ranked.sort(key=lambda item: (-float(item.get("score") or 0.0), -float(item.get("budget_coverage") or 0.0), -int(item.get("listing_count") or 0)))
        for idx, row in enumerate(ranked, start=1):
            row["rank"] = idx
        return ranked

    def _build_semantic_area_fallback(
        self,
        query_text: str,
        top_k: int,
        budget_vnd: int,
        property_type: str | None,
        destination_norm: str,
        family_signal: bool,
        city: str | None,
    ) -> List[Dict[str, Any]]:
        normalized_query = normalize_query_pipeline(str(query_text or "")).strip().lower()
        center_intent = any(
            term in normalized_query
            for term in ["gan trung tam", "gần trung tâm", "trung tam", "trung tâm", "sat trung tam", "sát trung tâm", "cbd"]
        )
        if city:
            city_key = normalize_query_pipeline(str(city or "")).strip().lower()
        else:
            city_key = "ho chi minh"

        if city_key in {"ha noi", "hanoi"}:
            centrality_map = self._HANOI_CENTRALITY_MAP
            candidates = ["Hoan Kiem", "Ba Dinh", "Hai Ba Trung", "Dong Da", "Tay Ho", "Cau Giay", "Thanh Xuan"]
        elif city_key in {"binh duong", "tinh binh duong"}:
            centrality_map = self._BINH_DUONG_CENTRALITY_MAP
            candidates = ["Thu Dau Mot", "Thuan An", "Di An", "Ben Cat", "Tan Uyen", "Bau Bang", "Dau Tieng", "Phu Giao", "Bac Tan Uyen"]
        else:
            centrality_map = self._HCM_CENTRALITY_MAP
            candidates = ["Quan 1", "Quan 3", "Binh Thanh", "Phu Nhuan", "Quan 4", "Quan 2", "Tan Binh", "Quan 5"]

        property_type_label = property_type or "căn hộ"
        budget_text = "15 tỷ" if budget_vnd >= 10_000_000_000 else ("khung ngân sách hiện tại" if budget_vnd <= 0 else f"{budget_vnd / 1_000_000_000:.1f} tỷ")

        out: List[Dict[str, Any]] = []
        for district in candidates:
            centrality = 0.5
            normalized_district = normalize_query_pipeline(district).strip().lower()
            for key, value in centrality_map.items():
                if key in normalized_district:
                    centrality = max(centrality, float(value))

            persona_score, persona_reasons = self._persona_score(
                query_text=query_text,
                budget_vnd=budget_vnd,
                property_type=property_type_label,
                family_signal=family_signal,
                location_score=centrality,
                property_type_fit=1.0,
            )
            if center_intent:
                score = round(self._clamp_unit((centrality * 0.72) + (persona_score * 0.28)), 3)
            else:
                score = round(self._clamp_unit((centrality * 0.60) + (persona_score * 0.40)), 3)

            reasons: List[str] = []
            if centrality >= 0.9:
                reasons.append("Sát trung tâm, thuận tiện di chuyển")
            elif centrality >= 0.8:
                reasons.append("Rất gần trung tâm")
            else:
                reasons.append("Là lựa chọn trung tâm vừa phải, vẫn cân bằng được di chuyển")
            reasons.extend(persona_reasons[:2])
            reasons.append(f"Ngân sách {budget_text} phù hợp để ưu tiên vị trí và chất lượng sống")
            if property_type_label:
                reasons.append(f"Phù hợp phân khúc {property_type_label}")

            out.append(
                {
                    "district": district,
                    "area": district,
                    "score": score,
                    "reason": "; ".join(list(dict.fromkeys(reasons))[:3]),
                    "matching_reasons": list(dict.fromkeys(reasons))[:4],
                    "estimated_price": None,
                    "estimated_price_vnd": None,
                    "inventory_level": 0,
                    "listing_count": 0,
                    "commute_minutes": None,
                    "market_comment": None,
                    "area_comment": None,
                    "common_property": "Chung cư" if property_type_label else None,
                    "price_range_text": None,
                    "rank": None,
                    "semantic_fallback": True,
                }
            )

        out.sort(key=lambda item: (-float(item.get("score") or 0.0), str(item.get("district") or "")))
        for idx, row in enumerate(out[: max(1, min(20, int(top_k)))], start=1):
            row["rank"] = idx
        return out[: max(1, min(20, int(top_k)))]

    def _build_district_market_snapshot(self, district: str, city: str | None, budget_vnd: int) -> Dict[str, Any]:
        district_aliases = self._expand_district_aliases(district)
        if not district_aliases:
            return {}

        city_aliases = self._expand_city_aliases(city or "") if city else []
        where = ["district IS NOT NULL", "BTRIM(district) <> ''", "price_value_vnd IS NOT NULL", "price_value_vnd > 0"]
        params: List[Any] = []

        placeholders = ", ".join(["%s"] * len(district_aliases))
        where.append(f"district ILIKE ANY(ARRAY[{placeholders}])")
        params.extend([f"%{alias}%" for alias in district_aliases])

        if city_aliases:
            placeholders = ", ".join(["%s"] * len(city_aliases))
            where.append(f"city ILIKE ANY(ARRAY[{placeholders}])")
            params.extend([f"%{alias}%" for alias in city_aliases])

        if budget_vnd > 0:
            where.append("price_value_vnd <= %s")
            params.append(int(budget_vnd))

        where_clause = " AND ".join(where)
        aggregate_sql = f"""
            SELECT
                COUNT(*)::int AS listing_count,
                AVG(price_value_vnd)::BIGINT AS avg_price_vnd,
                MIN(price_value_vnd)::BIGINT AS min_price_vnd,
                MAX(price_value_vnd)::BIGINT AS max_price_vnd,
                AVG(area_m2)::FLOAT AS avg_area_m2,
                AVG(
                    CASE
                        WHEN price_value_vnd IS NOT NULL AND area_m2 IS NOT NULL AND area_m2 > 0
                        THEN price_value_vnd / area_m2
                        ELSE NULL
                    END
                )::FLOAT AS avg_price_per_m2_vnd
            FROM listings
            WHERE {where_clause}
        """
        property_sql = f"""
            SELECT
                COALESCE(NULLIF(BTRIM(property_type), ''), 'Khac') AS property_type_bucket,
                COUNT(*)::int AS cnt
            FROM listings
            WHERE {where_clause}
            GROUP BY property_type_bucket
            ORDER BY cnt DESC, property_type_bucket ASC
            LIMIT 1
        """

        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(aggregate_sql, params)
                    agg_row = cur.fetchone() or (0, None, None, None, None, None)
                    cur.execute(property_sql, params)
                    property_row = cur.fetchone()
        except Exception as exc:
            self.logger.warning("suggest_area_snapshot_failed district=%s error=%s", district, exc)
            return {}

        listing_count = int(agg_row[0] or 0)
        avg_price_vnd = agg_row[1]
        min_price_vnd = agg_row[2]
        max_price_vnd = agg_row[3]
        avg_area_m2 = agg_row[4]
        avg_price_per_m2_vnd = agg_row[5]
        dominant_property_type = str(property_row[0] or "").strip() if property_row else ""

        if listing_count >= 30:
            market_comment = "nguồn cung dày, dễ so sánh nhiều phương án"
            confidence = "high"
        elif listing_count >= 12:
            market_comment = "nguồn cung đủ để chọn lọc, nhưng vẫn cần chốt tiêu chí kỹ"
            confidence = "medium"
        else:
            market_comment = "nguồn cung mỏng hơn, phù hợp khi muốn ưu tiên vị trí hoặc độ hiếm"
            confidence = "low"

        if avg_price_vnd is not None and budget_vnd > 0:
            if float(avg_price_vnd) <= budget_vnd * 0.9:
                market_comment = "mặt bằng còn mềm hơn ngân sách, có thể ưu tiên diện tích hoặc vị trí tốt hơn"
            elif float(avg_price_vnd) >= budget_vnd * 1.05:
                market_comment = "giá đang tiệm cận hoặc nhỉnh hơn ngân sách, cần lọc kỹ căn phù hợp"

        if avg_area_m2 is not None and float(avg_area_m2) >= 75:
            area_comment = "diện tích trung bình khá rộng"
        elif avg_area_m2 is not None and float(avg_area_m2) < 60:
            area_comment = "diện tích trung bình gọn hơn, hợp người muốn tối ưu vốn"
        else:
            area_comment = "diện tích trung bình ở mức cân bằng"

        price_range_text = None
        if min_price_vnd is not None and max_price_vnd is not None:
            price_range_text = f"{int(min_price_vnd) / 1_000_000_000:.1f}-{int(max_price_vnd) / 1_000_000_000:.1f} tỷ"

        return {
            "listing_count": listing_count,
            "avg_price_vnd": avg_price_vnd,
            "min_price_vnd": min_price_vnd,
            "max_price_vnd": max_price_vnd,
            "avg_area_m2": avg_area_m2,
            "avg_price_per_m2_vnd": avg_price_per_m2_vnd,
            "common_property": dominant_property_type,
            "market_comment": market_comment,
            "area_comment": area_comment,
            "confidence": confidence,
            "price_range_text": price_range_text,
        }

    def _extract_utilities(self, search_doc_text: str) -> List[str]:
        text = self._normalize_text(search_doc_text)
        utilities: List[str] = []
        if any(k in text for k in ["gym", "phong tap", "phòng tập"]):
            utilities.append("gym")
        if any(k in text for k in ["pool", "ho boi", "hồ bơi", "be boi", "bể bơi"]):
            utilities.append("pool")
        if any(k in text for k in ["truong", "trường", "school", "mam non", "mầm non"]):
            utilities.append("school_access")
        if any(k in text for k in ["metro", "ga", "tram", "trạm", "tau", "tàu"]):
            utilities.append("transport_access")
        if any(k in text for k in ["benh vien", "bệnh viện", "hospital", "phong kham", "phòng khám"]):
            utilities.append("healthcare_access")
        # Preserve order and deduplicate.
        return list(dict.fromkeys(utilities))

    def _build_price_analysis(self, record: Dict[str, Any], budget_cap: float | None) -> Dict[str, Any]:
        price_value = self._safe_float(record.get("price_value_vnd"))
        area_m2 = self._safe_float(record.get("area_m2"))
        price_per_m2 = None
        if price_value is not None and area_m2 is not None and area_m2 > 0:
            price_per_m2 = round(price_value / area_m2, 2)

        budget_fit = "unknown"
        if budget_cap is not None and price_value is not None:
            budget_fit = "within_budget" if price_value <= budget_cap else "over_budget"

        return {
            "price_value_vnd": int(price_value) if price_value is not None else None,
            "area_m2": area_m2,
            "price_per_m2": price_per_m2,
            "budget_cap_vnd": int(budget_cap) if budget_cap is not None else None,
            "budget_fit": budget_fit,
        }

    def _build_location_analysis(
        self,
        record: Dict[str, Any],
        district_match: bool,
        destination: str | None,
        utilities: List[str],
    ) -> Dict[str, Any]:
        amenity_hints = [
            value
            for value in ["transport_access", "school_access", "healthcare_access"]
            if value in utilities
        ]
        return {
            "city": record.get("city"),
            "district": record.get("district"),
            "ward": record.get("ward"),
            "destination_hint": destination,
            "district_match": district_match,
            "amenity_hints": amenity_hints,
        }

    def _build_size_analysis(self, record: Dict[str, Any], min_area: float | None, min_bedrooms: int | None) -> Dict[str, Any]:
        area_m2 = self._safe_float(record.get("area_m2"))
        bedrooms = self._safe_int(record.get("bedrooms"))
        bathrooms = self._safe_int(record.get("bathrooms"))
        return {
            "area_m2": area_m2,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "min_area_required": min_area,
            "min_bedrooms_required": min_bedrooms,
            "meets_area_requirement": bool(min_area is None or (area_m2 is not None and area_m2 >= min_area)),
            "meets_bedroom_requirement": bool(
                min_bedrooms is None or (bedrooms is not None and bedrooms >= min_bedrooms)
            ),
        }

    def search_listings(self, query: str, top_k: int = 5, parsed_filters: Optional[Dict[str, Any]] = None) -> SearchResult:
        query_text = str(query or "").strip()
        size = max(1, min(50, int(top_k)))

        if not query_text:
            return SearchResult(
                items=[],
                retrieval_stats={
                    "retrieval_mode": "db_contract_lexical",
                    "fallback_reason": "empty_query",
                    "requested_top_k": size,
                    "returned_count": 0,
                    "matched_signals": [],
                    "applied_filters": {},
                    "contract_source": None,
                },
            )

        retrieval_filters = None
        if isinstance(parsed_filters, dict) and parsed_filters:
            allowed = {f.name for f in fields(RetrievalQueryFilters)}
            sanitized = {k: v for k, v in parsed_filters.items() if k in allowed}
            if sanitized:
                retrieval_filters = RetrievalQueryFilters(**sanitized)

        parsed_query = parse_user_query(query_text)
        hard = parsed_query.hard_filters
        normalized_query = normalize_query_pipeline(query_text).strip().lower()
        price_ranking_order: str | None = None
        if any(term in normalized_query for term in ["re nhat", "thap nhat"]):
            price_ranking_order = "asc"
        elif any(term in normalized_query for term in ["cao nhat", "dat nhat"]):
            price_ranking_order = "desc"

        center_intent = any(
            term in normalized_query
            for term in [
                "gan trung tam",
                "gần trung tâm",
                "trung tam",
                "trung tâm",
                "sat trung tam",
                "sát trung tâm",
                "cbd",
            ]
        )
        has_explicit_district_scope = any(
            bool(str(value or "").strip())
            for value in (hard.district, hard.ward, hard.street, hard.project)
        )
        fetch_size = min(30, max(size, size * 4)) if center_intent and not has_explicit_district_scope else size

        retrieved = self.retrieval_service.search_listings(
            query=query_text,
            top_k=fetch_size,
            filters=retrieval_filters,
        )

        center_priority_applied = False
        boosted_items = list(retrieved.items or [])

        if center_intent and not has_explicit_district_scope and boosted_items and price_ranking_order is None:
            center_priority_applied = True
            city_hint = hard.city
            boosted: List[Dict[str, Any]] = []
            for item in boosted_items:
                district = str(item.get("district") or item.get("area") or "").strip()
                if not district:
                    boosted.append(item)
                    continue

                item_city = str(item.get("city") or city_hint or "").strip() or None
                centrality_score = self._location_centrality_score(district=district, city=item_city)
                boosted_item = dict(item)
                base_score = float(boosted_item.get("final_score") or boosted_item.get("score") or 0.0)
                matched_soft_preferences = self._soft_preference_signals_for_item(boosted_item, parsed_query.soft_preferences)
                soft_boost = self._soft_preference_boost(matched_soft_preferences)
                boost = round(centrality_score * self._SEARCH_CENTER_PRIORITY_BOOST_WEIGHT, 3)
                boosted_score = round(min(1.0, base_score + soft_boost + boost), 3)
                boosted_item["centrality_score"] = round(centrality_score, 3)
                boosted_item["centrality_boost"] = boost
                if matched_soft_preferences:
                    boosted_item["matched_soft_preferences"] = matched_soft_preferences
                    boosted_item["soft_preference_bonus"] = soft_boost
                boosted_item["final_score"] = boosted_score
                boosted_item["score"] = boosted_score
                boosted.append(boosted_item)

            boosted.sort(
                key=lambda item: (
                    -float(item.get("final_score") or 0.0),
                    -float(item.get("centrality_score") or 0.0),
                    str(item.get("listing_id") or ""),
                )
            )
            boosted_items = boosted[:size]
        else:
            boosted_items = boosted_items[:size]

        if boosted_items:
            soft_fields = (
                "near_metro",
                "near_school",
                "quiet_area",
                "family_friendly",
                "wants_gym",
                "wants_pool",
                "view",
                "nearby_transport",
                "nearby_landmarks",
                "nearby_roads",
            )
            has_soft_intent = any(bool(getattr(parsed_query.soft_preferences, field, False)) for field in soft_fields)
            if has_soft_intent:
                enriched_items: List[Dict[str, Any]] = []
                for item in boosted_items:
                    enriched_item = dict(item)
                    matched_soft_preferences = self._soft_preference_signals_for_item(enriched_item, parsed_query.soft_preferences)
                    if matched_soft_preferences:
                        soft_boost = self._soft_preference_boost(matched_soft_preferences)
                        base_score = float(enriched_item.get("final_score") or enriched_item.get("score") or 0.0)
                        boosted_score = round(min(1.0, base_score + soft_boost), 3)
                        enriched_item["matched_soft_preferences"] = matched_soft_preferences
                        enriched_item["soft_preference_bonus"] = soft_boost
                        enriched_item["final_score"] = boosted_score
                        enriched_item["score"] = boosted_score
                    enriched_items.append(enriched_item)

                if center_priority_applied:
                    enriched_items.sort(
                        key=lambda item: (
                            -float(item.get("final_score") or 0.0),
                            -float(item.get("centrality_score") or 0.0),
                            str(item.get("listing_id") or ""),
                        )
                    )
                else:
                    enriched_items.sort(
                        key=lambda item: (
                            -float(item.get("final_score") or 0.0),
                            str(item.get("listing_id") or ""),
                        )
                    )
                boosted_items = enriched_items[:size]

        if price_ranking_order is not None and boosted_items:
            def _price_sort_key(item: Dict[str, Any]) -> tuple[int, float, float, str]:
                raw_price = item.get("price_value_vnd")
                try:
                    price_value = float(raw_price)
                    has_price = 0
                except (TypeError, ValueError):
                    price_value = 0.0
                    has_price = 1

                raw_score = item.get("final_score") or item.get("score") or 0.0
                try:
                    score_value = float(raw_score)
                except (TypeError, ValueError):
                    score_value = 0.0

                ordered_price = price_value if price_ranking_order == "asc" else -price_value
                return (
                    has_price,
                    ordered_price,
                    -score_value,
                    str(item.get("listing_id") or ""),
                )

            boosted_items.sort(key=_price_sort_key)

        self.logger.info(
            "db_contract_search mode=%s count=%d",
            retrieved.retrieval_stats.retrieval_mode,
            len(boosted_items),
        )
        stats = retrieved.retrieval_stats.to_dict()
        if center_priority_applied:
            stats["center_priority_applied"] = True
            stats["center_priority_boost_weight"] = self._SEARCH_CENTER_PRIORITY_BOOST_WEIGHT
        if price_ranking_order is not None:
            stats["price_ranking_applied"] = True
            stats["price_ranking_order"] = price_ranking_order
        return SearchResult(
            items=boosted_items,
            retrieval_stats=stats,
        )

    def similar_listings(
        self,
        source: str,
        listing_id: str,
        context_query: str | None = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        return self.retrieval_service.similar_listings(
            source=source,
            listing_id=listing_id,
            context_query=context_query,
            top_k=top_k,
        )

    def explain_listing(self, source: str, listing_id: str, user_query: str = "") -> ExplainResult:
        parsed = parse_user_query(user_query or "")
        hard = parsed.hard_filters
        soft = parsed.soft_preferences

        with self._connect() as conn:
            for table_name in self.SOURCE_CANDIDATES:
                try:
                    record = self._get_listing_from_table(conn, table_name, source=source, listing_id=listing_id)
                except psycopg.errors.UndefinedTable:
                    conn.rollback()
                    continue

                if not record:
                    continue

                reasons: List[str] = []
                soft_reasons: List[str] = []
                enrichment_matches: List[str] = []

                district = self._normalize_text(record.get("district"))
                property_type = self._normalize_text(record.get("property_type"))
                if district and hard.district and hard.district in district:
                    reasons.append("district_match")
                if property_type and hard.property_type and hard.property_type in property_type:
                    reasons.append("property_type_match")

                price_value = self._safe_float(record.get("price_value_vnd"))
                area_m2 = self._safe_float(record.get("area_m2"))
                base_score = 0.0
                if price_value is not None:
                    base_score += 0.5
                if area_m2 is not None:
                    base_score += 0.5

                search_doc_raw = str(record.get("search_document") or "")
                search_doc = self._normalize_text(search_doc_raw)
                if soft.near_metro and any(k in search_doc for k in ["metro", "ga", "tram", "tau"]):
                    soft_reasons.append("near_metro")
                    enrichment_matches.append("transport_access")
                if soft.near_school and any(k in search_doc for k in ["truong", "school", "mam non"]):
                    soft_reasons.append("near_school")
                    enrichment_matches.append("school_access")
                if soft.wants_gym and "gym" in search_doc:
                    soft_reasons.append("wants_gym")
                    enrichment_matches.append("gym")
                if soft.wants_pool and any(k in search_doc for k in ["ho boi", "be boi", "pool"]):
                    soft_reasons.append("wants_pool")
                    enrichment_matches.append("pool")

                intent_bonus = 0.0
                if soft_reasons:
                    intent_bonus = min(0.25, 0.08 * len(soft_reasons))
                final_score = round(min(1.0, base_score + intent_bonus), 3)

                budget_cap = self._safe_float(hard.max_price_vnd)
                min_area = self._safe_float(hard.min_area_m2)
                min_bedrooms = self._safe_int(hard.min_bedrooms)
                utilities = self._extract_utilities(search_doc_raw)

                price_analysis = self._build_price_analysis(record, budget_cap)
                location_analysis = self._build_location_analysis(
                    record,
                    district_match=("district_match" in reasons),
                    destination=parsed.user_profile.commuting_destination,
                    utilities=utilities,
                )
                size_analysis = self._build_size_analysis(record, min_area=min_area, min_bedrooms=min_bedrooms)

                fit_score_breakdown = {
                    "price_fit": 1.0 if price_analysis.get("budget_fit") == "within_budget" else 0.4,
                    "location_fit": 1.0 if location_analysis.get("district_match") else 0.6,
                    "size_fit": 1.0 if size_analysis.get("meets_area_requirement") and size_analysis.get("meets_bedroom_requirement") else 0.5,
                    "intent_bonus": round(intent_bonus, 3),
                    "overall": final_score,
                }

                structured_analysis = {
                    "price_analysis": price_analysis,
                    "location_analysis": location_analysis,
                    "size_analysis": size_analysis,
                    "legal_analysis": {
                        "legal_status": record.get("legal_status"),
                        "is_legal_info_available": bool(record.get("legal_status")),
                    },
                    "utilities": utilities,
                    "fit_score_breakdown": fit_score_breakdown,
                }

                record["matched_hard_filters"] = reasons
                record["matched_soft_preferences"] = soft_reasons
                record["enrichment_matches"] = enrichment_matches
                record["similarity_summary"] = "Listing explained with hard+soft signal matching"
                record["final_score"] = final_score
                record["base_score"] = round(base_score, 3)
                record["intent_bonus"] = round(intent_bonus, 3)
                record["use_case"] = parsed.use_case or "listing_explanation"
                record["message"] = "Listing explained"
                # Structured groups for phase 2a while preserving old keys above.
                record["price_analysis"] = price_analysis
                record["location_analysis"] = location_analysis
                record["size_analysis"] = size_analysis
                record["legal_analysis"] = structured_analysis["legal_analysis"]
                record["utilities"] = utilities
                record["fit_score_breakdown"] = fit_score_breakdown
                record["analysis_structured"] = structured_analysis

                return ExplainResult(
                    found=True,
                    listing=record,
                    message="Listing explained",
                    analysis=structured_analysis,
                )

        return ExplainResult(
            found=False,
            listing={},
            message="Listing not found",
            analysis={},
        )

    def compare_listings(
        self,
        source_a: str,
        listing_id_a: str,
        source_b: str,
        listing_id_b: str,
        user_query: str | None = None,
        user_profile: Dict[str, Any] | None = None,
    ) -> CompareResult:
        with self._connect() as conn:
            listing_a: Dict[str, Any] | None = None
            listing_b: Dict[str, Any] | None = None
            for table_name in self.SOURCE_CANDIDATES:
                try:
                    if listing_a is None:
                        listing_a = self._get_listing_from_table(conn, table_name, source=source_a, listing_id=listing_id_a)
                    if listing_b is None:
                        listing_b = self._get_listing_from_table(conn, table_name, source=source_b, listing_id=listing_id_b)
                except psycopg.errors.UndefinedTable:
                    conn.rollback()
                    continue
                if listing_a and listing_b:
                    break

        if not listing_a or not listing_b:
            return CompareResult(
                found=False,
                listing_a=listing_a or {},
                listing_b=listing_b or {},
                recommendation={"winner": None, "winner_ref": None, "summary": "Missing listing(s)"},
                message="Cannot compare because one or both listings were not found",
            )

        def _price_per_m2(item: Dict[str, Any]) -> float | None:
            price = self._safe_float(item.get("price_value_vnd"))
            area = self._safe_float(item.get("area_m2"))
            if price is None or area is None or area <= 0:
                return None
            return price / area

        score_a, reasons_a = self._compute_compare_fit_score(listing_a, user_query)
        score_b, reasons_b = self._compute_compare_fit_score(listing_b, user_query)
        profile_breakdown_a: Dict[str, Any] | None = None
        profile_breakdown_b: Dict[str, Any] | None = None
        profile_reasons_a: List[str] | None = None
        profile_reasons_b: List[str] | None = None

        if user_profile:
            score_a, profile_breakdown_a, profile_reasons_a = self._compute_profile_fit_score(listing_a, user_profile)
            score_b, profile_breakdown_b, profile_reasons_b = self._compute_profile_fit_score(listing_b, user_profile)
            reasons_a = profile_reasons_a or reasons_a
            reasons_b = profile_reasons_b or reasons_b

        ppm_a = _price_per_m2(listing_a)
        ppm_b = _price_per_m2(listing_b)

        winner = "tie"
        winner_ref = None
        summary = "Two listings are tied with current information"

        if abs(score_a - score_b) >= (0.05 if user_profile else 0.5):
            if score_a > score_b:
                winner = "A"
                winner_ref = f"{source_a}/{listing_id_a}"
            else:
                winner = "B"
                winner_ref = f"{source_b}/{listing_id_b}"
            summary = (
                f"Winner selected by user-profile fit score ({winner})"
                if user_profile
                else f"Winner selected by intent-aware fit score ({winner})"
            )
        elif ppm_a is not None and ppm_b is not None and abs(ppm_a - ppm_b) > 1 and not user_profile:
            if ppm_a < ppm_b:
                winner = "A"
                winner_ref = f"{source_a}/{listing_id_a}"
            else:
                winner = "B"
                winner_ref = f"{source_b}/{listing_id_b}"
            summary = f"Winner selected by lower price per m2 ({winner})"
        else:
            bedroom_a = self._safe_int(listing_a.get("bedrooms")) or 0
            bedroom_b = self._safe_int(listing_b.get("bedrooms")) or 0
            if bedroom_a > bedroom_b:
                winner = "A"
                winner_ref = f"{source_a}/{listing_id_a}"
                summary = "Winner selected by higher bedroom count (A)"
            elif bedroom_b > bedroom_a:
                winner = "B"
                winner_ref = f"{source_b}/{listing_id_b}"
                summary = "Winner selected by higher bedroom count (B)"

        recommendation = {
            "winner": winner,
            "winner_ref": winner_ref,
            "summary": summary,
            "criterion": "user_profile_weighted_fit" if user_profile else "intent_fit_then_price_per_m2_then_bedrooms",
            "context_query": user_query,
            "fit_score_a": round(score_a, 3),
            "fit_score_b": round(score_b, 3),
            "fit_reasons_a": reasons_a,
            "fit_reasons_b": reasons_b,
            "user_profile_used": bool(user_profile),
            "profile_breakdown_a": profile_breakdown_a,
            "profile_breakdown_b": profile_breakdown_b,
        }

        return CompareResult(
            found=True,
            listing_a=listing_a,
            listing_b=listing_b,
            recommendation=recommendation,
            message=summary,
        )

    def suggest_area(self, query: str, top_k: int = 5) -> SuggestAreaResult:
        parsed = parse_user_query(query or "")
        missing_fields: List[str] = []
        next_user_action = "choose_area"
        normalized_query = normalize_query_pipeline(str(query or "")).strip().lower()
        center_intent = any(
            term in normalized_query
            for term in ["gan trung tam", "gần trung tâm", "trung tam", "trung tâm", "sat trung tam", "sát trung tâm", "cbd"]
        )

        family_signal = any(
            [
                parsed.user_profile.family_size is not None,
                parsed.user_profile.has_children,
                parsed.user_profile.has_elderly,
                parsed.soft_preferences.family_friendly,
            ]
        )
        if not parsed.user_profile.commuting_destination:
            missing_fields.append("commuting_destination")
        if parsed.hard_filters.max_price_vnd is None:
            missing_fields.append("budget")
        if not family_signal:
            missing_fields.append("family_profile")

        budget_vnd = int(parsed.hard_filters.max_price_vnd or 0)
        city = str(parsed.hard_filters.city or "").strip() or None
        property_type = str(parsed.hard_filters.property_type or "").strip() or None
        destination_norm = self._normalize_text(parsed.user_profile.commuting_destination)
        base_candidates = self._build_area_candidates(
            destination_norm=destination_norm,
            family_signal=family_signal,
            has_children=parsed.user_profile.has_children,
            has_elderly=parsed.user_profile.has_elderly,
            budget_vnd=budget_vnd,
            property_type=property_type,
            city=city,
            query_text=query,
        )
        recommendations = self._rank_area_candidates(
            base_candidates=base_candidates,
            query_text=query or "",
            destination_norm=destination_norm,
            budget_vnd=budget_vnd,
            family_signal=family_signal,
            property_type=property_type,
            city=city,
            near_metro=bool(getattr(parsed.soft_preferences, "near_metro", False)),
        )[: max(1, min(20, int(top_k)))]

        if center_intent:
            recommendations = self._build_semantic_area_fallback(
                query_text=query or "",
                top_k=top_k,
                budget_vnd=budget_vnd,
                property_type=property_type,
                destination_norm=destination_norm,
                family_signal=family_signal,
                city=city,
            )

        if not recommendations:
            recommendations = self._build_semantic_area_fallback(
                query_text=query or "",
                top_k=top_k,
                budget_vnd=budget_vnd,
                property_type=property_type,
                destination_norm=destination_norm,
                family_signal=family_signal,
                city=city,
            )

        enriched_recommendations: List[Dict[str, Any]] = []
        def _format_budget_text(value: Any) -> str | None:
            try:
                amount = int(value)
            except (TypeError, ValueError):
                return None
            if amount <= 0:
                return None
            if amount >= 1_000_000_000:
                return f"{amount / 1_000_000_000:.1f} tỷ"
            return f"{amount / 1_000_000:.0f} triệu"

        for row in recommendations:
            district = str(row.get("district") or "").strip()
            if not district:
                continue
            snapshot = self._build_district_market_snapshot(district=district, city=city, budget_vnd=budget_vnd)
            combined = dict(row)
            combined.update(snapshot)
            if combined.get("common_property"):
                combined.setdefault("matching_reasons", [])
                reasons = list(combined.get("matching_reasons") or [])
                reasons.append(f"Loai hinh pho bien: {combined.get('common_property')}")
                combined["matching_reasons"] = list(dict.fromkeys(str(item) for item in reasons if str(item).strip()))[:4]
            estimated_price_vnd = combined.get("avg_price_vnd") or combined.get("max_price_vnd") or combined.get("min_price_vnd")
            reason_parts: list[str] = []
            for item in list(combined.get("matching_reasons") or [])[:3]:
                text = str(item or "").strip()
                if text and text not in reason_parts:
                    reason_parts.append(text)
            if combined.get("market_comment"):
                market_comment = str(combined.get("market_comment") or "").strip()
                if market_comment and market_comment not in reason_parts:
                    reason_parts.append(market_comment)
            if combined.get("area_comment"):
                area_comment = str(combined.get("area_comment") or "").strip()
                if area_comment and area_comment not in reason_parts:
                    reason_parts.append(area_comment)
            ranking_row = {
                "area": district,
                "district": district,
                "score": round(float(combined.get("score") or 0.0), 3),
                "reason": "; ".join(reason_parts[:3]),
                "matching_reasons": reason_parts[:4],
                "estimated_price": _format_budget_text(estimated_price_vnd),
                "estimated_price_vnd": int(estimated_price_vnd) if estimated_price_vnd is not None else None,
                "inventory_level": int(combined.get("listing_count") or 0),
                "listing_count": int(combined.get("listing_count") or 0),
                "commute_minutes": combined.get("commute_minutes"),
                "market_comment": combined.get("market_comment"),
                "area_comment": combined.get("area_comment"),
                "common_property": combined.get("common_property"),
                "price_range_text": combined.get("price_range_text"),
                "rank": combined.get("rank"),
            }
            combined.update(ranking_row)
            enriched_recommendations.append(combined)

        recommendations = enriched_recommendations

        # Prefer progressive follow-up over hard-blocking the user flow.
        need_clarification = not bool(recommendations)

        if missing_fields:
            clarify_parts: List[str] = []
            if "commuting_destination" in missing_fields:
                clarify_parts.append("anh/chị muốn ưu tiên khu nào hoặc đi làm ở đâu")
            if "budget" in missing_fields:
                clarify_parts.append("ngân sách dự kiến khoảng bao nhiêu")
            if "family_profile" in missing_fields:
                clarify_parts.append("anh/chị mua để ở hay đầu tư, và gia đình có mấy người")
            if "property_type" in missing_fields:
                clarify_parts.append("anh/chị ưu tiên căn hộ, nhà phố hay đất")

            if clarify_parts:
                next_prompt = (
                    "Để mình tư vấn sát hơn, anh/chị cho mình thêm một chút thông tin: "
                    + "; ".join(clarify_parts)
                    + "."
                )
            else:
                next_prompt = (
                    "Để mình tư vấn sát hơn, anh/chị cho mình thêm thông tin về khu vực ưu tiên, ngân sách và nhu cầu sử dụng nhé."
                )
        elif not recommendations:
            next_prompt = (
                "Các khu đang có quá ít listing phù hợp để xếp hạng chắc tay. Nếu anh/chị nới khu vực, đổi loại hình hoặc mở rộng ngân sách, mình sẽ lọc lại ngay."
            )
        elif recommendations:
            top_area = recommendations[0]
            next_prompt = (
                f"Mình đang nghiêng về khu {top_area.get('area')}. "
                "Nếu anh/chị chọn một khu trong danh sách, mình sẽ lọc listing phù hợp ngay."
            )
        else:
            next_prompt = "Anh/chị có thể mở rộng ngân sách hoặc khu vực mong muốn để mình đề xuất thêm lựa chọn phù hợp."

        if need_clarification:
            return SuggestAreaResult(
                need_clarification=True,
                missing_fields=missing_fields,
                area_recommendations=recommendations,
                summary="Không đủ dữ liệu listing phù hợp để recommendation.",
                next_clarification_prompt=next_prompt,
            )

        return SuggestAreaResult(
            need_clarification=False,
            missing_fields=missing_fields,
            area_recommendations=recommendations,
            summary=(
                "Area suggestion generated from intent-aware rules"
                if not missing_fields
                else "Area suggestion generated with partial profile"
            ),
            next_clarification_prompt=next_prompt,
            suggested_next_tool="search_listings" if recommendations else None,
            next_user_action=next_user_action if recommendations else None,
        )

    def get_listing_statistics(self, query: str) -> Dict[str, Any]:
        """
        Get aggregate statistics (count, avg price, etc.) for listings matching the query.
        
        Supports queries like:
        - "Bao nhieu listing o Thu Duc" -> count
        - "Gia trung binh o Binh Thanh" -> avg_price_vnd
        - "Top 3 quận" -> district ranking (routes to get_district_rankings)
        - "Giá/m2 trung bình" -> avg price per m2 (routes to get_avg_price_per_m2_by_district)
        - "Phân bố theo loại" -> property type breakdown (routes to get_property_type_breakdown)
        """
        try:
            parsed = parse_user_query(query or "")
            hard = parsed.hard_filters
            qn = str(parsed.normalized_query or "")
            metric_type = self._detect_analytics_metric_from_query(qn)
            
            # Route to specialized functions based on metric type
            if metric_type == "district_ranking":
                return {
                    **self.get_district_rankings(query, limit=3),
                    "metric_type": metric_type
                }

            if metric_type == "ward_ranking":
                return {
                    **self.get_ward_rankings(query, limit=3),
                    "metric_type": metric_type,
                }
            
            if metric_type == "avg_price_per_m2":
                return {
                    **self.get_avg_price_per_m2_by_district(query),
                    "metric_type": metric_type
                }

            if metric_type == "district_compare_avg_price":
                return {
                    **self.get_district_avg_price_comparison(query),
                    "metric_type": metric_type,
                }
            
            if metric_type == "property_type_grouping":
                return {
                    **self.get_property_type_breakdown(query),
                    "metric_type": metric_type
                }
            
            # Default analytics path
            districts = self._extract_districts_from_normalized_query(qn)
            if hard.district and hard.district not in districts:
                districts.insert(0, hard.district)
            
            # Build WHERE conditions
            where = ["search_document IS NOT NULL", "BTRIM(search_document) <> ''"]
            params: List[Any] = []
            
            if hard.transaction_type:
                aliases = self._expand_transaction_type_aliases(hard.transaction_type)
                placeholders = ", ".join(["%s"] * len(aliases))
                where.append(f"transaction_type ILIKE ANY(ARRAY[{placeholders}])")
                params.extend([f"%{alias}%" for alias in aliases])
            if hard.property_type:
                aliases = self._expand_property_type_aliases(hard.property_type)
                placeholders = ", ".join(["%s"] * len(aliases))
                where.append(f"property_type ILIKE ANY(ARRAY[{placeholders}])")
                params.extend([f"%{alias}%" for alias in aliases])
            if hard.city:
                city_aliases = self._expand_city_aliases(hard.city)
                placeholders = ", ".join(["%s"] * len(city_aliases))
                where.append(f"city ILIKE ANY(ARRAY[{placeholders}])")
                params.extend([f"%{alias}%" for alias in city_aliases])
            if districts:
                district_clauses: List[str] = []
                for district in districts:
                    aliases = self._expand_district_aliases(district)
                    if not aliases:
                        continue
                    placeholders = ", ".join(["%s"] * len(aliases))
                    district_clauses.append(f"district ILIKE ANY(ARRAY[{placeholders}])")
                    params.extend([f"%{alias}%" for alias in aliases])
                if district_clauses:
                    where.append("(" + " OR ".join(district_clauses) + ")")
            if hard.max_price_vnd is not None:
                where.append("price_value_vnd <= %s")
                params.append(int(hard.max_price_vnd))
            if hard.min_price_vnd is not None:
                where.append("price_value_vnd >= %s")
                params.append(int(hard.min_price_vnd))
            if hard.min_area_m2 is not None:
                where.append("area_m2 >= %s")
                params.append(float(hard.min_area_m2))
            if hard.max_area_m2 is not None:
                where.append("area_m2 <= %s")
                params.append(float(hard.max_area_m2))
            if hard.min_bedrooms is not None:
                where.append("bedrooms IS NOT NULL AND bedrooms >= %s")
                params.append(int(hard.min_bedrooms))

            self._append_soft_preference_filters(where, params, parsed)
            
            where_clause = " AND ".join(where) if where else "TRUE"
            
            sql = f"""
                SELECT
                    COUNT(*) as total_count,
                    AVG(price_value_vnd)::BIGINT as avg_price_vnd,
                    MIN(price_value_vnd)::BIGINT as min_price_vnd,
                    MAX(price_value_vnd)::BIGINT as max_price_vnd,
                    AVG(area_m2)::FLOAT as avg_area_m2,
                    AVG(
                        CASE
                            WHEN price_value_vnd IS NOT NULL AND area_m2 IS NOT NULL AND area_m2 > 0
                            THEN price_value_vnd / area_m2
                            ELSE NULL
                        END
                    )::FLOAT as avg_price_per_m2_vnd
                FROM listings
                WHERE {where_clause}
            """
            
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    row = cur.fetchone()
                    if not row:
                        row = (0, None, None, None, None, None)
                    
                    cols = [
                        "total_count",
                        "avg_price_vnd",
                        "min_price_vnd",
                        "max_price_vnd",
                        "avg_area_m2",
                        "avg_price_per_m2_vnd",
                    ]
                    result = dict(zip(cols, row))

                    district_breakdown: List[Dict[str, Any]] = []
                    ward_breakdown: List[Dict[str, Any]] = []
                    wants_district_view = self._query_requests_district_view(qn)
                    wants_ward_view = self._query_requests_ward_view(qn)
                    area_distribution_requested = wants_district_view or wants_ward_view
                    if len(districts) > 1:
                        district_case_clauses: List[str] = []
                        district_case_params: List[Any] = []
                        for district in districts:
                            aliases = self._expand_district_aliases(district)
                            if not aliases:
                                continue
                            placeholders = ", ".join(["%s"] * len(aliases))
                            district_case_clauses.append(
                                f"WHEN district ILIKE ANY(ARRAY[{placeholders}]) THEN %s"
                            )
                            district_case_params.extend([f"%{alias}%" for alias in aliases])
                            district_case_params.append(district)

                        if district_case_clauses:
                            breakdown_sql = f"""
                                SELECT
                                    district_bucket,
                                    COUNT(*) AS district_count
                                FROM (
                                    SELECT
                                        CASE
                                            {' '.join(district_case_clauses)}
                                            ELSE NULL
                                        END AS district_bucket
                                    FROM listings
                                    WHERE {where_clause}
                                ) t
                                WHERE district_bucket IS NOT NULL
                                GROUP BY district_bucket
                                ORDER BY district_count DESC, district_bucket ASC
                            """
                            cur.execute(breakdown_sql, district_case_params + params)
                            district_breakdown = [
                                {"district": r[0], "count": int(r[1] or 0)}
                                for r in cur.fetchall() or []
                            ]
                    elif wants_district_view:
                        # Generic district distribution for overview queries like
                        # "phan tich gia ... theo/cac quan" where no explicit district list is provided.
                        district_limit = 80 if hard.city else 20
                        overview_breakdown_sql = f"""
                            SELECT
                                BTRIM(district) AS district_bucket,
                                COUNT(*) AS district_count,
                                AVG(price_value_vnd)::BIGINT AS avg_price_vnd,
                                AVG(price_value_vnd / NULLIF(area_m2, 0))::FLOAT AS avg_price_per_m2_vnd
                            FROM listings
                            WHERE {where_clause}
                              AND district IS NOT NULL
                              AND BTRIM(district) <> ''
                            GROUP BY BTRIM(district)
                            ORDER BY district_count DESC, avg_price_vnd DESC
                            LIMIT %s
                        """
                        cur.execute(overview_breakdown_sql, params + [district_limit])
                        district_breakdown = [
                            {
                                "district": r[0],
                                "count": int(r[1] or 0),
                                "avg_price_vnd": r[2],
                                "avg_price_per_m2_vnd": r[3],
                            }
                            for r in (cur.fetchall() or [])
                        ]

                    # For district-scoped queries, provide ward-level distribution only when requested.
                    if len(districts) == 1 and wants_ward_view:
                        ward_sql = f"""
                            SELECT
                                BTRIM(ward) AS ward_bucket,
                                COUNT(*) AS ward_count,
                                AVG(price_value_vnd)::BIGINT AS avg_price_vnd,
                                AVG(price_value_vnd / NULLIF(area_m2, 0))::FLOAT AS avg_price_per_m2_vnd
                            FROM listings
                            WHERE {where_clause}
                              AND ward IS NOT NULL
                              AND BTRIM(ward) <> ''
                            GROUP BY BTRIM(ward)
                            ORDER BY ward_count DESC, avg_price_vnd DESC
                            LIMIT 30
                        """
                        cur.execute(ward_sql, params)
                        ward_breakdown = [
                            {
                                "ward": r[0],
                                "count": int(r[1] or 0),
                                "avg_price_vnd": r[2],
                                "avg_price_per_m2_vnd": r[3],
                            }
                            for r in (cur.fetchall() or [])
                        ]

                    # City-level fallback only for explicit ward-distribution intent.
                    if hard.city and wants_ward_view and not district_breakdown and not ward_breakdown:
                        ward_fallback_sql = f"""
                            SELECT
                                BTRIM(ward) AS ward_bucket,
                                COUNT(*) AS ward_count,
                                AVG(price_value_vnd)::BIGINT AS avg_price_vnd,
                                AVG(price_value_vnd / NULLIF(area_m2, 0))::FLOAT AS avg_price_per_m2_vnd
                            FROM listings
                            WHERE {where_clause}
                              AND ward IS NOT NULL
                              AND BTRIM(ward) <> ''
                            GROUP BY BTRIM(ward)
                            ORDER BY ward_count DESC, avg_price_vnd DESC
                            LIMIT 80
                        """
                        cur.execute(ward_fallback_sql, params)
                        ward_breakdown = [
                            {
                                "ward": r[0],
                                "count": int(r[1] or 0),
                                "avg_price_vnd": r[2],
                                "avg_price_per_m2_vnd": r[3],
                            }
                            for r in (cur.fetchall() or [])
                        ]

                    max_price_per_m2_vnd = None
                    max_price_per_m2_listing: Dict[str, Any] | None = None
                    if metric_type == "max_price_per_m2":
                        ppm_sql = f"""
                            SELECT
                                source,
                                listing_id,
                                title,
                                url,
                                district,
                                city,
                                price_value_vnd,
                                area_m2,
                                (price_value_vnd / NULLIF(area_m2, 0))::FLOAT AS price_per_m2_vnd
                            FROM listings
                            WHERE {where_clause}
                              AND price_value_vnd IS NOT NULL
                              AND area_m2 IS NOT NULL
                              AND area_m2 > 0
                            ORDER BY price_per_m2_vnd DESC
                            LIMIT 1
                        """
                        cur.execute(ppm_sql, params)
                        ppm_row = cur.fetchone()
                        if ppm_row:
                            max_price_per_m2_listing = {
                                "source": ppm_row[0],
                                "listing_id": ppm_row[1],
                                "title": ppm_row[2],
                                "url": ppm_row[3],
                                "district": ppm_row[4],
                                "city": ppm_row[5],
                                "price_value_vnd": ppm_row[6],
                                "area_m2": ppm_row[7],
                                "price_per_m2_vnd": ppm_row[8],
                            }
                            max_price_per_m2_vnd = ppm_row[8]

                    insights: Dict[str, Any] = {
                        "property_mix": {},
                        "legal": {},
                        "infrastructure": {},
                        "amenities": {},
                        "user_fit": {},
                    }

                    total_count_value = int(result.get("total_count") or 0)
                    if total_count_value > 0:
                        # Property mix from actual filtered inventory.
                        mix_sql = f"""
                            SELECT
                                COALESCE(NULLIF(BTRIM(property_type), ''), 'Khac') AS property_type_bucket,
                                COUNT(*) AS cnt
                            FROM listings
                            WHERE {where_clause}
                            GROUP BY property_type_bucket
                            ORDER BY cnt DESC
                            LIMIT 4
                        """
                        cur.execute(mix_sql, params)
                        mix_rows = cur.fetchall() or []
                        top_types = [
                            {
                                "name": str(r[0] or "Khac").strip() or "Khac",
                                "count": int(r[1] or 0),
                                "share_pct": round((int(r[1] or 0) * 100.0) / total_count_value, 1),
                            }
                            for r in mix_rows
                        ]
                        if top_types:
                            insights["property_mix"] = {
                                "top_types": top_types,
                                "dominant_type": top_types[0]["name"],
                            }

                        # Legal transparency and top legal statuses.
                        legal_count_sql = f"""
                            SELECT
                                COUNT(*) FILTER (WHERE legal_status IS NOT NULL AND BTRIM(legal_status) <> '') AS legal_known_count
                            FROM listings
                            WHERE {where_clause}
                        """
                        cur.execute(legal_count_sql, params)
                        legal_known_count = int((cur.fetchone() or [0])[0] or 0)

                        legal_top_sql = f"""
                            SELECT
                                BTRIM(legal_status) AS legal_status_bucket,
                                COUNT(*) AS cnt
                            FROM listings
                            WHERE {where_clause}
                              AND legal_status IS NOT NULL
                              AND BTRIM(legal_status) <> ''
                            GROUP BY legal_status_bucket
                            ORDER BY cnt DESC
                            LIMIT 3
                        """
                        cur.execute(legal_top_sql, params)
                        legal_rows = cur.fetchall() or []
                        insights["legal"] = {
                            "known_count": legal_known_count,
                            "known_ratio_pct": round((legal_known_count * 100.0) / total_count_value, 1),
                            "top_statuses": [
                                {"name": str(r[0] or "").strip(), "count": int(r[1] or 0)}
                                for r in legal_rows
                                if str(r[0] or "").strip()
                            ],
                        }

                        # Infrastructure and amenities coverage from enrichment columns and search_document.
                        coverage_sql = f"""
                            SELECT
                                COUNT(*) FILTER (
                                    WHERE (nearby_transport IS NOT NULL AND BTRIM(nearby_transport) <> '')
                                        OR search_document ILIKE '%%metro%%'
                                        OR search_document ILIKE '%%bus%%'
                                        OR search_document ILIKE '%%ben xe%%'
                                ) AS transport_count,
                                COUNT(*) FILTER (
                                    WHERE (nearby_roads IS NOT NULL AND BTRIM(nearby_roads) <> '')
                                       OR (access IS NOT NULL AND BTRIM(access) <> '')
                                       OR road_access_width_m IS NOT NULL
                                ) AS road_access_count,
                                COUNT(*) FILTER (
                                    WHERE (amenities_area IS NOT NULL AND BTRIM(amenities_area) <> '')
                                       OR (amenities_building IS NOT NULL AND BTRIM(amenities_building) <> '')
                                                    OR search_document ILIKE '%%tien ich%%'
                                ) AS amenities_count,
                                COUNT(*) FILTER (
                                                WHERE (suitable_for IS NOT NULL AND suitable_for ILIKE '%%gia dinh%%')
                                                    OR search_document ILIKE '%%gia dinh%%'
                                ) AS family_fit_count,
                                COUNT(*) FILTER (
                                                WHERE (suitable_for IS NOT NULL AND suitable_for ILIKE '%%dau tu%%')
                                                    OR search_document ILIKE '%%cho thue%%'
                                                    OR search_document ILIKE '%%dong tien%%'
                                ) AS investment_fit_count
                            FROM listings
                            WHERE {where_clause}
                        """
                        cur.execute(coverage_sql, params)
                        coverage_row = cur.fetchone() or (0, 0, 0, 0, 0)
                        transport_count = int(coverage_row[0] or 0)
                        road_access_count = int(coverage_row[1] or 0)
                        amenities_count = int(coverage_row[2] or 0)
                        family_fit_count = int(coverage_row[3] or 0)
                        investment_fit_count = int(coverage_row[4] or 0)

                        top_transport_signal = ""
                        top_amenity_signal = ""
                        top_suitable_for_signal = ""

                        transport_top_sql = f"""
                            SELECT BTRIM(nearby_transport) AS signal, COUNT(*) AS cnt
                            FROM listings
                            WHERE {where_clause}
                              AND nearby_transport IS NOT NULL
                              AND BTRIM(nearby_transport) <> ''
                            GROUP BY signal
                            ORDER BY cnt DESC
                            LIMIT 1
                        """
                        cur.execute(transport_top_sql, params)
                        transport_row = cur.fetchone()
                        if transport_row and str(transport_row[0] or "").strip():
                            top_transport_signal = str(transport_row[0]).strip()

                        amenity_top_sql = f"""
                            SELECT signal, COUNT(*) AS cnt
                            FROM (
                                SELECT BTRIM(amenities_area) AS signal
                                FROM listings
                                WHERE {where_clause}
                                  AND amenities_area IS NOT NULL
                                  AND BTRIM(amenities_area) <> ''
                                UNION ALL
                                SELECT BTRIM(amenities_building) AS signal
                                FROM listings
                                WHERE {where_clause}
                                  AND amenities_building IS NOT NULL
                                  AND BTRIM(amenities_building) <> ''
                            ) t
                            GROUP BY signal
                            ORDER BY cnt DESC
                            LIMIT 1
                        """
                        cur.execute(amenity_top_sql, params + params)
                        amenity_row = cur.fetchone()
                        if amenity_row and str(amenity_row[0] or "").strip():
                            top_amenity_signal = str(amenity_row[0]).strip()

                        suitable_top_sql = f"""
                            SELECT BTRIM(suitable_for) AS signal, COUNT(*) AS cnt
                            FROM listings
                            WHERE {where_clause}
                              AND suitable_for IS NOT NULL
                              AND BTRIM(suitable_for) <> ''
                            GROUP BY signal
                            ORDER BY cnt DESC
                            LIMIT 1
                        """
                        cur.execute(suitable_top_sql, params)
                        suitable_row = cur.fetchone()
                        if suitable_row and str(suitable_row[0] or "").strip():
                            top_suitable_for_signal = str(suitable_row[0]).strip()

                        insights["infrastructure"] = {
                            "transport_coverage_pct": round((transport_count * 100.0) / total_count_value, 1),
                            "road_access_coverage_pct": round((road_access_count * 100.0) / total_count_value, 1),
                            "top_transport_signal": top_transport_signal,
                        }
                        insights["amenities"] = {
                            "coverage_pct": round((amenities_count * 100.0) / total_count_value, 1),
                            "top_amenity_signal": top_amenity_signal,
                        }
                        insights["user_fit"] = {
                            "family_friendly_pct": round((family_fit_count * 100.0) / total_count_value, 1),
                            "investment_fit_pct": round((investment_fit_count * 100.0) / total_count_value, 1),
                            "top_suitable_for_signal": top_suitable_for_signal,
                        }
            
            # Build location context string
            location_parts = []
            if districts:
                location_parts.append(", ".join(districts))
            elif hard.district:
                location_parts.append(f"{hard.district}")
            if hard.city:
                location_parts.append(f"TP {hard.city}")
            if hard.project:
                location_parts.append(f"DA {hard.project}")
            
            location_context = " ".join(location_parts) if location_parts else "Toan TP"
            
            applied_filters = {
                "distance": ", ".join(districts) if districts else (hard.district or hard.city),
                "property_type": hard.property_type,
                "transaction_type": hard.transaction_type,
                "price_max_vnd": hard.max_price_vnd,
                "area_min_m2": hard.min_area_m2,
                "area_max_m2": hard.max_area_m2,
                "bedrooms_min": hard.min_bedrooms,
                "near_metro": bool(getattr(parsed.soft_preferences, "near_metro", False)),
                "wants_gym": bool(getattr(parsed.soft_preferences, "wants_gym", False)),
                "wants_pool": bool(getattr(parsed.soft_preferences, "wants_pool", False)),
                "family_friendly": bool(getattr(parsed.soft_preferences, "family_friendly", False)),
            }
            applied_filters = {k: v for k, v in applied_filters.items() if v is not None}
            
            return {
                "total_count": result.get("total_count", 0),
                "avg_price_vnd": result.get("avg_price_vnd"),
                "min_price_vnd": result.get("min_price_vnd"),
                "max_price_vnd": result.get("max_price_vnd"),
                "avg_area_m2": result.get("avg_area_m2"),
                "avg_price_per_m2_vnd": result.get("avg_price_per_m2_vnd"),
                "metric_type": metric_type,
                "area_distribution_requested": area_distribution_requested,
                "districts": districts,
                "wards": ward_breakdown,
                "ward_breakdown": ward_breakdown,
                "district_breakdown": district_breakdown,
                "max_price_per_m2_vnd": max_price_per_m2_vnd,
                "max_price_per_m2_listing": max_price_per_m2_listing,
                "insights": insights,
                "location_context": location_context,
                "filters_applied": applied_filters,
            }
        except Exception as exc:
            self.logger.exception("get_listing_statistics failed query=%s error=%s", query, exc)
            return {
                "total_count": 0,
                "avg_price_vnd": None,
                "min_price_vnd": None,
                "max_price_vnd": None,
                "avg_area_m2": None,
                "avg_price_per_m2_vnd": None,
                "metric_type": "default",
                "area_distribution_requested": False,
                "districts": [],
                "wards": [],
                "ward_breakdown": [],
                "district_breakdown": [],
                "max_price_per_m2_vnd": None,
                "max_price_per_m2_listing": None,
                "insights": {},
                "location_context": "",
                "filters_applied": {},
                "error": str(exc),
            }

    def get_district_rankings(self, query: str, limit: int = 3) -> Dict[str, Any]:
        """
        Get top-K districts ranked by count or average price.
        Used for queries like "Top 3 quận có nhiều listing nhất"
        
        Returns:
            {
                "districts": [
                    {"district": "Quận 1", "count": 331, "avg_price_vnd": 12000000000},
                    {"district": "Quận 2", "count": 284, "avg_price_vnd": 11000000000},
                    {"district": "Quận 3", "count": 221, "avg_price_vnd": 10000000000}
                ],
                "total_count": 836
            }
        """
        try:
            parsed = parse_user_query(query or "")
            hard = parsed.hard_filters
            qn = str(parsed.normalized_query or "")
            ranking_metric, ranking_order = self._detect_ranking_metric_from_query(qn)
            ranking_min_count = 50 if ranking_metric == "avg_price_vnd" else None
            
            # Build WHERE conditions (excluding district filter)
            where = ["search_document IS NOT NULL", "BTRIM(search_document) <> ''"]
            params: List[Any] = []
            
            if hard.transaction_type:
                aliases = self._expand_transaction_type_aliases(hard.transaction_type)
                placeholders = ", ".join(["%s"] * len(aliases))
                where.append(f"transaction_type ILIKE ANY(ARRAY[{placeholders}])")
                params.extend([f"%{alias}%" for alias in aliases])
            if hard.property_type:
                aliases = self._expand_property_type_aliases(hard.property_type)
                placeholders = ", ".join(["%s"] * len(aliases))
                where.append(f"property_type ILIKE ANY(ARRAY[{placeholders}])")
                params.extend([f"%{alias}%" for alias in aliases])
            if hard.city:
                city_aliases = self._expand_city_aliases(hard.city)
                placeholders = ", ".join(["%s"] * len(city_aliases))
                where.append(f"city ILIKE ANY(ARRAY[{placeholders}])")
                params.extend([f"%{alias}%" for alias in city_aliases])
            if hard.max_price_vnd is not None:
                where.append("price_value_vnd <= %s")
                params.append(int(hard.max_price_vnd))
            if hard.min_area_m2 is not None:
                where.append("area_m2 >= %s")
                params.append(float(hard.min_area_m2))
            if hard.min_bedrooms is not None:
                where.append("bedrooms IS NOT NULL AND bedrooms >= %s")
                params.append(int(hard.min_bedrooms))

            self._append_soft_preference_filters(where, params, parsed)
            
            where_clause = " AND ".join(where) if where else "TRUE"
            
            if ranking_metric == "avg_price_vnd":
                order_clause = (
                    "avg_price_vnd ASC NULLS LAST, district_count DESC"
                    if ranking_order == "asc"
                    else "avg_price_vnd DESC NULLS LAST, district_count DESC"
                )
            else:
                order_clause = "district_count DESC, avg_price_vnd DESC"

            having_clause = ""
            ranking_params: List[Any] = []
            if ranking_min_count is not None:
                having_clause = "HAVING COUNT(*) >= %s"
                ranking_params.append(int(ranking_min_count))

            sql = f"""
                SELECT
                    district,
                    COUNT(*) as district_count,
                    AVG(price_value_vnd)::BIGINT as avg_price_vnd,
                    AVG(area_m2)::FLOAT as avg_area_m2,
                    AVG(
                        CASE
                            WHEN price_value_vnd IS NOT NULL AND area_m2 IS NOT NULL AND area_m2 > 0
                            THEN price_value_vnd / area_m2
                            ELSE NULL
                        END
                    )::FLOAT as avg_price_per_m2_vnd
                FROM listings
                WHERE {where_clause}
                  AND district IS NOT NULL
                  AND BTRIM(district) <> ''
                GROUP BY district
                                {having_clause}
                                ORDER BY {order_clause}
                LIMIT %s
            """
            
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params + ranking_params + [limit])
                    rows = cur.fetchall() or []

                    if not rows:
                        # Fallback to ward/xa ranking when district data is missing.
                        ward_fallback = self.get_ward_rankings(query, limit=limit)
                        wards = ward_fallback.get("wards") or []
                        if wards:
                            districts_result = [
                                {
                                    "district": item.get("ward"),
                                    "count": int(item.get("count") or 0),
                                    "avg_price_vnd": item.get("avg_price_vnd"),
                                }
                                for item in wards
                            ]
                            return {
                                "districts": districts_result,
                                "total_count": int(ward_fallback.get("total_count") or 0),
                                "avg_price_vnd": ward_fallback.get("avg_price_vnd"),
                                "avg_price_per_m2_vnd": ward_fallback.get("avg_price_per_m2_vnd"),
                                "avg_area_m2": ward_fallback.get("avg_area_m2"),
                                "min_price_vnd": ward_fallback.get("min_price_vnd"),
                                "max_price_vnd": ward_fallback.get("max_price_vnd"),
                                "limit": limit,
                                "ranking_scope": "ward",
                                "ranking_metric": ranking_metric,
                                "ranking_min_count": ranking_min_count,
                            }

                        # Last resort: keep a synthetic bucket when source data misses both district and ward.
                        fallback_sql = f"""
                            SELECT
                                'Khong ro quan/huyen' AS district_bucket,
                                COUNT(*) as district_count,
                                AVG(price_value_vnd)::BIGINT as avg_price_vnd,
                                AVG(area_m2)::FLOAT as avg_area_m2,
                                AVG(
                                    CASE
                                        WHEN price_value_vnd IS NOT NULL AND area_m2 IS NOT NULL AND area_m2 > 0
                                        THEN price_value_vnd / area_m2
                                        ELSE NULL
                                    END
                                )::FLOAT as avg_price_per_m2_vnd
                            FROM listings
                            WHERE {where_clause}
                        """
                        cur.execute(fallback_sql, params)
                        rows = cur.fetchall() or []
                    
                    districts_result = [
                        {
                            "district": r[0],
                            "count": int(r[1] or 0),
                            "avg_price_vnd": r[2],
                            "avg_area_m2": r[3],
                            "avg_price_per_m2_vnd": r[4],
                        }
                        for r in rows
                    ]
                    
                    aggregate_sql = f"""
                        SELECT
                            COUNT(*) AS total_count,
                            AVG(price_value_vnd)::BIGINT AS avg_price_vnd,
                            AVG(area_m2)::FLOAT AS avg_area_m2,
                            AVG(
                                CASE
                                    WHEN price_value_vnd IS NOT NULL AND area_m2 IS NOT NULL AND area_m2 > 0
                                    THEN price_value_vnd / area_m2
                                    ELSE NULL
                                END
                            )::FLOAT AS avg_price_per_m2_vnd,
                            MIN(price_value_vnd)::BIGINT AS min_price_vnd,
                            MAX(price_value_vnd)::BIGINT AS max_price_vnd
                        FROM listings
                        WHERE {where_clause}
                    """
                    cur.execute(aggregate_sql, params)
                    agg = cur.fetchone() or (0, None, None, None, None, None)
                    total_count = int(agg[0] or 0)
                    
                    return {
                        "districts": districts_result,
                        "total_count": total_count,
                        "avg_price_vnd": agg[1],
                        "avg_area_m2": agg[2],
                        "avg_price_per_m2_vnd": agg[3],
                        "min_price_vnd": agg[4],
                        "max_price_vnd": agg[5],
                        "limit": limit,
                        "ranking_scope": "district",
                        "ranking_metric": ranking_metric,
                        "ranking_min_count": ranking_min_count,
                    }
        except Exception as exc:
            self.logger.exception("get_district_rankings failed query=%s error=%s", query, exc)
            return {"districts": [], "total_count": 0, "limit": limit, "error": str(exc)}

    def get_ward_rankings(self, query: str, limit: int = 3) -> Dict[str, Any]:
        """
        Get top-K wards/communes ranked by listing count.
        Used for queries like "Top 3 phuong co nhieu listing nhat".
        """
        try:
            parsed = parse_user_query(query or "")
            hard = parsed.hard_filters
            qn = str(parsed.normalized_query or "")
            ranking_metric, ranking_order = self._detect_ranking_metric_from_query(qn)
            ranking_min_count = 5 if ranking_metric == "avg_price_vnd" else None

            where = ["search_document IS NOT NULL", "BTRIM(search_document) <> ''"]
            params: List[Any] = []

            if hard.transaction_type:
                aliases = self._expand_transaction_type_aliases(hard.transaction_type)
                placeholders = ", ".join(["%s"] * len(aliases))
                where.append(f"transaction_type ILIKE ANY(ARRAY[{placeholders}])")
                params.extend([f"%{alias}%" for alias in aliases])
            if hard.property_type:
                aliases = self._expand_property_type_aliases(hard.property_type)
                placeholders = ", ".join(["%s"] * len(aliases))
                where.append(f"property_type ILIKE ANY(ARRAY[{placeholders}])")
                params.extend([f"%{alias}%" for alias in aliases])
            if hard.city:
                city_aliases = self._expand_city_aliases(hard.city)
                placeholders = ", ".join(["%s"] * len(city_aliases))
                where.append(f"city ILIKE ANY(ARRAY[{placeholders}])")
                params.extend([f"%{alias}%" for alias in city_aliases])
            if hard.district:
                district_aliases = self._expand_district_aliases(hard.district)
                if district_aliases:
                    placeholders = ", ".join(["%s"] * len(district_aliases))
                    where.append(f"district ILIKE ANY(ARRAY[{placeholders}])")
                    params.extend([f"%{alias}%" for alias in district_aliases])
            if hard.max_price_vnd is not None:
                where.append("price_value_vnd <= %s")
                params.append(int(hard.max_price_vnd))
            if hard.min_area_m2 is not None:
                where.append("area_m2 >= %s")
                params.append(float(hard.min_area_m2))
            if hard.min_bedrooms is not None:
                where.append("bedrooms IS NOT NULL AND bedrooms >= %s")
                params.append(int(hard.min_bedrooms))

            self._append_soft_preference_filters(where, params, parsed)

            where_clause = " AND ".join(where) if where else "TRUE"

            if ranking_metric == "avg_price_vnd":
                order_clause = (
                    "avg_price_vnd ASC NULLS LAST, ward_count DESC"
                    if ranking_order == "asc"
                    else "avg_price_vnd DESC NULLS LAST, ward_count DESC"
                )
            else:
                order_clause = "ward_count DESC, avg_price_vnd DESC"

            having_clause = ""
            ranking_params: List[Any] = []
            if ranking_min_count is not None:
                having_clause = "HAVING COUNT(*) >= %s"
                ranking_params.append(int(ranking_min_count))

            sql = f"""
                SELECT
                    BTRIM(ward) AS ward,
                    COUNT(*) as ward_count,
                    AVG(price_value_vnd)::BIGINT as avg_price_vnd,
                    AVG(area_m2)::FLOAT as avg_area_m2,
                    AVG(
                        CASE
                            WHEN price_value_vnd IS NOT NULL AND area_m2 IS NOT NULL AND area_m2 > 0
                            THEN price_value_vnd / area_m2
                            ELSE NULL
                        END
                    )::FLOAT as avg_price_per_m2_vnd
                FROM listings
                WHERE {where_clause}
                  AND ward IS NOT NULL
                  AND BTRIM(ward) <> ''
                GROUP BY BTRIM(ward)
                                {having_clause}
                                ORDER BY {order_clause}
                LIMIT %s
            """

            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params + ranking_params + [limit])
                    rows = cur.fetchall() or []

                    wards_result = [
                        {
                            "ward": r[0],
                            "count": int(r[1] or 0),
                            "avg_price_vnd": r[2],
                            "avg_area_m2": r[3],
                            "avg_price_per_m2_vnd": r[4],
                        }
                        for r in rows
                    ]

                    aggregate_sql = f"""
                        SELECT
                            COUNT(*) AS total_count,
                            AVG(price_value_vnd)::BIGINT AS avg_price_vnd,
                            AVG(area_m2)::FLOAT AS avg_area_m2,
                            AVG(
                                CASE
                                    WHEN price_value_vnd IS NOT NULL AND area_m2 IS NOT NULL AND area_m2 > 0
                                    THEN price_value_vnd / area_m2
                                    ELSE NULL
                                END
                            )::FLOAT AS avg_price_per_m2_vnd,
                            MIN(price_value_vnd)::BIGINT AS min_price_vnd,
                            MAX(price_value_vnd)::BIGINT AS max_price_vnd
                        FROM listings
                        WHERE {where_clause}
                    """
                    cur.execute(aggregate_sql, params)
                    agg = cur.fetchone() or (0, None, None, None, None, None)
                    total_count = int(agg[0] or 0)

                    # Keep districts key for backward compatibility in runtime/UI pipelines.
                    districts_compat = [
                        {
                            "district": item.get("ward"),
                            "count": int(item.get("count") or 0),
                            "avg_price_vnd": item.get("avg_price_vnd"),
                            "avg_area_m2": item.get("avg_area_m2"),
                            "avg_price_per_m2_vnd": item.get("avg_price_per_m2_vnd"),
                        }
                        for item in wards_result
                    ]

                    return {
                        "wards": wards_result,
                        "districts": districts_compat,
                        "total_count": total_count,
                        "avg_price_vnd": agg[1],
                        "avg_area_m2": agg[2],
                        "avg_price_per_m2_vnd": agg[3],
                        "min_price_vnd": agg[4],
                        "max_price_vnd": agg[5],
                        "limit": limit,
                        "ranking_scope": "ward",
                        "ranking_metric": ranking_metric,
                        "ranking_min_count": ranking_min_count,
                    }
        except Exception as exc:
            self.logger.exception("get_ward_rankings failed query=%s error=%s", query, exc)
            return {
                "wards": [],
                "districts": [],
                "total_count": 0,
                "limit": limit,
                "ranking_scope": "ward",
                "ranking_metric": "count",
                "ranking_min_count": None,
                "error": str(exc),
            }

    def get_district_avg_price_comparison(self, query: str) -> Dict[str, Any]:
        """Compare average prices across explicitly mentioned districts in one query."""
        try:
            parsed = parse_user_query(query or "")
            hard = parsed.hard_filters
            qn = str(parsed.normalized_query or "")

            districts = self._extract_districts_from_normalized_query(qn)
            if hard.district and hard.district not in districts:
                districts.insert(0, hard.district)
            districts = list(dict.fromkeys(districts))

            if len(districts) < 2:
                return {
                    "districts": [],
                    "total_count": 0,
                    "comparison_mode": "avg_price_between_districts",
                    "error": "Can it nhat 2 quan/huyen de so sanh gia trung binh.",
                }

            where = ["search_document IS NOT NULL", "BTRIM(search_document) <> ''"]
            params: List[Any] = []

            if hard.transaction_type:
                aliases = self._expand_transaction_type_aliases(hard.transaction_type)
                placeholders = ", ".join(["%s"] * len(aliases))
                where.append(f"transaction_type ILIKE ANY(ARRAY[{placeholders}])")
                params.extend([f"%{alias}%" for alias in aliases])
            if hard.property_type:
                aliases = self._expand_property_type_aliases(hard.property_type)
                placeholders = ", ".join(["%s"] * len(aliases))
                where.append(f"property_type ILIKE ANY(ARRAY[{placeholders}])")
                params.extend([f"%{alias}%" for alias in aliases])
            if hard.city:
                city_aliases = self._expand_city_aliases(hard.city)
                placeholders = ", ".join(["%s"] * len(city_aliases))
                where.append(f"city ILIKE ANY(ARRAY[{placeholders}])")
                params.extend([f"%{alias}%" for alias in city_aliases])
            if hard.max_price_vnd is not None:
                where.append("price_value_vnd <= %s")
                params.append(int(hard.max_price_vnd))
            if hard.min_price_vnd is not None:
                where.append("price_value_vnd >= %s")
                params.append(int(hard.min_price_vnd))
            if hard.min_area_m2 is not None:
                where.append("area_m2 >= %s")
                params.append(float(hard.min_area_m2))
            if hard.max_area_m2 is not None:
                where.append("area_m2 <= %s")
                params.append(float(hard.max_area_m2))
            if hard.min_bedrooms is not None:
                where.append("bedrooms IS NOT NULL AND bedrooms >= %s")
                params.append(int(hard.min_bedrooms))

            self._append_soft_preference_filters(where, params, parsed)

            district_case_clauses: List[str] = []
            district_case_params: List[Any] = []
            for district in districts:
                aliases = self._expand_district_aliases(district)
                if not aliases:
                    continue
                placeholders = ", ".join(["%s"] * len(aliases))
                district_case_clauses.append(
                    f"WHEN district ILIKE ANY(ARRAY[{placeholders}]) THEN %s"
                )
                district_case_params.extend([f"%{alias}%" for alias in aliases])
                district_case_params.append(district)

            if not district_case_clauses:
                return {
                    "districts": [],
                    "total_count": 0,
                    "comparison_mode": "avg_price_between_districts",
                    "error": "Khong nhan dien duoc quan/huyen trong truy van.",
                }

            where_clause = " AND ".join(where) if where else "TRUE"
            sql = f"""
                SELECT
                    district_bucket,
                    COUNT(*) AS district_count,
                    AVG(price_value_vnd)::BIGINT AS avg_price_vnd,
                    AVG(area_m2)::FLOAT AS avg_area_m2,
                    AVG(
                        CASE
                            WHEN price_value_vnd IS NOT NULL AND area_m2 IS NOT NULL AND area_m2 > 0
                            THEN price_value_vnd / area_m2
                            ELSE NULL
                        END
                    )::FLOAT AS avg_price_per_m2_vnd
                FROM (
                    SELECT
                        CASE
                            {' '.join(district_case_clauses)}
                            ELSE NULL
                        END AS district_bucket,
                        price_value_vnd,
                        area_m2
                    FROM listings
                    WHERE {where_clause}
                ) t
                WHERE district_bucket IS NOT NULL
                GROUP BY district_bucket
                ORDER BY avg_price_vnd DESC NULLS LAST, district_count DESC
            """

            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, district_case_params + params)
                    rows = cur.fetchall() or []

                    districts_result = [
                        {
                            "district": r[0],
                            "count": int(r[1] or 0),
                            "avg_price_vnd": r[2],
                            "avg_area_m2": r[3],
                            "avg_price_per_m2_vnd": r[4],
                        }
                        for r in rows
                    ]

                    total_count = sum(int(item.get("count") or 0) for item in districts_result)
                    return {
                        "districts": districts_result,
                        "total_count": total_count,
                        "comparison_mode": "avg_price_between_districts",
                        "ranking_scope": "district",
                        "ranking_metric": "avg_price_vnd",
                    }
        except Exception as exc:
            self.logger.exception("get_district_avg_price_comparison failed query=%s error=%s", query, exc)
            return {
                "districts": [],
                "total_count": 0,
                "comparison_mode": "avg_price_between_districts",
                "error": str(exc),
            }

    def get_avg_price_per_m2_by_district(self, query: str) -> Dict[str, Any]:
        """
        Calculate average price per m2 by district (for Issue 8).
        Used for queries like "Giá/m2 trung bình ở quận 1 là bao nhiêu?"
        
        Returns:
            {
                "district": "Quận 1",
                "avg_price_per_m2_vnd": 150000000.5,
                "avg_price_vnd": 12000000000,
                "avg_area_m2": 80,
                "count": 331
            }
        """
        try:
            parsed = parse_user_query(query or "")
            hard = parsed.hard_filters
            qn = str(parsed.normalized_query or "")
            districts = self._extract_districts_from_normalized_query(qn)
            
            if hard.district and hard.district not in districts:
                districts.insert(0, hard.district)
            
            if not districts:
                return {"error": "No district specified in query"}
            
            target_district = districts[0]  # Use first mentioned district
            
            # Build WHERE conditions
            where = ["search_document IS NOT NULL", "BTRIM(search_document) <> ''"]
            params: List[Any] = []
            
            if hard.transaction_type:
                aliases = self._expand_transaction_type_aliases(hard.transaction_type)
                placeholders = ", ".join(["%s"] * len(aliases))
                where.append(f"transaction_type ILIKE ANY(ARRAY[{placeholders}])")
                params.extend([f"%{alias}%" for alias in aliases])
            if hard.property_type:
                aliases = self._expand_property_type_aliases(hard.property_type)
                placeholders = ", ".join(["%s"] * len(aliases))
                where.append(f"property_type ILIKE ANY(ARRAY[{placeholders}])")
                params.extend([f"%{alias}%" for alias in aliases])
            if hard.city:
                city_aliases = self._expand_city_aliases(hard.city)
                placeholders = ", ".join(["%s"] * len(city_aliases))
                where.append(f"city ILIKE ANY(ARRAY[{placeholders}])")
                params.extend([f"%{alias}%" for alias in city_aliases])
            
            # Add district filter
            district_aliases = self._expand_district_aliases(target_district)
            if not district_aliases:
                return {"error": f"Cannot resolve district: {target_district}"}
            
            placeholders = ", ".join(["%s"] * len(district_aliases))
            where.append(f"district ILIKE ANY(ARRAY[{placeholders}])")
            params.extend([f"%{alias}%" for alias in district_aliases])
            
            if hard.max_price_vnd is not None:
                where.append("price_value_vnd <= %s")
                params.append(int(hard.max_price_vnd))
            if hard.min_price_vnd is not None:
                where.append("price_value_vnd >= %s")
                params.append(int(hard.min_price_vnd))
            if hard.min_area_m2 is not None:
                where.append("area_m2 >= %s")
                params.append(float(hard.min_area_m2))
            if hard.min_bedrooms is not None:
                where.append("bedrooms IS NOT NULL AND bedrooms >= %s")
                params.append(int(hard.min_bedrooms))
            
            where_clause = " AND ".join(where) if where else "TRUE"
            
            sql = f"""
                SELECT
                    COUNT(*) as count,
                    AVG(price_value_vnd)::BIGINT as avg_price_vnd,
                    AVG(area_m2)::FLOAT as avg_area_m2,
                    AVG(price_value_vnd / NULLIF(area_m2, 0))::FLOAT as avg_price_per_m2
                FROM listings
                WHERE {where_clause}
                  AND price_value_vnd IS NOT NULL
                  AND area_m2 IS NOT NULL
                  AND area_m2 > 0
            """
            
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    row = cur.fetchone()
                    if not row:
                        return {"error": "No listings found for district"}
                    
                    return {
                        "district": target_district,
                        "count": int(row[0] or 0),
                        "avg_price_vnd": row[1],
                        "avg_area_m2": row[2],
                        "avg_price_per_m2_vnd": row[3]
                    }
        except Exception as exc:
            self.logger.exception("get_avg_price_per_m2_by_district failed query=%s error=%s", query, exc)
            return {"error": str(exc)}

    def get_property_type_breakdown(self, query: str) -> Dict[str, Any]:
        """
        Get breakdown of listings by property_type (for Issue 12).
        Used for queries like "Phân bố số lượng listing theo loại bất động sản"
        
        Returns:
            {
                "breakdown": [
                    {"property_type": "Căn hộ", "count": 2500, "avg_price_vnd": 10000000000},
                    {"property_type": "Nhà phố", "count": 1200, "avg_price_vnd": 12000000000},
                    ...
                ],
                "total_count": 5146
            }
        """
        try:
            parsed = parse_user_query(query or "")
            hard = parsed.hard_filters
            qn = str(parsed.normalized_query or "")

            breakdown_metric = "count"
            count_intent_tokens = [
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
            if ("m2" in qn or "gia/m2" in qn or "gia tren m2" in qn) and any(token in qn for token in ["gia", "trung binh"]):
                breakdown_metric = "avg_price_per_m2_vnd"
            elif "dien tich" in qn and "trung binh" in qn:
                breakdown_metric = "avg_area_m2"
            elif "gia" in qn and "trung binh" in qn and not any(token in qn for token in count_intent_tokens):
                breakdown_metric = "avg_price_vnd"
            
            # Build WHERE conditions
            where = ["search_document IS NOT NULL", "BTRIM(search_document) <> ''"]
            params: List[Any] = []
            
            if hard.transaction_type:
                aliases = self._expand_transaction_type_aliases(hard.transaction_type)
                placeholders = ", ".join(["%s"] * len(aliases))
                where.append(f"transaction_type ILIKE ANY(ARRAY[{placeholders}])")
                params.extend([f"%{alias}%" for alias in aliases])
            if hard.city:
                where.append("city ILIKE %s")
                params.append(f"%{hard.city}%")
            if hard.district:
                aliases = self._expand_district_aliases(hard.district)
                if aliases:
                    placeholders = ", ".join(["%s"] * len(aliases))
                    where.append(f"district ILIKE ANY(ARRAY[{placeholders}])")
                    params.extend([f"%{alias}%" for alias in aliases])
            if hard.max_price_vnd is not None:
                where.append("price_value_vnd <= %s")
                params.append(int(hard.max_price_vnd))
            if hard.min_price_vnd is not None:
                where.append("price_value_vnd >= %s")
                params.append(int(hard.min_price_vnd))
            if hard.min_area_m2 is not None:
                where.append("area_m2 >= %s")
                params.append(float(hard.min_area_m2))
            if hard.min_bedrooms is not None:
                where.append("bedrooms IS NOT NULL AND bedrooms >= %s")
                params.append(int(hard.min_bedrooms))
            
            where_clause = " AND ".join(where) if where else "TRUE"
            
            sql = f"""
                SELECT
                    property_type,
                    COUNT(*) as type_count,
                    AVG(price_value_vnd)::BIGINT as avg_price_vnd,
                    AVG(area_m2)::FLOAT as avg_area_m2,
                    AVG(
                        CASE
                            WHEN price_value_vnd IS NOT NULL AND area_m2 IS NOT NULL AND area_m2 > 0
                            THEN price_value_vnd / area_m2
                            ELSE NULL
                        END
                    )::FLOAT as avg_price_per_m2_vnd
                FROM listings
                WHERE {where_clause}
                GROUP BY property_type
                ORDER BY type_count DESC, avg_price_vnd DESC
            """
            
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall() or []
                    
                    breakdown = [
                        {
                            "property_type": r[0],
                            "count": int(r[1] or 0),
                            "avg_price_vnd": r[2],
                            "avg_area_m2": r[3],
                            "avg_price_per_m2_vnd": r[4],
                        }
                        for r in rows
                    ]

                    def _sort_key(item: Dict[str, Any]) -> float:
                        value = item.get("count") if breakdown_metric == "count" else item.get(breakdown_metric)
                        if value is None:
                            return float("-inf")
                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            return float("-inf")

                    breakdown = sorted(breakdown, key=_sort_key, reverse=True)
                    
                    # Get total count
                    total_sql = f"SELECT COUNT(*) FROM listings WHERE {where_clause}"
                    cur.execute(total_sql, params)
                    total_count = cur.fetchone()[0] or 0
                    
                    return {
                        "breakdown": breakdown,
                        "total_count": total_count,
                        "breakdown_metric": breakdown_metric,
                    }
        except Exception as exc:
            self.logger.exception("get_property_type_breakdown failed query=%s error=%s", query, exc)
            return {"breakdown": [], "total_count": 0, "breakdown_metric": "count", "error": str(exc)}

    @staticmethod
    def _detect_clarification_needed(qn: str) -> Optional[str]:
        """
        Detect queries that need user clarification before proceeding with analytics.
        Returns clarification prompt if needed, None otherwise.
        
        Examples:
        - "giá rẻ" → "Giá 'rẻ' theo bạn là khoảng bao nhiêu tỷ?"
        - "khu vực" → "Bạn muốn xem theo quận hay theo thành phố?"
        - "phù hợp gia đình" → "Gia đình bạn có mấy người, ngân sách khoảng bao nhiêu?"
        """
        text = str(qn or "").strip().lower()
        if not text:
            return None

        try:
            parsed = parse_user_query(text)
        except Exception:
            parsed = None

        has_geo_scope_from_parse = bool(
            parsed
            and any(
                [
                    bool(getattr(parsed.hard_filters, "city", None)),
                    bool(getattr(parsed.hard_filters, "district", None)),
                    bool(getattr(parsed.hard_filters, "ward", None)),
                    bool(getattr(parsed.hard_filters, "street", None)),
                    bool(getattr(parsed.hard_filters, "project", None)),
                ]
            )
        )
        use_case = str(getattr(parsed, "use_case", "") or "").strip().lower() if parsed else ""

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
        if use_case in {"market_overview", "suggest_area"} or (
            has_geo_scope_from_parse and any(token in text for token in market_overview_tokens)
        ):
            return None

        is_analytics_ranking_query = (
            any(term in text for term in ["cao nhat", "thap nhat", "top", "xep hang", "max", "min"])
            and any(term in text for term in ["gia", "trung binh", "so luong", "listing", "gia/m2", "m2"])
        )
        is_family_listing_ranking_query = (
            any(term in text for term in ["nhieu listing", "nhieu nhat", "top"])
            and any(term in text for term in ["gia dinh", "family"])
            and any(term in text for term in ["khu", "khu vuc", "quan", "huyen", "phuong", "xa", "noi nao"])
        )
        is_analytics_ranking_query = is_analytics_ranking_query or is_family_listing_ranking_query
        has_generic_area_phrase = any(term in text for term in ["khu vuc", "khu nao", "noi nao"])
        has_explicit_geo_scope = any(
            term in text
            for term in [
                "quan",
                "huyen",
                "phuong",
                "xa",
                "district",
                "ward",
                "commune",
                "thanh pho",
                "tp",
                "city",
                "hcm",
                "ha noi",
                "da nang",
            ]
        )

        # Ranking queries still need scope clarification when user only asks generic "khu vực nào".
        if (
            is_analytics_ranking_query
            and has_generic_area_phrase
            and not has_explicit_geo_scope
            and not is_family_listing_ranking_query
        ):
            return (
                "Cau hoi nay dang hoi o muc kha chung. Ban muon minh xep hang theo "
                "quan/huyen hay theo phuong/xa, va trong thanh pho nao? "
                "Vi du: 'o TP.HCM, quan nao co gia trung binh cao nhat'."
            )
        
        # Check for ambiguous price terms
        if any(term in text for term in ["gia re", "giá rẻ", "rẻ", "cheap"]):
            return (
                "Mình có thể tư vấn chính xác hơn nếu bạn cho mình khung ngân sách cụ thể. "
                "Bạn đang xem mức khoảng bao nhiêu tỷ (ví dụ: dưới 3 tỷ, 3-5 tỷ, hay trên 5 tỷ)?"
            )
        
        # Check for vague location terms without specifics
        if (
            any(term in text for term in ["khu vuc", "khu", "noi nao"])
            and not any(term in text for term in ["quan", "q", "district", "thanh pho", "tptp"])
            and not is_analytics_ranking_query
        ):
            return (
                "Bạn đang muốn hỏi khu vực theo phạm vi nào để mình tư vấn chuẩn hơn: "
                "theo thành phố hay theo quận/huyện? "
                "Nếu có ngân sách và loại bất động sản mong muốn, mình sẽ lọc giúp ngay."
            )
        
        # Check for family-fit without specifics
        if any(term in text for term in ["phu hop gia dinh", "cho gia dinh", "gia dinh tre"]):
            return (
                "Để tư vấn đúng nhu cầu gia đình, bạn cho mình thêm 2 thông tin: "
                "gia đình có bao nhiêu người và ngân sách dự kiến khoảng bao nhiêu tỷ nhé."
            )
        
        # Check for vague suitability
        if any(term in text for term in ["phu hop", "ok", "can bang", "toi", "hop ly"]) and not any(
            term in text for term in ["quan", "q", "gia", "tien", "tỷ"]
        ):
            return (
                "Mình có thể tư vấn chi tiết hơn nếu bạn bổ sung tiêu chí cụ thể, "
                "ví dụ: khu vực ưu tiên (thành phố/quận), ngân sách, và loại bất động sản bạn quan tâm."
            )
        
        return None