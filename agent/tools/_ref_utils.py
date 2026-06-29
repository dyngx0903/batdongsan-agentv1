from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def parse_listing_ref(value: Any) -> tuple[str | None, str | None]:
    text = str(value or "").strip()
    if not text or "/" not in text:
        return None, None
    source, listing_id = text.split("/", 1)
    source = source.strip()
    listing_id = listing_id.strip()
    if not source or not listing_id:
        return None, None
    return source, listing_id


def normalize_single_listing_ref(args: Dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    source = str(args.get("source") or "").strip() or None
    listing_id = str(args.get("listing_id") or "").strip() or None

    # Support common aliases from UI/runtime payloads.
    if not source:
        source = str(args.get("listing_source") or "").strip() or None
    if not listing_id:
        listing_id = str(args.get("id") or "").strip() or None

    listing_ref = str(args.get("listing_ref") or args.get("ref") or "").strip() or None
    if (not source or not listing_id) and listing_ref:
        parsed_source, parsed_listing_id = parse_listing_ref(listing_ref)
        source = source or parsed_source
        listing_id = listing_id or parsed_listing_id

    normalized_ref = f"{source}/{listing_id}" if source and listing_id else listing_ref
    return source, listing_id, normalized_ref


def normalize_compare_listing_refs(args: Dict[str, Any]) -> Dict[str, Optional[str]]:
    source_a = str(args.get("source_a") or "").strip() or None
    listing_id_a = str(args.get("listing_id_a") or "").strip() or None
    source_b = str(args.get("source_b") or "").strip() or None
    listing_id_b = str(args.get("listing_id_b") or "").strip() or None

    listing_ref_a = str(args.get("listing_ref_a") or "").strip() or None
    listing_ref_b = str(args.get("listing_ref_b") or "").strip() or None

    refs_value = args.get("listing_refs")
    refs = refs_value if isinstance(refs_value, list) else []
    if not listing_ref_a and len(refs) > 0:
        listing_ref_a = str(refs[0] or "").strip() or None
    if not listing_ref_b and len(refs) > 1:
        listing_ref_b = str(refs[1] or "").strip() or None

    if (not source_a or not listing_id_a) and listing_ref_a:
        parsed_source_a, parsed_id_a = parse_listing_ref(listing_ref_a)
        source_a = source_a or parsed_source_a
        listing_id_a = listing_id_a or parsed_id_a

    if (not source_b or not listing_id_b) and listing_ref_b:
        parsed_source_b, parsed_id_b = parse_listing_ref(listing_ref_b)
        source_b = source_b or parsed_source_b
        listing_id_b = listing_id_b or parsed_id_b

    return {
        "source_a": source_a,
        "listing_id_a": listing_id_a,
        "source_b": source_b,
        "listing_id_b": listing_id_b,
        "listing_ref_a": f"{source_a}/{listing_id_a}" if source_a and listing_id_a else listing_ref_a,
        "listing_ref_b": f"{source_b}/{listing_id_b}" if source_b and listing_id_b else listing_ref_b,
    }
