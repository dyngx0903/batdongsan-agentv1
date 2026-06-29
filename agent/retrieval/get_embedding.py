from __future__ import annotations

import os
from typing import List, Optional

try:
    from google import genai
except Exception:
    genai = None


EMBED_MODEL_NAME = "gemini-embedding-001"
EMBED_TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"
EMBED_TASK_TYPE_QUERY = "RETRIEVAL_QUERY"


def _to_float_vector(values: object) -> Optional[List[float]]:
    if not isinstance(values, list):
        return None

    out: List[float] = []
    for item in values:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return None

    return out or None


def get_embedding(
    text: str,
    task_type: str = EMBED_TASK_TYPE_DOCUMENT,
    model_name: str = EMBED_MODEL_NAME,
    output_dimensionality: int | None = None,
) -> Optional[List[float]]:
    payload = (text or "").strip()
    if not payload:
        return None

    if genai is None:
        return None

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    config = {"task_type": task_type}
    if isinstance(output_dimensionality, int) and output_dimensionality > 0:
        config["output_dimensionality"] = output_dimensionality

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.embed_content(
            model=model_name,
            contents=payload,
            config=config,
        )
        values = getattr(getattr(response, "embeddings", [None])[0], "values", None)
        return _to_float_vector(values)
    except Exception:
        return None
