import json
import os
import re
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI


_DEFAULT_ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=_DEFAULT_ENV_PATH, override=False)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _coerce_confidence(value: Any) -> float:
    try:
        return _clamp01(float(value))
    except Exception:
        return 0.0


def _extract_json(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"LLM did not return JSON: {text[:200]}")
    return json.loads(m.group(0))


def score_subclaim_to_superclaim_confidence(
    *,
    subclaim_text: str,
    superclaim_text: str,
    subclaim_id: str | None = None,
    superclaim_id: str | None = None,
) -> Dict[str, Any]:
    """
    Live LLM-scored confidence for whether a subclaim belongs under a superclaim.
    Returns a dict with at least: {"confidence": float, "verdict": str, "reason": str}
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment (.env).")

    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    client = OpenAI(api_key=api_key)

    system_msg = (
        "You score hierarchical claim mappings.\n"
        "You MUST return ONLY a single JSON object, no markdown, no prose.\n"
        'Schema: {"verdict": "valid|invalid|uncertain", "confidence": 0.0-1.0, "reason": string}.'
    )

    user_msg = f"""
Task: Score how appropriate it is for the SUBCLAIM to be mapped under the SUPERCLAIM in a claim hierarchy.

Definitions:
- VALID: subclaim clearly supports, instantiates, refines, or is a component of the superclaim.
- INVALID: out of scope, only loosely related, or mismatched in type.
- UNCERTAIN: unclear or ambiguous relationship.

SUBCLAIM:
- id: {subclaim_id}
- text: {subclaim_text}

SUPERCLAIM:
- id: {superclaim_id}
- text: {superclaim_text}

Return ONLY valid JSON, no markdown, with this exact shape:
{{
  "verdict": "valid" | "invalid" | "uncertain",
  "confidence": number between 0.0 and 1.0,
  "reason": "short explanation"
}}
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    text = (resp.choices[0].message.content or "").strip()
    data = _extract_json(text)

    verdict = data.get("verdict")
    if verdict not in ("valid", "invalid", "uncertain"):
        verdict = "uncertain"

    confidence = _coerce_confidence(data.get("confidence"))
    reason = str(data.get("reason") or "").strip() or "No reason provided."

    return {"verdict": verdict, "confidence": confidence, "reason": reason}

