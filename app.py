import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm_confidence import score_subclaim_to_superclaim_confidence


ROOT = Path(__file__).resolve().parent

_DEFAULT_ENV_PATH = ROOT / ".env"
load_dotenv(dotenv_path=str(_DEFAULT_ENV_PATH), override=False)

CODEBOOK_NAME = "greenwashing_codebook.json"
SUPERCLAIMS_NAME = "greenwashing_superclaims.json"
MAP_NAME = "claim_superclaim_map.json"
HISTORY_NAME = "greenwashing_claim_history.json"

CODEBOOK_PATH = ROOT / CODEBOOK_NAME
SUPERCLAIMS_PATH = ROOT / SUPERCLAIMS_NAME
MAP_PATH = ROOT / MAP_NAME
HISTORY_PATH = ROOT / HISTORY_NAME

# Local-only proposal queue (dev / non-Supabase). On Vercel, use Supabase DB instead.
APP_DATA_DIR = Path(os.getenv("CLAIMS_LOCAL_DATA_DIR", str(ROOT / "data")))
PROPOSALS_PATH = APP_DATA_DIR / "proposals.json"
MERGES_PATH = APP_DATA_DIR / "merges.json"

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip()
SUPABASE_KEY = (
    (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    or (os.getenv("SUPABASE_KEY") or "").strip()
)
SUPABASE_CLAIMS_BUCKET = (os.getenv("SUPABASE_CLAIMS_BUCKET") or "").strip()
SUPABASE_CLAIMS_PREFIX = (os.getenv("SUPABASE_CLAIMS_PREFIX") or "").strip().strip("/")

_supabase_client = None


def _supabase_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _claims_storage_enabled() -> bool:
    return _supabase_enabled() and bool(SUPABASE_CLAIMS_BUCKET)


def _get_supabase():
    global _supabase_client
    if not _supabase_enabled():
        raise HTTPException(
            status_code=500,
            detail="Supabase is not configured (set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY).",
        )
    if _supabase_client is None:
        from supabase import create_client

        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


def _storage_object_path(filename: str) -> str:
    fn = str(filename).lstrip("/")
    if SUPABASE_CLAIMS_PREFIX:
        return f"{SUPABASE_CLAIMS_PREFIX}/{fn}"
    return fn


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Missing required file: {path.name}") from e
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON in {path.name}: {e}") from e


def _read_claim_json_bytes(filename: str) -> bytes:
    if _claims_storage_enabled():
        sb = _get_supabase()
        path = _storage_object_path(filename)
        try:
            data = sb.storage.from_(SUPABASE_CLAIMS_BUCKET).download(path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Supabase Storage download failed for {path}: {e}",
            ) from e
        if data is None:
            raise HTTPException(status_code=500, detail=f"Missing object in storage: {path}")
        return data if isinstance(data, (bytes, bytearray)) else bytes(data)

    path = ROOT / filename
    try:
        return path.read_bytes()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Missing required file: {filename}") from e


def _read_claim_json(filename: str) -> Any:
    raw = _read_claim_json_bytes(filename)
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON in {filename}: {e}") from e


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _upload_claim_json(filename: str, obj: Any) -> None:
    if not _claims_storage_enabled():
        path = ROOT / filename
        _write_json_atomic(path, obj)
        return

    sb = _get_supabase()
    path = _storage_object_path(filename)
    body = (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    try:
        sb.storage.from_(SUPABASE_CLAIMS_BUCKET).upload(
            path,
            body,
            {"content-type": "application/json", "upsert": "true"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Supabase Storage upload failed for {path}: {e}",
        ) from e


def _bundle_fingerprint() -> str:
    h = hashlib.sha256()
    for name in (CODEBOOK_NAME, SUPERCLAIMS_NAME, MAP_NAME, HISTORY_NAME):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(_read_claim_json_bytes(name))
        h.update(b"\0\0")
    return h.hexdigest()[:12]


def split_into_paragraphs(text: str) -> List[str]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    # Match frontend behavior: every newline starts a new paragraph; ignore empties.
    out: List[str] = []
    for line in normalized.split("\n"):
        s = " ".join(line.split()).strip()
        if s:
            out.append(s)
    return out


def _normalize_subclaim_id(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    if s.startswith("NC_"):
        return s
    if s.startswith("SC_"):
        s = s.replace("SC_", "", 1)
    return f"NC_{s}"


def _normalize_superclaim_id(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    if s.startswith("SC_"):
        return s
    if s.startswith("NC_"):
        s = s.replace("NC_", "", 1)
    return f"SC_{s}"


def _next_id(existing_ids: List[str], prefix: str) -> str:
    nums: List[int] = []
    for x in existing_ids:
        if not isinstance(x, str) or not x.startswith(prefix):
            continue
        try:
            nums.append(int(x.replace(prefix, "", 1)))
        except Exception:
            continue
    n = (max(nums) + 1) if nums else 1
    return f"{prefix}{n}"


def _load_taxonomy() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    codebook = _read_claim_json(CODEBOOK_NAME)
    superclaims = _read_claim_json(SUPERCLAIMS_NAME)
    claim_map = _read_claim_json(MAP_NAME)

    if not isinstance(codebook, dict):
        raise HTTPException(status_code=500, detail="greenwashing_codebook.json must be an object {NC_*: text}")
    if not isinstance(superclaims, dict):
        raise HTTPException(status_code=500, detail="greenwashing_superclaims.json must be an object {SC_*: text}")
    if not isinstance(claim_map, dict):
        # Keep support narrow for now; your frontend supports multiple shapes, but dynamic editing is safest with dict.
        raise HTTPException(status_code=500, detail="claim_superclaim_map.json must be an object {NC_*: SC_*} for approvals")

    codebook_norm = {_normalize_subclaim_id(k): str(v).strip() for k, v in codebook.items() if str(v).strip()}
    super_norm = {_normalize_superclaim_id(k): str(v).strip() for k, v in superclaims.items() if str(v).strip()}
    map_norm = {_normalize_subclaim_id(k): _normalize_superclaim_id(v) for k, v in claim_map.items() if v}
    return codebook_norm, super_norm, map_norm


def _tfidf_topk(query: str, items: List[Tuple[str, str]], k: int = 8) -> List[Tuple[str, str, float]]:
    if not items:
        return []
    texts = [t for _, t in items]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    X = vec.fit_transform(texts)
    q = vec.transform([query])
    sims = cosine_similarity(q, X)[0]
    ranked = sorted(zip(items, sims), key=lambda x: float(x[1]), reverse=True)[:k]
    return [(sid, text, float(score)) for (sid, text), score in ranked]


def _tfidf_pairwise_cosine_matrix(texts: List[str]) -> Any:
    """
    Pairwise cosine similarities on TF‑IDF vectors (same general approach as _tfidf_topk).
    Returns an (n, n) dense matrix (sklearn returns ndarray-like).
    """
    if not texts:
        return []

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    X = vec.fit_transform(texts)
    return cosine_similarity(X, X)


def _pick_canonical_id(ids: List[str]) -> str:
    """
    Deterministic canonical pick: prefer lowest numeric suffix when ids share NC_/SC_ prefix,
    else lexicographic.
    """

    def sort_key(x: str) -> Tuple[int, str]:
        s = str(x)
        for prefix in ("NC_", "SC_"):
            if s.startswith(prefix):
                body = s[len(prefix) :]
                try:
                    return (0, f"{prefix}{int(body)}")
                except Exception:
                    return (1, s)
        return (1, s)

    return sorted([str(i) for i in ids], key=sort_key)[0]


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Raw pasted article text.")
    max_candidates: int = Field(10, ge=1, le=30)
    propose_new_if_below: float = Field(0.55, ge=0.0, le=1.0)
    merge_pair_min_cosine: float = Field(0.82, ge=0.0, le=1.0)
    merge_max_pairs: int = Field(2, ge=0, le=10)


class MatchRow(BaseModel):
    subclaimId: str
    subclaimText: str
    superclaimId: str
    superclaimText: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


ProposalType = Literal[
    "new_subclaim",
    "new_superclaim",
    "link_subclaim_to_superclaim",
    "merge_subclaims",
    "merge_superclaims",
]


class Proposal(BaseModel):
    id: str
    type: ProposalType
    status: Literal["pending", "approved", "rejected"] = "pending"
    createdAt: float
    bundleVersion: str
    paragraph: str
    payload: Dict[str, Any]
    rationale: str = ""
    reviewedBy: str = ""
    reviewedAt: float = 0.0
    appliedBy: str = ""
    appliedAt: float = 0.0


class AnalyzeParagraphResult(BaseModel):
    paragraph: str
    matches: List[MatchRow]
    proposals: List[Proposal]


class AnalyzeResponse(BaseModel):
    bundleVersion: str
    paragraphs: List[AnalyzeParagraphResult]


class ProposalActionResponse(BaseModel):
    ok: bool
    proposal: Proposal


class ReviewerActionRequest(BaseModel):
    reviewer_name: str = Field(..., min_length=1, max_length=120)


def _parse_ts_to_epoch(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0
    return 0.0


def _proposal_row_to_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    created = row.get("created_at") or row.get("createdAt")
    created_f: float
    if isinstance(created, (int, float)):
        created_f = float(created)
    elif isinstance(created, str):
        try:
            from datetime import datetime

            created_f = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
        except Exception:
            created_f = time.time()
    else:
        created_f = time.time()

    reviewed_at = _parse_ts_to_epoch(row.get("reviewed_at") or row.get("reviewedAt"))
    applied_at = _parse_ts_to_epoch(row.get("applied_at") or row.get("appliedAt"))

    return {
        "id": row.get("id"),
        "type": row.get("type"),
        "status": row.get("status"),
        "createdAt": created_f,
        "bundleVersion": row.get("bundle_version") or row.get("bundleVersion") or "",
        "paragraph": row.get("paragraph") or "",
        "payload": row.get("payload") or {},
        "rationale": row.get("rationale") or "",
        "reviewedBy": str(row.get("reviewed_by") or row.get("reviewedBy") or "").strip(),
        "reviewedAt": reviewed_at,
        "appliedBy": str(row.get("applied_by") or row.get("appliedBy") or "").strip(),
        "appliedAt": applied_at,
    }


def _load_proposals() -> Dict[str, Any]:
    if _supabase_enabled():
        sb = _get_supabase()
        try:
            resp = sb.table("taxonomy_proposals").select("*").order("created_at", desc=True).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase read proposals failed: {e}") from e
        rows = getattr(resp, "data", None) or []
        normalized = []
        for r in rows:
            if isinstance(r, dict):
                normalized.append(_proposal_row_to_dict(r))
        return {"proposals": normalized}

    if not PROPOSALS_PATH.exists():
        return {"proposals": []}
    data = _read_json(PROPOSALS_PATH)
    if not isinstance(data, dict) or not isinstance(data.get("proposals"), list):
        return {"proposals": []}
    return data


def _save_proposals(doc: Dict[str, Any]) -> None:
    if _supabase_enabled():
        raise RuntimeError("_save_proposals should not be used in Supabase mode")

    _write_json_atomic(PROPOSALS_PATH, doc)


def _upsert_proposal_db(p: Proposal) -> None:
    sb = _get_supabase()
    reviewed_at = datetime.fromtimestamp(p.reviewedAt, tz=timezone.utc).isoformat() if p.reviewedAt else None
    applied_at = datetime.fromtimestamp(p.appliedAt, tz=timezone.utc).isoformat() if p.appliedAt else None
    row = {
        "id": p.id,
        "type": p.type,
        "status": p.status,
        "bundle_version": p.bundleVersion,
        "paragraph": p.paragraph,
        "payload": p.payload,
        "rationale": p.rationale,
        "reviewed_by": p.reviewedBy or None,
        "reviewed_at": reviewed_at,
        "applied_by": p.appliedBy or None,
        "applied_at": applied_at,
    }
    try:
        sb.table("taxonomy_proposals").upsert(row, on_conflict="id").execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase upsert proposal failed: {e}") from e


def _upsert_proposal(p: Proposal) -> Proposal:
    if _supabase_enabled():
        _upsert_proposal_db(p)
        return p

    doc = _load_proposals()
    lst: List[Dict[str, Any]] = doc.get("proposals", [])
    found = False
    for i, row in enumerate(lst):
        if isinstance(row, dict) and row.get("id") == p.id:
            lst[i] = p.model_dump()
            found = True
            break
    if not found:
        lst.append(p.model_dump())
    doc["proposals"] = lst
    _save_proposals(doc)
    return p


def _proposal_from_row(row: Dict[str, Any]) -> Proposal:
    return Proposal.model_validate(row)


def _store_new_proposal(*, ptype: ProposalType, paragraph: str, payload: Dict[str, Any], rationale: str) -> Proposal:
    bundle = _bundle_fingerprint()
    p = Proposal(
        id=f"prop_{uuid.uuid4().hex[:10]}",
        type=ptype,
        status="pending",
        createdAt=time.time(),
        bundleVersion=bundle,
        paragraph=paragraph,
        payload=payload,
        rationale=rationale or "",
    )
    return _upsert_proposal(p)


def _llm_suggest_mapping(
    *,
    paragraph: str,
    sub_candidates: List[Tuple[str, str, float]],
    super_candidates: List[Tuple[str, str, float]],
    claim_map: Dict[str, str],
    merge_pair_min_cosine: float,
    merge_max_pairs: int,
    propose_new_if_below: float,
) -> Tuple[List[MatchRow], List[Tuple[ProposalType, Dict[str, Any], str]]]:
    """
    Minimal “B” behavior:
    - Pick best existing (subclaim, superclaim) by TF‑IDF and then verify with LLM confidence scorer.
    - If LLM confidence is low, propose a new subclaim under the best superclaim (or propose new superclaim if none fit).
    - Propose merges using TF‑IDF cosine similarity between top candidate texts (subclaims and superclaims).
    """
    proposals: List[Tuple[ProposalType, Dict[str, Any], str]] = []

    if not sub_candidates or not super_candidates:
        proposals.append(
            (
                "new_superclaim",
                {"superclaimText": paragraph[:140]},
                "No existing taxonomy candidates available.",
            )
        )
        return ([], proposals)

    # Merge proposals (TF‑IDF cosine similarity between subclaim texts, same family of scoring as candidate retrieval).
    if merge_max_pairs > 0 and len(sub_candidates) >= 2:
        topn = min(8, len(sub_candidates))
        slice_rows = sub_candidates[:topn]
        ids = [sid for sid, _, _ in slice_rows]
        texts = [t for _, t, _ in slice_rows]
        sim = _tfidf_pairwise_cosine_matrix(texts)

        emitted_sub = 0
        for i in range(len(ids)):
            if emitted_sub >= merge_max_pairs:
                break
            for j in range(i + 1, len(ids)):
                if emitted_sub >= merge_max_pairs:
                    break
                try:
                    sij = float(sim[i][j])
                except Exception:
                    continue
                if sij < merge_pair_min_cosine:
                    continue

                a = _normalize_subclaim_id(ids[i])
                b = _normalize_subclaim_id(ids[j])
                if not a or not b or a == b:
                    continue

                sc_a = claim_map.get(a)
                sc_b = claim_map.get(b)
                if not sc_a or not sc_b:
                    continue
                sc_a = _normalize_superclaim_id(sc_a)
                sc_b = _normalize_superclaim_id(sc_b)
                if sc_a != sc_b:
                    # Only propose merges within the same mapped superclaim to reduce accidental cross-branch merges.
                    continue

                canonical = _pick_canonical_id([a, b])
                other = b if canonical == a else a
                text_by_id = {ids[k]: texts[k] for k in range(len(ids))}
                proposals.append(
                    (
                        "merge_subclaims",
                        {
                            "canonicalSubclaimId": canonical,
                            "canonicalSubclaimText": str(text_by_id.get(canonical) or ""),
                            "mergeSubclaimIds": sorted([a, b]),
                            "removeSubclaimId": other,
                            "removeSubclaimText": str(text_by_id.get(other) or ""),
                            "sharedSuperclaimId": sc_a,
                            "pairCosine": sij,
                        },
                        f"High TF‑IDF cosine similarity between subclaim texts ({sij:.2f}); consider merging within superclaim {sc_a}.",
                    )
                )
                emitted_sub += 1

    # Superclaim merge proposals: pairwise similarity among top superclaim texts.
    if merge_max_pairs > 0 and len(super_candidates) >= 2:
        topn = min(6, len(super_candidates))
        slice_rows = super_candidates[:topn]
        ids = [sid for sid, _, _ in slice_rows]
        texts = [t for _, t, _ in slice_rows]
        sim = _tfidf_pairwise_cosine_matrix(texts)

        emitted_super = 0
        for i in range(len(ids)):
            if emitted_super >= merge_max_pairs:
                break
            for j in range(i + 1, len(ids)):
                if emitted_super >= merge_max_pairs:
                    break
                try:
                    sij = float(sim[i][j])
                except Exception:
                    continue
                if sij < merge_pair_min_cosine:
                    continue

                a = _normalize_superclaim_id(ids[i])
                b = _normalize_superclaim_id(ids[j])
                if not a or not b or a == b:
                    continue

                canonical = _pick_canonical_id([a, b])
                other = b if canonical == a else a
                text_by_id = {ids[k]: texts[k] for k in range(len(ids))}
                proposals.append(
                    (
                        "merge_superclaims",
                        {
                            "canonicalSuperclaimId": canonical,
                            "canonicalSuperclaimText": str(text_by_id.get(canonical) or ""),
                            "mergeSuperclaimIds": sorted([a, b]),
                            "removeSuperclaimId": other,
                            "removeSuperclaimText": str(text_by_id.get(other) or ""),
                            "pairCosine": sij,
                        },
                        f"High TF‑IDF cosine similarity between superclaim texts ({sij:.2f}); consider merging superclaims.",
                    )
                )
                emitted_super += 1

    best_sub_id, best_sub_text, _ = sub_candidates[0]
    best_super_id, best_super_text, _ = super_candidates[0]

    score = score_subclaim_to_superclaim_confidence(
        subclaim_text=best_sub_text,
        superclaim_text=best_super_text,
        subclaim_id=best_sub_id,
        superclaim_id=best_super_id,
    )
    conf = float(score.get("confidence") or 0.0)
    verdict = str(score.get("verdict") or "uncertain")
    reason = str(score.get("reason") or "").strip()

    if conf >= propose_new_if_below and verdict in ("valid", "uncertain"):
        return (
            [
                MatchRow(
                    subclaimId=best_sub_id,
                    subclaimText=best_sub_text,
                    superclaimId=best_super_id,
                    superclaimText=best_super_text,
                    confidence=conf,
                    reason=reason,
                )
            ],
            proposals,
        )

    # Propose a new subclaim (text distilled from paragraph) under best superclaim.
    proposals.append(
        (
            "new_subclaim",
            {
                "subclaimText": paragraph,
                "suggestedSuperclaimId": best_super_id,
                "suggestedSuperclaimText": best_super_text,
                "basedOnCandidateSubclaimId": best_sub_id,
                "basedOnCandidateSubclaimText": best_sub_text,
                "confidence": conf,
                "verdict": verdict,
            },
            reason or "Low confidence mapping; propose creating a new subclaim instead.",
        )
    )
    return ([], proposals)


app = FastAPI(title="Claim Mapper API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # Browsers reject `Access-Control-Allow-Origin: *` together with credentialed fetches.
    # This API does not rely on cookies; keep credentials off so wildcard origins work.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "bundleVersion": _bundle_fingerprint(),
        "supabase": _supabase_enabled(),
        "claimsStorage": _claims_storage_enabled(),
        "claimsBucket": SUPABASE_CLAIMS_BUCKET or None,
        "claimsPrefix": SUPABASE_CLAIMS_PREFIX or None,
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    codebook, superclaims, claim_map = _load_taxonomy()
    bundle = _bundle_fingerprint()

    paragraphs = split_into_paragraphs(req.text)
    sub_items = list(codebook.items())
    super_items = list(superclaims.items())

    results: List[AnalyzeParagraphResult] = []
    for p in paragraphs:
        sub_cands = _tfidf_topk(p, sub_items, k=req.max_candidates)
        super_cands = _tfidf_topk(p, super_items, k=max(6, min(req.max_candidates, 12)))

        matches, proposal_specs = _llm_suggest_mapping(
            paragraph=p,
            sub_candidates=sub_cands,
            super_candidates=super_cands,
            claim_map=claim_map,
            merge_pair_min_cosine=req.merge_pair_min_cosine,
            merge_max_pairs=req.merge_max_pairs,
            propose_new_if_below=req.propose_new_if_below,
        )

        stored_props: List[Proposal] = []
        for ptype, payload, rationale in proposal_specs:
            stored_props.append(
                _store_new_proposal(ptype=ptype, paragraph=p, payload=payload, rationale=rationale)
            )

        results.append(
            AnalyzeParagraphResult(paragraph=p, matches=matches, proposals=stored_props)
        )

    return AnalyzeResponse(bundleVersion=bundle, paragraphs=results)


@app.get("/api/proposals", response_model=List[Proposal])
def list_proposals(status: Optional[str] = None) -> List[Proposal]:
    doc = _load_proposals()
    out: List[Proposal] = []
    for row in doc.get("proposals", []):
        if not isinstance(row, dict):
            continue
        p = _proposal_from_row(row)
        if status and p.status != status:
            continue
        out.append(p)
    out.sort(key=lambda x: x.createdAt, reverse=True)
    return out


def _append_merge_event(p: Proposal) -> None:
    if _supabase_enabled():
        sb = _get_supabase()
        try:
            sb.table("taxonomy_merge_log").insert(
                {
                    "proposal_id": p.id,
                    "merge_type": p.type,
                    "bundle_version": p.bundleVersion,
                    "payload": {
                        **p.payload,
                        "appliedBy": p.appliedBy,
                        "appliedAt": p.appliedAt,
                        "reviewedBy": p.reviewedBy,
                        "reviewedAt": p.reviewedAt,
                    },
                }
            ).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase merge log insert failed: {e}") from e
        return

    doc = _read_json(MERGES_PATH) if MERGES_PATH.exists() else {"merges": []}
    if not isinstance(doc, dict) or not isinstance(doc.get("merges"), list):
        doc = {"merges": []}
    doc["merges"].append(
        {
            "id": p.id,
            "type": p.type,
            "createdAt": p.createdAt,
            "bundleVersion": p.bundleVersion,
            "payload": {
                **p.payload,
                "appliedBy": p.appliedBy,
                "appliedAt": p.appliedAt,
                "reviewedBy": p.reviewedBy,
                "reviewedAt": p.reviewedAt,
            },
        }
    )
    _write_json_atomic(MERGES_PATH, doc)


def _merge_claim_histories(*, canonical: str, remove: str) -> None:
    hist = _read_claim_json(HISTORY_NAME)
    if not isinstance(hist, dict):
        raise HTTPException(status_code=500, detail="greenwashing_claim_history.json must be an object")

    claims = hist.get("claims")
    if not isinstance(claims, dict):
        raise HTTPException(status_code=500, detail="greenwashing_claim_history.json must contain a claims object")

    canon_key = canonical if canonical in claims else canonical.replace("NC_", "", 1)
    remove_key = remove if remove in claims else remove.replace("NC_", "", 1)

    src = claims.get(remove_key)
    dst = claims.get(canon_key)
    if not isinstance(dst, dict):
        dst = {"history": []}
        claims[canon_key] = dst

    if isinstance(src, dict):
        src_hist = src.get("history")
        if isinstance(src_hist, list) and src_hist:
            dst_hist = dst.get("history")
            if not isinstance(dst_hist, list):
                dst_hist = []
            dst_hist.extend(src_hist)
            dst["history"] = dst_hist

    if remove_key in claims:
        del claims[remove_key]

    hist["claims"] = claims
    _upload_claim_json(HISTORY_NAME, hist)


def _get_proposal_or_404(pid: str) -> Proposal:
    if _supabase_enabled():
        sb = _get_supabase()
        try:
            resp = sb.table("taxonomy_proposals").select("*").eq("id", pid).limit(1).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase read proposal failed: {e}") from e
        rows = getattr(resp, "data", None) or []
        if not rows or not isinstance(rows[0], dict):
            raise HTTPException(status_code=404, detail="Proposal not found")
        return _proposal_from_row(_proposal_row_to_dict(rows[0]))

    doc = _load_proposals()
    for row in doc.get("proposals", []):
        if isinstance(row, dict) and row.get("id") == pid:
            return _proposal_from_row(row)
    raise HTTPException(status_code=404, detail="Proposal not found")


@app.post("/api/proposals/{proposal_id}/approve", response_model=ProposalActionResponse)
def approve_proposal(proposal_id: str, req: ReviewerActionRequest) -> ProposalActionResponse:
    p = _get_proposal_or_404(proposal_id)
    p.status = "approved"
    p.reviewedBy = req.reviewer_name.strip()
    p.reviewedAt = time.time()
    p = _upsert_proposal(p)
    return ProposalActionResponse(ok=True, proposal=p)


@app.post("/api/proposals/{proposal_id}/reject", response_model=ProposalActionResponse)
def reject_proposal(proposal_id: str, req: ReviewerActionRequest) -> ProposalActionResponse:
    p = _get_proposal_or_404(proposal_id)
    p.status = "rejected"
    p.reviewedBy = req.reviewer_name.strip()
    p.reviewedAt = time.time()
    p = _upsert_proposal(p)
    return ProposalActionResponse(ok=True, proposal=p)


@app.post("/api/proposals/{proposal_id}/apply", response_model=ProposalActionResponse)
def apply_proposal(proposal_id: str, req: ReviewerActionRequest) -> ProposalActionResponse:
    p = _get_proposal_or_404(proposal_id)
    if p.status != "approved":
        raise HTTPException(status_code=400, detail="Proposal must be approved before applying.")

    codebook, superclaims, claim_map = _load_taxonomy()

    if p.type == "new_superclaim":
        text = str(p.payload.get("superclaimText") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Missing superclaimText")
        new_id = _next_id(list(superclaims.keys()), "SC_")
        superclaims[new_id] = text
        _upload_claim_json(SUPERCLAIMS_NAME, dict(sorted(superclaims.items())))

    elif p.type == "new_subclaim":
        text = str(p.payload.get("subclaimText") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Missing subclaimText")
        suggested_sc = _normalize_superclaim_id(str(p.payload.get("suggestedSuperclaimId") or "").strip())
        if not suggested_sc:
            raise HTTPException(status_code=400, detail="Missing suggestedSuperclaimId")
        if suggested_sc not in superclaims:
            # If the suggested superclaim is missing, create it as well.
            suggested_sc = _next_id(list(superclaims.keys()), "SC_")
            superclaims[suggested_sc] = str(p.payload.get("suggestedSuperclaimText") or "New superclaim").strip()
            _upload_claim_json(SUPERCLAIMS_NAME, dict(sorted(superclaims.items())))

        new_nc = _next_id(list(codebook.keys()), "NC_")
        codebook[new_nc] = text
        claim_map[new_nc] = suggested_sc
        _upload_claim_json(CODEBOOK_NAME, dict(sorted(codebook.items())))
        _upload_claim_json(MAP_NAME, dict(sorted(claim_map.items())))

    elif p.type == "merge_subclaims":
        canonical = _normalize_subclaim_id(str(p.payload.get("canonicalSubclaimId") or "").strip())
        remove = _normalize_subclaim_id(str(p.payload.get("removeSubclaimId") or "").strip())
        if not canonical or not remove or canonical == remove:
            raise HTTPException(status_code=400, detail="Missing canonical/remove subclaim ids")

        if remove not in codebook:
            raise HTTPException(status_code=400, detail=f"Unknown subclaim to remove: {remove}")
        if canonical not in codebook:
            raise HTTPException(status_code=400, detail=f"Unknown canonical subclaim: {canonical}")

        sc_canon = _normalize_superclaim_id(str(claim_map.get(canonical, "")).strip())
        sc_remove = _normalize_superclaim_id(str(claim_map.get(remove, "")).strip())
        if not sc_canon or not sc_remove or sc_canon != sc_remove:
            raise HTTPException(
                status_code=400,
                detail="Refusing merge: subclaims are not mapped to the same superclaim in the current map.",
            )

        del codebook[remove]
        if remove in claim_map:
            del claim_map[remove]

        _merge_claim_histories(canonical=canonical, remove=remove)
        _upload_claim_json(CODEBOOK_NAME, dict(sorted(codebook.items())))
        _upload_claim_json(MAP_NAME, dict(sorted(claim_map.items())))
        _append_merge_event(p)

    elif p.type == "merge_superclaims":
        canonical = _normalize_superclaim_id(str(p.payload.get("canonicalSuperclaimId") or "").strip())
        remove = _normalize_superclaim_id(str(p.payload.get("removeSuperclaimId") or "").strip())
        if not canonical or not remove or canonical == remove:
            raise HTTPException(status_code=400, detail="Missing canonical/remove superclaim ids")

        if remove not in superclaims:
            raise HTTPException(status_code=400, detail=f"Unknown superclaim to remove: {remove}")
        if canonical not in superclaims:
            raise HTTPException(status_code=400, detail=f"Unknown canonical superclaim: {canonical}")

        del superclaims[remove]

        for sid, sc in list(claim_map.items()):
            if _normalize_superclaim_id(str(sc)) == remove:
                claim_map[_normalize_subclaim_id(sid)] = canonical

        _upload_claim_json(SUPERCLAIMS_NAME, dict(sorted(superclaims.items())))
        _upload_claim_json(MAP_NAME, dict(sorted(claim_map.items())))
        _append_merge_event(p)

    elif p.type == "link_subclaim_to_superclaim":
        sub_id = _normalize_subclaim_id(str(p.payload.get("subclaimId") or "").strip())
        sc_id = _normalize_superclaim_id(str(p.payload.get("superclaimId") or "").strip())
        if not sub_id or not sc_id:
            raise HTTPException(status_code=400, detail="Missing subclaimId/superclaimId")
        if sub_id not in codebook:
            raise HTTPException(status_code=400, detail=f"Unknown subclaim: {sub_id}")
        if sc_id not in superclaims:
            raise HTTPException(status_code=400, detail=f"Unknown superclaim: {sc_id}")
        claim_map[sub_id] = sc_id
        _upload_claim_json(MAP_NAME, dict(sorted(claim_map.items())))

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported proposal type: {p.type}")

    # Mark applied by flipping status to rejected? Better: keep approved, but add an applied flag.
    now = time.time()
    p.payload = {**p.payload, "appliedAt": now}
    p.appliedBy = req.reviewer_name.strip()
    p.appliedAt = now
    p = _upsert_proposal(p)
    return ProposalActionResponse(ok=True, proposal=p)

