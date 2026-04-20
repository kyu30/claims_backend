import hashlib
import json
import os
import time
import uuid
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

CODEBOOK_PATH = ROOT / "greenwashing_codebook.json"
SUPERCLAIMS_PATH = ROOT / "greenwashing_superclaims.json"
MAP_PATH = ROOT / "claim_superclaim_map.json"
HISTORY_PATH = ROOT / "greenwashing_claim_history.json"

APP_DATA_DIR = ROOT / "data"
PROPOSALS_PATH = APP_DATA_DIR / "proposals.json"
MERGES_PATH = APP_DATA_DIR / "merges.json"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Missing required file: {path.name}") from e
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON in {path.name}: {e}") from e


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _bundle_fingerprint() -> str:
    h = hashlib.sha256()
    for p in (CODEBOOK_PATH, SUPERCLAIMS_PATH, MAP_PATH, HISTORY_PATH):
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
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
    codebook = _read_json(CODEBOOK_PATH)
    superclaims = _read_json(SUPERCLAIMS_PATH)
    claim_map = _read_json(MAP_PATH)

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


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Raw pasted article text.")
    max_candidates: int = Field(10, ge=1, le=30)
    propose_new_if_below: float = Field(0.55, ge=0.0, le=1.0)


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


def _load_proposals() -> Dict[str, Any]:
    if not PROPOSALS_PATH.exists():
        return {"proposals": []}
    data = _read_json(PROPOSALS_PATH)
    if not isinstance(data, dict) or not isinstance(data.get("proposals"), list):
        return {"proposals": []}
    return data


def _save_proposals(doc: Dict[str, Any]) -> None:
    _write_json_atomic(PROPOSALS_PATH, doc)


def _upsert_proposal(p: Proposal) -> Proposal:
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
    propose_new_if_below: float,
) -> Tuple[List[MatchRow], List[Tuple[ProposalType, Dict[str, Any], str]]]:
    """
    Minimal “B” behavior:
    - Pick best existing (subclaim, superclaim) by TF‑IDF and then verify with LLM confidence scorer.
    - If LLM confidence is low, propose a new subclaim under the best superclaim (or propose new superclaim if none fit).
    - Also optionally propose merges if paragraph strongly matches multiple subclaims under same superclaim (not implemented here yet).
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
            [],
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
    return {"ok": True, "bundleVersion": _bundle_fingerprint()}


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    codebook, superclaims, _claim_map = _load_taxonomy()
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


def _get_proposal_or_404(pid: str) -> Proposal:
    doc = _load_proposals()
    for row in doc.get("proposals", []):
        if isinstance(row, dict) and row.get("id") == pid:
            return _proposal_from_row(row)
    raise HTTPException(status_code=404, detail="Proposal not found")


@app.post("/api/proposals/{proposal_id}/approve", response_model=ProposalActionResponse)
def approve_proposal(proposal_id: str) -> ProposalActionResponse:
    p = _get_proposal_or_404(proposal_id)
    p.status = "approved"
    p = _upsert_proposal(p)
    return ProposalActionResponse(ok=True, proposal=p)


@app.post("/api/proposals/{proposal_id}/reject", response_model=ProposalActionResponse)
def reject_proposal(proposal_id: str) -> ProposalActionResponse:
    p = _get_proposal_or_404(proposal_id)
    p.status = "rejected"
    p = _upsert_proposal(p)
    return ProposalActionResponse(ok=True, proposal=p)


@app.post("/api/proposals/{proposal_id}/apply", response_model=ProposalActionResponse)
def apply_proposal(proposal_id: str) -> ProposalActionResponse:
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
        _write_json_atomic(SUPERCLAIMS_PATH, dict(sorted(superclaims.items())))

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
            _write_json_atomic(SUPERCLAIMS_PATH, dict(sorted(superclaims.items())))

        new_nc = _next_id(list(codebook.keys()), "NC_")
        codebook[new_nc] = text
        claim_map[new_nc] = suggested_sc
        _write_json_atomic(CODEBOOK_PATH, dict(sorted(codebook.items())))
        _write_json_atomic(MAP_PATH, dict(sorted(claim_map.items())))

    elif p.type in ("merge_subclaims", "merge_superclaims"):
        # Minimal bookkeeping: store merge request; do not rewrite your canonical JSON automatically.
        doc = _read_json(MERGES_PATH) if MERGES_PATH.exists() else {"merges": []}
        if not isinstance(doc, dict) or not isinstance(doc.get("merges"), list):
            doc = {"merges": []}
        doc["merges"].append(
            {
                "id": p.id,
                "type": p.type,
                "createdAt": p.createdAt,
                "bundleVersion": p.bundleVersion,
                "payload": p.payload,
            }
        )
        _write_json_atomic(MERGES_PATH, doc)

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
        _write_json_atomic(MAP_PATH, dict(sorted(claim_map.items())))

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported proposal type: {p.type}")

    # Mark applied by flipping status to rejected? Better: keep approved, but add an applied flag.
    p.payload = {**p.payload, "appliedAt": time.time()}
    p = _upsert_proposal(p)
    return ProposalActionResponse(ok=True, proposal=p)

