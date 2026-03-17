from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm_confidence import score_subclaim_to_superclaim_confidence


app = FastAPI(title="Claims MVP Backend", version="0.1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConfidenceRequest(BaseModel):
    subclaim_text: str = Field(min_length=1)
    superclaim_text: str = Field(min_length=1)
    subclaim_id: str | None = None
    superclaim_id: str | None = None


class ConfidenceResponse(BaseModel):
    verdict: str
    confidence: float
    reason: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/confidence", response_model=ConfidenceResponse)
def confidence(req: ConfidenceRequest):
    # This endpoint must never 500; the UI expects a numeric confidence for every mapping.
    try:
        result = score_subclaim_to_superclaim_confidence(
            subclaim_text=req.subclaim_text,
            superclaim_text=req.superclaim_text,
            subclaim_id=req.subclaim_id,
            superclaim_id=req.superclaim_id,
        )
        return ConfidenceResponse(**result)
    except Exception as e:
        # Log the error so we can inspect it in deployment logs (e.g. Vercel).
        print("LLM error in /confidence endpoint:", e)
        try:
            return ConfidenceResponse(
                verdict="uncertain",
                confidence=0.0,
                reason=f"Backend/LLM error: {e}",
            )
        except Exception:
            # Absolute last resort: return a plain dict that still matches the schema.
            return {"verdict": "uncertain", "confidence": 0.0, "reason": "Backend/LLM error"}

