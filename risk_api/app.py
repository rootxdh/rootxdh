"""
Ghost Shopper Detector — risk scoring API.

POST behaviour signals collected by the browser tracker, get back a risk
probability, a verdict, and a recommended action. High-risk orders are the
ones you'd route to AI voice-bot validation (Step 3) before dispatch.

Run:  uvicorn app:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import os

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from features import BehaviorSignals, FEATURE_NAMES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

app = FastAPI(
    title="Ghost Shopper Detector — Risk API",
    version="1.0.0",
    description="Behavioral fraud scoring for COD orders.",
)

# Allow the demo page / a Shopify storefront to call this from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_bundle = None


def get_model():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError("model.joblib not found — run train_model.py first")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class ScoreRequest(BaseModel):
    time_on_page_s: float = Field(0, ge=0, description="Seconds before confirming order")
    desc_opened: bool = False
    size_selected: bool = False
    scroll_depth: float = Field(0, ge=0, le=100, description="Max scroll depth %")
    mouse_moves: int = Field(0, ge=0)
    keystrokes: int = Field(0, ge=0)
    pastes: int = Field(0, ge=0)
    device_reuse: int = Field(0, ge=0, description="Prior orders from same device")
    order_id: str | None = None


class ScoreResponse(BaseModel):
    risk: int = Field(..., description="0-100, higher = more likely a ghost buyer")
    verdict: str
    action: str
    top_factors: list[str]
    order_id: str | None = None


def top_factors(sig: BehaviorSignals) -> list[str]:
    """Human-readable reasons — what pushed the score up."""
    reasons = []
    if sig.time_on_page_s < 6:
        reasons.append(f"Ordered in {sig.time_on_page_s:.1f}s (very fast)")
    if not sig.desc_opened:
        reasons.append("Description never opened")
    if not sig.size_selected:
        reasons.append("No size selected")
    if sig.scroll_depth < 25:
        reasons.append(f"Barely scrolled ({sig.scroll_depth:.0f}%)")
    if sig.mouse_moves < 5:
        reasons.append("Almost no mouse movement")
    if sig.pastes > 0:
        reasons.append(f"Pasted form fields ({sig.pastes}x)")
    if sig.device_reuse >= 2:
        reasons.append(f"Device reused {sig.device_reuse}x")
    return reasons[:4]


@app.get("/health")
def health():
    try:
        get_model()
        return {"status": "ok", "features": FEATURE_NAMES}
    except RuntimeError as e:
        return {"status": "model_missing", "detail": str(e)}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    bundle = get_model()
    model = bundle["model"]

    sig = BehaviorSignals(
        time_on_page_s=req.time_on_page_s,
        desc_opened=req.desc_opened,
        size_selected=req.size_selected,
        scroll_depth=req.scroll_depth,
        mouse_moves=req.mouse_moves,
        keystrokes=req.keystrokes,
        pastes=req.pastes,
        device_reuse=req.device_reuse,
    )

    x = np.array([sig.to_vector()])
    prob_ghost = float(model.predict_proba(x)[0, 1])
    risk = int(round(prob_ghost * 100))

    if risk < 35:
        verdict = "genuine"
        action = "auto_approve"          # dispatch COD normally
    elif risk < 65:
        verdict = "review"
        action = "soft_confirm_whatsapp"  # send a confirmation message
    else:
        verdict = "ghost"
        action = "voice_verify"           # trigger AI voice-bot before dispatch

    return ScoreResponse(
        risk=risk,
        verdict=verdict,
        action=action,
        top_factors=top_factors(sig),
        order_id=req.order_id,
    )
