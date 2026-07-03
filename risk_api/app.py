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
from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from features import BehaviorSignals, FEATURE_NAMES
import voice_bot as vb

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

# Public base URL the voice provider can reach back on (ngrok / your domain).
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://127.0.0.1:8000")

# Which voice provider to use: mock (default) | twilio | bland.
VOICE_PROVIDER = os.environ.get("VOICE_PROVIDER", "mock")


def get_voice_provider() -> "vb.VoiceProvider":
    if VOICE_PROVIDER == "twilio":
        return vb.TwilioVoiceProvider(
            account_sid=os.environ.get("TWILIO_ACCOUNT_SID", ""),
            auth_token=os.environ.get("TWILIO_AUTH_TOKEN", ""),
            from_number=os.environ.get("TWILIO_FROM_NUMBER", ""),
        )
    if VOICE_PROVIDER == "bland":
        return vb.BlandVoiceProvider(api_key=os.environ.get("BLAND_API_KEY", ""))
    return vb.MockVoiceProvider()

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


def score_signals(sig: BehaviorSignals) -> tuple[int, str, str]:
    """Core scorer shared by /score and /orders/validate."""
    model = get_model()["model"]
    x = np.array([sig.to_vector()])
    prob_ghost = float(model.predict_proba(x)[0, 1])
    risk = int(round(prob_ghost * 100))

    if risk < 35:
        return risk, "genuine", "auto_approve"          # dispatch COD normally
    if risk < 65:
        return risk, "review", "soft_confirm_whatsapp"   # send a confirmation
    return risk, "ghost", "voice_verify"                 # trigger AI voice-bot


def signals_from(req) -> BehaviorSignals:
    return BehaviorSignals(
        time_on_page_s=req.time_on_page_s,
        desc_opened=req.desc_opened,
        size_selected=req.size_selected,
        scroll_depth=req.scroll_depth,
        mouse_moves=req.mouse_moves,
        keystrokes=req.keystrokes,
        pastes=req.pastes,
        device_reuse=req.device_reuse,
    )


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    sig = signals_from(req)
    risk, verdict, action = score_signals(sig)
    return ScoreResponse(
        risk=risk,
        verdict=verdict,
        action=action,
        top_factors=top_factors(sig),
        order_id=req.order_id,
    )


# ============================================================================
# Step 3 — order validation + AI voice-bot
# ============================================================================

class ValidateRequest(ScoreRequest):
    customer_name: str = "Customer"
    customer_phone: str = ""
    brand: str = "Urbanic India"
    product: str = "your order"


class OrderView(BaseModel):
    order_id: str
    status: str
    risk: int
    verdict: str
    reason: str
    call_attempts: int
    events: list[str]
    voice_call: dict | None = None


def _view(order: "vb.Order", voice_call: dict | None = None) -> OrderView:
    verdict = ("ghost" if order.risk >= 65
               else "review" if order.risk >= 35 else "genuine")
    return OrderView(
        order_id=order.order_id,
        status=order.status,
        risk=order.risk,
        verdict=verdict,
        reason=order.reason,
        call_attempts=order.call_attempts,
        events=order.events,
        voice_call=voice_call,
    )


@app.post("/orders/validate", response_model=OrderView)
def validate_order(req: ValidateRequest):
    """
    Score an order and route it: genuine -> auto-approve, review -> soft
    WhatsApp confirm, ghost -> place an AI voice validation call.
    """
    sig = signals_from(req)
    risk, verdict, action = score_signals(sig)

    order = vb.new_order(
        order_id=req.order_id,
        customer_name=req.customer_name,
        customer_phone=req.customer_phone,
        brand=req.brand,
        product=req.product,
        risk=risk,
        status="auto_approved",
    )
    order.log(f"scored risk={risk} verdict={verdict}")

    if action == "auto_approve":
        order.status = "auto_approved"
        order.reason = "Low risk — dispatched normally"
        return _view(order)

    if action == "soft_confirm_whatsapp":
        order.status = "soft_confirm"
        order.reason = "Medium risk — WhatsApp confirmation sent"
        order.log("[whatsapp] confirmation message sent (mock)")
        return _view(order)

    # High risk -> voice validation
    order.status = "voice_pending"
    order.reason = "High risk — awaiting voice validation"
    provider = get_voice_provider()
    answer_url = f"{PUBLIC_BASE_URL}/voice/twiml/{order.order_id}"
    call = provider.place_call(order, answer_url)
    return _view(order, voice_call=call)


@app.get("/voice/twiml/{order_id}")
def voice_twiml(order_id: str):
    """TwiML the phone provider fetches when the call connects."""
    order = vb.ORDERS.get(order_id)
    if not order:
        return JSONResponse({"detail": "order not found"}, status_code=404)
    action_url = f"{PUBLIC_BASE_URL}/voice/response/{order_id}"
    xml = vb.build_twiml(order, action_url)
    return Response(content=xml, media_type="application/xml")


@app.post("/voice/response/{order_id}", response_model=OrderView)
async def voice_response(
    order_id: str,
    request: Request,
    Digits: str | None = Form(default=None),
    SpeechResult: str | None = Form(default=None),
):
    """
    Webhook the provider calls with the customer's keypress / speech. Also
    accepts JSON (for the offline simulator). Sets the order confirmed/cancelled.
    """
    order = vb.ORDERS.get(order_id)
    if not order:
        return JSONResponse({"detail": "order not found"}, status_code=404)

    answered = request.query_params.get("answered", "true") != "false"

    # Simulator may POST JSON instead of form fields.
    if Digits is None and SpeechResult is None:
        ctype = request.headers.get("content-type", "")
        if "application/json" in ctype:
            body = await request.json()
            Digits = body.get("digits")
            SpeechResult = body.get("speech")
            answered = body.get("answered", answered)

    analysis = vb.analyse_response(Digits, SpeechResult, answered)
    order.status = analysis.outcome
    order.reason = f"{analysis.reason} (confidence {analysis.confidence}%)"
    order.log(f"[voice] {order.status}: {analysis.reason}")
    return _view(order)


@app.get("/orders/{order_id}", response_model=OrderView)
def get_order(order_id: str):
    order = vb.ORDERS.get(order_id)
    if not order:
        return JSONResponse({"detail": "order not found"}, status_code=404)
    return _view(order)
