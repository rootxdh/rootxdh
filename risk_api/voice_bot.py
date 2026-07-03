"""
Ghost Shopper Detector — Step 3: AI voice-bot validation.

When the risk API flags an order as `voice_verify`, we place an automated
voice call to the customer in a human-sounding voice:

    "Hello Rahul ji, aapne Urbanic India se Tan Jutti order ki hai.
     Kya aap ise sach me lena chahte hain? Confirm karne ke liye 1 dabayein."

The customer's response is analysed two ways:
  1. Keypress (DTMF) — pressed 1 = confirm, 2 = cancel.
  2. Speech — the transcript's tone/language is scored for fake-sounding or
     evasive answers, and no-answer / switched-off phone counts as a fail.

This module is provider-agnostic. `MockVoiceProvider` runs the whole flow
offline (for the demo / tests). `TwilioVoiceProvider` builds real TwiML and
`BlandVoiceProvider` builds a real Bland AI call payload — drop in credentials
to go live. Nothing here dials out on its own.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal
from xml.sax.saxutils import escape

# ----------------------------------------------------------------------------
# Order + call state (in-memory; swap for Redis/Postgres in production)
# ----------------------------------------------------------------------------

OrderStatus = Literal[
    "auto_approved",     # low risk, dispatch normally
    "soft_confirm",      # medium risk, WhatsApp confirmation sent
    "voice_pending",     # high risk, waiting for the customer to respond
    "confirmed",         # customer validated the order
    "cancelled",         # customer declined / failed validation
]


@dataclass
class Order:
    order_id: str
    customer_name: str
    customer_phone: str
    brand: str
    product: str
    risk: int
    status: OrderStatus
    reason: str = ""
    call_attempts: int = 0
    events: list[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.events.append(f"{time.strftime('%H:%M:%S')}  {msg}")


ORDERS: dict[str, Order] = {}


def new_order(**kw) -> Order:
    oid = kw.pop("order_id", None) or f"ord_{uuid.uuid4().hex[:8]}"
    order = Order(order_id=oid, **kw)
    ORDERS[oid] = order
    return order


# ----------------------------------------------------------------------------
# The spoken script
# ----------------------------------------------------------------------------

def call_script(order: Order) -> str:
    """The Hinglish line the bot speaks when the call connects."""
    return (
        f"Hello {order.customer_name} ji. "
        f"Aapne {order.brand} se {order.product} order ki hai, "
        f"cash on delivery par. "
        f"Kya aap ise sach mein lena chahte hain? "
        f"Confirm karne ke liye 1 dabayein. "
        f"Cancel karne ke liye 2 dabayein."
    )


# ----------------------------------------------------------------------------
# Response analysis — is the answer genuine?
# ----------------------------------------------------------------------------

# Words a hesitant / joking / evasive caller tends to use.
_EVASIVE = [
    "pata nahi", "nahi pata", "kaunsa", "kaun", "maine nahi", "galti",
    "mazak", "mazaak", "joke", "test", "timepass", "kisi aur", "wrong number",
    "who is this", "kya", "hmm", "uhh", "abhi busy",
]
_POSITIVE = ["haan", "yes", "bilkul", "confirm", "chahiye", "sahi hai", "ok", "theek"]


@dataclass
class ResponseAnalysis:
    outcome: Literal["confirmed", "cancelled"]
    confidence: int          # 0-100 confidence in the outcome
    reason: str


def analyse_response(
    digits: str | None,
    speech: str | None,
    answered: bool,
) -> ResponseAnalysis:
    """
    Decide the order's fate from the call outcome.

    Priority: an unanswered call fails; an explicit keypress is trusted; only
    then do we fall back to analysing the spoken transcript's tone.
    """
    if not answered:
        return ResponseAnalysis("cancelled", 90,
                                "Call not answered / phone switched off")

    if digits:
        if digits.strip() == "1":
            return ResponseAnalysis("confirmed", 95, "Pressed 1 to confirm")
        if digits.strip() == "2":
            return ResponseAnalysis("cancelled", 95, "Pressed 2 to cancel")
        return ResponseAnalysis("cancelled", 70,
                                f"Unexpected keypress '{digits}'")

    text = (speech or "").strip().lower()
    if not text:
        return ResponseAnalysis("cancelled", 80, "Answered but stayed silent")

    evasive_hits = [w for w in _EVASIVE if w in text]
    positive_hits = [w for w in _POSITIVE if w in text]

    if evasive_hits and not positive_hits:
        return ResponseAnalysis(
            "cancelled", 75,
            f"Evasive / fake-sounding reply ({', '.join(evasive_hits[:2])})")
    if positive_hits and not evasive_hits:
        return ResponseAnalysis(
            "confirmed", 80,
            f"Clear positive reply ({', '.join(positive_hits[:2])})")
    if positive_hits and evasive_hits:
        return ResponseAnalysis(
            "cancelled", 55, "Mixed / uncertain reply — erring on cancel")
    return ResponseAnalysis("cancelled", 60,
                            "Could not detect a clear confirmation")


# ----------------------------------------------------------------------------
# Providers
# ----------------------------------------------------------------------------

class VoiceProvider:
    name = "base"

    def place_call(self, order: Order, answer_url: str) -> dict:
        raise NotImplementedError


class MockVoiceProvider(VoiceProvider):
    """Runs offline. Records intent instead of dialling out."""
    name = "mock"

    def place_call(self, order: Order, answer_url: str) -> dict:
        order.call_attempts += 1
        order.log(f"[mock] would dial {order.customer_phone}; "
                  f"on answer fetch {answer_url}")
        return {
            "provider": self.name,
            "to": order.customer_phone,
            "answer_url": answer_url,
            "script": call_script(order),
            "note": "No real call placed (mock provider).",
        }


class TwilioVoiceProvider(VoiceProvider):
    """
    Builds a real Twilio call request. Needs account SID / auth token / from
    number to actually dial; without them we return the request we *would* send.
    """
    name = "twilio"

    def __init__(self, account_sid="", auth_token="", from_number=""):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number

    def place_call(self, order: Order, answer_url: str) -> dict:
        order.call_attempts += 1
        req = {
            "provider": self.name,
            "endpoint": f"https://api.twilio.com/2010-04-01/Accounts/"
                        f"{self.account_sid or '<SID>'}/Calls.json",
            "params": {
                "To": order.customer_phone,
                "From": self.from_number or "<FROM_NUMBER>",
                "Url": answer_url,        # Twilio fetches TwiML from here
                "Method": "GET",
            },
        }
        if not (self.account_sid and self.auth_token and self.from_number):
            req["note"] = "Credentials not set — returning the request only."
            order.log("[twilio] credentials missing; call not placed")
        else:
            order.log(f"[twilio] placing call to {order.customer_phone}")
        return req


class BlandVoiceProvider(VoiceProvider):
    """Builds a Bland AI (bland.ai) call payload — human-like TTS + transcript."""
    name = "bland"

    def __init__(self, api_key=""):
        self.api_key = api_key

    def place_call(self, order: Order, answer_url: str) -> dict:
        order.call_attempts += 1
        payload = {
            "provider": self.name,
            "endpoint": "https://api.bland.ai/v1/calls",
            "body": {
                "phone_number": order.customer_phone,
                "task": call_script(order),
                "voice": "hindi-female",
                "webhook": answer_url,
                "wait_for_greeting": True,
            },
        }
        if not self.api_key:
            payload["note"] = "BLAND_API_KEY not set — returning payload only."
            order.log("[bland] api key missing; call not placed")
        return payload


# ----------------------------------------------------------------------------
# TwiML for the call — spoken prompt + gather keypress/speech
# ----------------------------------------------------------------------------

def build_twiml(order: Order, action_url: str) -> str:
    """
    XML Twilio plays when the call connects. Gathers a keypress OR speech, then
    POSTs the result to `action_url`.
    """
    script = escape(call_script(order))
    action = escape(action_url, {'"': "&quot;"})
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<Response>\n"
        f'  <Gather input="dtmf speech" numDigits="1" timeout="6" '
        f'language="hi-IN" action="{action}" method="POST">\n'
        f'    <Say language="hi-IN">{script}</Say>\n'
        "  </Gather>\n"
        # Fallback if nothing was captured.
        f'  <Redirect method="POST">{action}?answered=false</Redirect>\n'
        "</Response>"
    )
