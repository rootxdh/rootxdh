"""
Offline end-to-end simulation of the Ghost Shopper voice-validation flow.

Runs against a live API (`uvicorn app:app`) with the default mock provider —
no Twilio/Bland credentials or real phone calls needed. It walks three orders
through the full pipeline and prints what happens at each step:

  1. Genuine buyer   -> auto-approved, no call
  2. Ghost + presses 1 on the call        -> confirmed
  3. Ghost + evasive/silent, phone off    -> cancelled

Usage:
    uvicorn app:app --port 8000     # in one terminal
    python simulate_voice_flow.py   # in another
"""

from __future__ import annotations

import json
import os
import urllib.request

BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path) as r:
        return json.load(r)


def show(title: str, order: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"  order_id : {order['order_id']}")
    print(f"  risk     : {order['risk']}  ({order['verdict']})")
    print(f"  status   : {order['status']}")
    print(f"  reason   : {order['reason']}")
    if order.get("voice_call"):
        vc = order["voice_call"]
        print(f"  voice    : [{vc['provider']}] {vc.get('note', 'call queued')}")
        if vc.get("script"):
            print(f"  script   : \"{vc['script']}\"")


GENUINE = dict(customer_name="Rahul", customer_phone="+91900000001",
               brand="Urbanic India", product="Tan Jutti",
               time_on_page_s=42, desc_opened=True, size_selected=True,
               scroll_depth=90, mouse_moves=40, keystrokes=38,
               pastes=0, device_reuse=0)

GHOST = dict(customer_name="Rahul", customer_phone="+91900000002",
             brand="Urbanic India", product="Tan Jutti",
             time_on_page_s=3, desc_opened=False, size_selected=False,
             scroll_depth=8, mouse_moves=1, keystrokes=10,
             pastes=1, device_reuse=3)


def main() -> None:
    # 1) Genuine buyer -> auto-approved, no call.
    o = post("/orders/validate", GENUINE)
    show("1. Genuine buyer", o)

    # 2) Ghost buyer -> voice call -> customer presses 1 -> confirmed.
    o = post("/orders/validate", GHOST)
    show("2. Ghost buyer (will press 1)", o)
    twiml = urllib.request.urlopen(f"{BASE}/voice/twiml/{o['order_id']}").read().decode()
    print("  --- TwiML the phone would play ---")
    for line in twiml.splitlines():
        print("    " + line)
    o2 = post(f"/voice/response/{o['order_id']}",
              {"digits": "1", "answered": True})
    show("   -> after keypress '1'", o2)

    # 3) Ghost buyer -> voice call -> phone switched off -> cancelled.
    o = post("/orders/validate", GHOST)
    show("3. Ghost buyer (phone off)", o)
    o3 = post(f"/voice/response/{o['order_id']}",
              {"answered": False})
    show("   -> after unanswered call", o3)

    # 4) Ghost buyer -> answers but gives an evasive reply -> cancelled.
    o = post("/orders/validate", GHOST)
    o4 = post(f"/voice/response/{o['order_id']}",
              {"speech": "kaunsa order? maine to kuch order nahi kiya", "answered": True})
    show("4. Ghost buyer (evasive speech)", o4)

    print("\nDone.")


if __name__ == "__main__":
    main()
