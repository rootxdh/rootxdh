# Ghost Shopper Detector — Risk API (Step 2)

The backend brain of the Ghost Shopper Detector. It takes the behaviour
signals captured by the browser tracker (`../ghost-shopper-demo.html`) and
returns a **risk probability**, a **verdict**, and a **recommended action** for
each COD order.

```
Browser tracker  ──POST /score──▶  FastAPI + scikit-learn  ──▶  { risk, verdict, action }
(scroll, time,                                                    │
 device reuse…)                                                   ▼
                                              high risk → AI voice-bot (Step 3)
```

## Files

| File | What it does |
|------|--------------|
| `features.py`    | Single source of truth for the feature list + a `BehaviorSignals` helper. Imported by both training and serving so the feature order can't drift. |
| `train_model.py` | Builds a synthetic dataset (genuine vs ghost buyers), trains a calibrated logistic-regression model, saves `model.joblib`. |
| `app.py`         | FastAPI service exposing `POST /score` and `GET /health`. |
| `requirements.txt` | Pinned dependencies. |

## Quick start

```bash
cd risk_api
pip install -r requirements.txt

python train_model.py          # -> writes model.joblib (ROC-AUC ~0.90)
uvicorn app:app --reload       # -> http://127.0.0.1:8000/docs
```

## Scoring an order

```bash
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "time_on_page_s": 3,
    "desc_opened": false,
    "size_selected": false,
    "scroll_depth": 8,
    "mouse_moves": 1,
    "keystrokes": 10,
    "pastes": 1,
    "device_reuse": 3,
    "order_id": "SHOP-1042"
  }'
```

```json
{
  "risk": 95,
  "verdict": "ghost",
  "action": "voice_verify",
  "top_factors": [
    "Ordered in 3.0s (very fast)",
    "Description never opened",
    "No size selected",
    "Barely scrolled (8%)"
  ],
  "order_id": "SHOP-1042"
}
```

## Verdict bands → actions

| Risk score | Verdict   | Action                   | What you'd do in production |
|-----------:|-----------|--------------------------|-----------------------------|
| `0–34`     | `genuine` | `auto_approve`           | Dispatch COD normally |
| `35–64`    | `review`  | `soft_confirm_whatsapp`  | Send a WhatsApp confirmation |
| `65–100`   | `ghost`   | `voice_verify`           | Trigger the AI voice-bot before dispatch |

## Features the model uses

`time_on_page_s`, `desc_opened`, `size_selected`, `scroll_depth`,
`mouse_moves`, `keystrokes`, `pastes`, `device_reuse`.

These map 1:1 to what the browser tracker collects.

## Wiring the demo page to this API

The tracker currently scores in-browser with transparent rules. To use this
model instead, POST the collected signals when the order is confirmed:

```js
const res = await fetch("http://127.0.0.1:8000/score", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    time_on_page_s: s.timeOnPage / 1000,
    desc_opened: s.descOpened,
    size_selected: s.sizeSelected,
    scroll_depth: s.scrollDepth,
    mouse_moves: s.mouseMoves,
    keystrokes: s.keystrokes,
    pastes: s.pastes,
    device_reuse: reuseCount,
    order_id: orderId,
  }),
});
const { risk, verdict, action } = await res.json();
```

## Step 3 — AI voice-bot validation

When an order scores in the `ghost` band, instead of dispatching it we place an
automated, human-sounding voice call to confirm intent. Files:

- `voice_bot.py` — order/call state, the Hinglish call script, response
  analysis, and three providers (`mock`, `twilio`, `bland`).
- New endpoints in `app.py` (below).
- `simulate_voice_flow.py` — runs the whole thing offline with the mock provider.

### Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /orders/validate` | Score an order and route it: genuine → auto-approve, review → WhatsApp soft-confirm, ghost → place a voice call. |
| `GET /voice/twiml/{id}` | TwiML the phone provider fetches when the call connects (spoken prompt + gather keypress/speech). |
| `POST /voice/response/{id}` | Webhook the provider hits with the customer's keypress (`Digits`) / speech (`SpeechResult`). Sets the order confirmed/cancelled. |
| `GET /orders/{id}` | Current status + event log for an order. |

### The call

> "Hello Rahul ji. Aapne Urbanic India se Tan Jutti order ki hai, cash on
> delivery par. Kya aap ise sach mein lena chahte hain? Confirm karne ke liye
> 1 dabayein. Cancel karne ke liye 2 dabayein."

The response is judged in priority order: an unanswered / switched-off phone
fails; an explicit keypress (`1` confirm / `2` cancel) is trusted; otherwise the
spoken transcript's tone is scored (evasive/joking words → cancel, clear
positive → confirm).

### Try it offline (no Twilio needed)

```bash
python train_model.py
uvicorn app:app --port 8000        # terminal 1
python simulate_voice_flow.py      # terminal 2
```

Output walks a genuine order (auto-approved), a ghost who presses 1 (confirmed),
a ghost whose phone is off (cancelled), and a ghost who gives an evasive reply
(cancelled).

### Going live

Set env vars before starting uvicorn:

```bash
export PUBLIC_BASE_URL="https://your-domain.com"   # provider calls back here
export VOICE_PROVIDER="twilio"                      # or "bland"
export TWILIO_ACCOUNT_SID=... TWILIO_AUTH_TOKEN=... TWILIO_FROM_NUMBER=...
# or:  export BLAND_API_KEY=...
```

With the mock provider (default) nothing dials out — the API returns the exact
request it *would* send, so the flow is fully testable without credentials.

## From demo to production

- **Real labels:** replace `sample_population()` in `train_model.py` with your
  own order history labelled by outcome (delivered vs RTO). The pipeline is
  unchanged; only the data source swaps.
- **Persist fingerprints server-side** (Redis/Postgres) instead of the browser's
  `localStorage`, so device reuse is counted across users and can't be cleared.
- **Auth + rate limiting** before exposing `/score` publicly.
- `action: "voice_verify"` is the hook for Step 3 (Twilio / Bland AI voice bot).
