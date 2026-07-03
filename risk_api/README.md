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

## From demo to production

- **Real labels:** replace `sample_population()` in `train_model.py` with your
  own order history labelled by outcome (delivered vs RTO). The pipeline is
  unchanged; only the data source swaps.
- **Persist fingerprints server-side** (Redis/Postgres) instead of the browser's
  `localStorage`, so device reuse is counted across users and can't be cleared.
- **Auth + rate limiting** before exposing `/score` publicly.
- `action: "voice_verify"` is the hook for Step 3 (Twilio / Bland AI voice bot).
