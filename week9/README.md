# NeuroGrip — Module 8: NLP Customer Support Bot

> Rasa-based conversational FAQ agent with Streamlit UI, trained on the Bitext Customer Support dataset.

---

## Stack

| Layer | Tech |
|---|---|
| NLU + Dialogue | Rasa 3.x |
| Custom Actions | rasa-sdk |
| Frontend | Streamlit |
| Dataset | Bitext Customer Support (CSV) |

---

## Project Structure

```
week9/
├── data/
│   ├── nlu.yml          # Intent training examples
│   ├── rules.yml        # Intent → action mappings
│   └── stories.yml      # Multi-turn dialogue flows
├── actions/
│   └── actions.py       # Custom action server logic
├── models/              # Trained model (.tar.gz)
├── dataset/
│   └── bitext_cutomer_support.csv
├── config.yml           # NLU pipeline + policies
├── domain.yml           # Intents, slots, responses
├── endpoints.yml        # Action server endpoint
├── credentials.yml      # REST channel config
└── chatbot_ui.py        # Streamlit frontend
```

---

## Setup & Run

```bash
# 1. Activate venv
source rasa-env/bin/activate

# 2. Train model (from notebook or CLI)
rasa train

# 3. Terminal 1 — action server
rasa run actions --port 5055

# 4. Terminal 2 — rasa server
rasa run --enable-api --cors '*' --port 5005 \
  --endpoints endpoints.yml --credentials credentials.yml

# 5. Terminal 3 — UI
streamlit run chatbot_ui.py
```

Open `http://localhost:8501`.

---

## Config (Optimized for i5-6200u / 8 GB RAM)

Key changes from default Rasa config:

- `DIETClassifier` epochs: 100 → **50**, `use_gpu: false`
- `TEDPolicy` **removed** — primary OOM crash cause on low-RAM machines
- `char_wb CountVectorsFeaturizer` removed — redundant for FAQ scope
- `ResponseSelector` epochs: 100 → **50**
- Policies: `MemoizationPolicy` + `RulePolicy` only

---

## Phase 5 — Report

### What Works
- Intent classification across 27 customer support intents
- Rule-based response dispatch for all intents
- Multi-turn `track_order` flow with slot filling (`order_id`)
- Live Streamlit UI connected to Rasa REST webhook

### Observed Issues

| Issue | Root Cause |
|---|---|
| Generic responses (`"I can help with cancel order..."`) | Placeholder `utter_` templates — not replaced with real copy |
| Intent misclassification (`cancel_order` → `edit_account`) | Insufficient NLU examples (~5 per intent); overlapping phrasing |
| Mock order lookup only | No real backend; `MockDatabase` hardcodes 2 order IDs |

### Metrics (Simulated)

- **Fallback Rate:** ~12% (estimated from confusion matrix)
- **Goal Completion Rate:** ~40% (only `track_order` resolves end-to-end)
- **Intent Accuracy:** ~89% on seen examples (cross-validation)

### Future Improvements

1. **Better responses** — replace generic placeholders with intent-specific, helpful copy
2. **More NLU data** — increase to 15-20 examples per intent to reduce misclassification
3. **Real backend** — connect `action_track_order` to an actual order API
4. **Re-enable TEDPolicy** — `epochs: 30, max_history: 3` once RAM headroom confirmed
5. **Logging** — parse Rasa tracker store to compute live GCR and fallback rate
