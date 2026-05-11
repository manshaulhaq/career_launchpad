# NLP Customer Support Bot
Rasa 3.x + Streamlit FAQ agent trained on Bitext Customer Support dataset.

## Run
```bash
source rasa-env/bin/activate
rasa run actions --port 5055
rasa run --enable-api --cors '*' --port 5005 --endpoints endpoints.yml --credentials credentials.yml
streamlit run chatbot_ui.py  # → http://localhost:8501
```

## Phase 5 — Report

**What works:** 27 intents classified, rule-based dispatch, `track_order` multi-turn flow with slot filling, Streamlit UI on REST webhook.

**Issues:**
- Generic placeholder responses per intent
- Misclassification between similar intents (~5 training examples each)
- Mock order DB only — no real backend

**Metrics (simulated):** Intent accuracy ~89% · GCR ~40% · Fallback rate ~12%

**Improvements:** More NLU examples per intent · intent-specific responses · real order API · tracker store logging for live metrics