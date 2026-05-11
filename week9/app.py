import streamlit as st
import requests
import uuid
import time

# ── Config ────────────────────────────────────────────────────────────────────
RASA_URL = "http://localhost:5005/webhooks/rest/webhook"

st.set_page_config(
    page_title="Support Bot",
    page_icon="🤖",
    layout="centered"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0f0f0f;
    color: #e8e8e8;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 6rem 2rem; max-width: 720px; }

/* Title */
.chat-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #00ff88;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-bottom: 1px solid #1e1e1e;
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
}
.chat-subtitle {
    font-size: 0.78rem;
    color: #555;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: -1rem;
    margin-bottom: 1.5rem;
}

/* Message bubbles */
.msg-row { display: flex; margin-bottom: 1rem; align-items: flex-end; gap: 0.6rem; }
.msg-row.user { flex-direction: row-reverse; }

.bubble {
    padding: 0.65rem 1rem;
    border-radius: 16px;
    max-width: 78%;
    font-size: 0.9rem;
    line-height: 1.5;
    word-wrap: break-word;
}
.bubble.user {
    background: #00ff88;
    color: #0f0f0f;
    border-bottom-right-radius: 4px;
    font-weight: 500;
}
.bubble.bot {
    background: #1a1a1a;
    color: #e8e8e8;
    border: 1px solid #2a2a2a;
    border-bottom-left-radius: 4px;
}

.avatar {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
}
.avatar.bot { background: #1a1a1a; border: 1px solid #2a2a2a; }
.avatar.user { background: #00ff88; color: #0f0f0f; }

.timestamp {
    font-size: 0.65rem;
    color: #444;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.2rem;
    padding: 0 0.5rem;
}
.ts-right { text-align: right; }

/* Status badge */
.status-bar {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #444;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 1.5rem;
}
.dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00ff88;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Typing indicator */
.typing { display: flex; gap: 4px; padding: 0.65rem 1rem; }
.typing span {
    width: 6px; height: 6px;
    background: #444;
    border-radius: 50%;
    animation: bounce 1.2s infinite;
}
.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-6px); }
}

/* Input area */
.stTextInput > div > div > input {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    color: #e8e8e8 !important;
    border-radius: 12px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #00ff88 !important;
    box-shadow: 0 0 0 1px #00ff8833 !important;
}
.stButton > button {
    background: #00ff88 !important;
    color: #0f0f0f !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    padding: 0.65rem 1.4rem !important;
    letter-spacing: 0.05em !important;
    cursor: pointer !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.clear-btn > button {
    background: transparent !important;
    color: #444 !important;
    border: 1px solid #2a2a2a !important;
    font-size: 0.75rem !important;
    padding: 0.4rem 0.8rem !important;
}
.clear-btn > button:hover { color: #ff4444 !important; border-color: #ff4444 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rasa_ok" not in st.session_state:
    st.session_state.rasa_ok = False
if "input_key" not in st.session_state:
    st.session_state.input_key = 0
if "pending" not in st.session_state:
    st.session_state.pending = None


# ── Rasa health check ─────────────────────────────────────────────────────────
def check_rasa():
    try:
        r = requests.get("http://localhost:5005/", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def send_message(text: str) -> list[str]:
    try:
        payload = {"sender": st.session_state.session_id, "message": text}
        r = requests.post(RASA_URL, json=payload, timeout=10)
        r.raise_for_status()
        responses = r.json()
        return [msg.get("text", "") for msg in responses if msg.get("text")]
    except requests.exceptions.ConnectionError:
        return ["⚠️ Cannot reach Rasa server. Is it running on port 5005?"]
    except Exception as e:
        return [f"⚠️ Error: {str(e)}"]


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="chat-title">⬡ &nbsp;Support Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="chat-subtitle">// powered by rasa · localhost:5005</div>', unsafe_allow_html=True)

# Status
rasa_ok = check_rasa()
if rasa_ok:
    st.markdown('<div class="status-bar"><div class="dot"></div> RASA SERVER ONLINE</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-bar"><div class="dot" style="background:#ff4444;animation:none"></div> RASA SERVER OFFLINE — run: <code>rasa run --enable-api --cors "*"</code></div>', unsafe_allow_html=True)


# ── Chat history ──────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div class="msg-row">
            <div class="avatar bot">🤖</div>
            <div>
                <div class="bubble bot">Hello! I'm your support assistant. Ask me about orders, refunds, shipping, account issues, and more.</div>
                <div class="timestamp">now</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        role = msg["role"]
        text = msg["text"]
        ts = msg.get("time", "")
        if role == "user":
            st.markdown(f"""
            <div class="msg-row user">
                <div class="avatar user">U</div>
                <div>
                    <div class="bubble user">{text}</div>
                    <div class="timestamp ts-right">{ts}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-row">
                <div class="avatar bot">🤖</div>
                <div>
                    <div class="bubble bot">{text}</div>
                    <div class="timestamp">{ts}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        label="message",
        placeholder="Type your message...",
        label_visibility="collapsed",
        key=f"input_box_{st.session_state.input_key}"
    )
with col2:
    send = st.button("SEND")

col3, col4 = st.columns([5, 1])
with col4:
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("clear"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.input_key += 1
        st.session_state.pending = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ── Send logic ────────────────────────────────────────────────────────────────
# Stage 1: capture input, clear box, rerun
if (send or user_input) and user_input.strip() and st.session_state.pending is None:
    st.session_state.pending = user_input.strip()
    st.session_state.input_key += 1  # clears the input box on next render
    st.rerun()

# Stage 2: pending message exists — send to Rasa, append, clear pending
if st.session_state.pending:
    text = st.session_state.pending
    ts = time.strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "text": text, "time": ts})

    with st.spinner(""):
        replies = send_message(text)

    for reply in replies:
        st.session_state.messages.append({
            "role": "bot",
            "text": reply,
            "time": time.strftime("%H:%M")
        })

    st.session_state.pending = None
    st.rerun()