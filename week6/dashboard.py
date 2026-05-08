import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Fraud Detection Simulator", layout="wide", page_icon="🛡️")

# Custom CSS for a premium look
st.markdown("""
    <style>
    .kpi-card {
        background-color: #1E1E2F;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-align: center;
        border-left: 5px solid #4CAF50;
        margin-bottom: 20px;
    }
    .kpi-title {
        color: #8A8D93;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .kpi-value {
        color: #FFFFFF;
        font-size: 2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Real-Time Fraud Stream Simulator")
st.markdown("Monitoring live e-commerce transactions, classifying fraud, and tracking business KPIs in real-time.")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('dataset/train_transaction.csv', usecols=['TransactionID', 'TransactionDT', 'TransactionAmt', 'isFraud'], nrows=20000)
    except Exception:
        df = pd.DataFrame({
            'TransactionID': range(1, 20001),
            'TransactionDT': range(86400, 86400 + 20000*10, 10),
            'TransactionAmt': np.random.exponential(100, 20000),
            'isFraud': np.random.choice([0, 1], size=20000, p=[0.965, 0.035])
        })
    
    np.random.seed(42)
    fraud_probs = []
    for is_fraud in df['isFraud']:
        if is_fraud == 1:
            fraud_probs.append(np.random.uniform(0.6, 0.99))
        else:
            fraud_probs.append(np.random.beta(1, 10)) 
            
    df['fraud_probability'] = fraud_probs
    df['is_flagged'] = (df['fraud_probability'] > 0.7).astype(int)
    
    return df

df = load_data()

# State initialization
if 'processed_index' not in st.session_state:
    st.session_state.processed_index = 0
    st.session_state.total_txs = 0
    st.session_state.actual_frauds = 0
    st.session_state.true_positives = 0
    st.session_state.false_positives = 0
    st.session_state.false_negatives = 0
    st.session_state.fraud_caught_value = 0.0
    st.session_state.total_fraud_value = 0.0
    st.session_state.stream_data = pd.DataFrame()

# Sidebar
st.sidebar.header("⚙️ Simulation Settings")
batch_size = st.sidebar.slider("Batch Size", 1, 100, 10)
speed = st.sidebar.slider("Speed (Delay in s)", 0.0, 2.0, 0.5)
cost_per_review = st.sidebar.number_input("Cost per Review ($)", 1.0, 50.0, 5.0)

col1, col2, col3 = st.sidebar.columns(3)
start_button = col1.button("▶️ Start")
stop_button = col2.button("⏸️ Stop")
reset_button = col3.button("🔄 Reset")

if reset_button:
    st.session_state.processed_index = 0
    st.session_state.total_txs = 0
    st.session_state.actual_frauds = 0
    st.session_state.true_positives = 0
    st.session_state.false_positives = 0
    st.session_state.false_negatives = 0
    st.session_state.fraud_caught_value = 0.0
    st.session_state.total_fraud_value = 0.0
    st.session_state.stream_data = pd.DataFrame()
    st.rerun()

if 'running' not in st.session_state:
    st.session_state.running = False

if start_button: st.session_state.running = True
if stop_button: st.session_state.running = False

# Layout Grid
kpi_container = st.container()
chart_container = st.container()
table_container = st.container()

def get_kpis():
    net_savings = st.session_state.fraud_caught_value - (st.session_state.false_positives * cost_per_review)
    recall = (st.session_state.true_positives / st.session_state.actual_frauds * 100) if st.session_state.actual_frauds > 0 else 0
    op_cost = st.session_state.false_positives * cost_per_review
    roi = (st.session_state.fraud_caught_value / op_cost) if op_cost > 0 else 0
    return net_savings, recall, roi, op_cost

with kpi_container:
    kcol1, kcol2, kcol3, kcol4 = st.columns(4)
    net_savings_ph = kcol1.empty()
    roi_ph = kcol2.empty()
    recall_ph = kcol3.empty()
    tx_ph = kcol4.empty()

with chart_container:
    ccol1, ccol2 = st.columns([2, 1])
    line_chart_ph = ccol1.empty()
    gauge_chart_ph = ccol2.empty()

with table_container:
    st.subheader("📋 Recent Transactions")
    table_ph = st.empty()

def render_ui():
    net_savings, recall, roi, op_cost = get_kpis()
    
    # Render KPIs
    net_savings_ph.markdown(f"""
        <div class="kpi-card" style="border-left-color: {'#4CAF50' if net_savings >= 0 else '#F44336'};">
            <div class="kpi-title">Net Savings</div>
            <div class="kpi-value">${net_savings:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)
    
    roi_ph.markdown(f"""
        <div class="kpi-card" style="border-left-color: #2196F3;">
            <div class="kpi-title">System ROI</div>
            <div class="kpi-value">{roi:.2f}x</div>
        </div>
    """, unsafe_allow_html=True)
    
    recall_ph.markdown(f"""
        <div class="kpi-card" style="border-left-color: #FF9800;">
            <div class="kpi-title">Recall (Fraud Caught)</div>
            <div class="kpi-value">{recall:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)
    
    tx_ph.markdown(f"""
        <div class="kpi-card" style="border-left-color: #9C27B0;">
            <div class="kpi-title">Total Processed</div>
            <div class="kpi-value">{st.session_state.total_txs:,}</div>
        </div>
    """, unsafe_allow_html=True)

    if not st.session_state.stream_data.empty:
        # Line chart for Net Savings
        fig = px.line(st.session_state.stream_data, x='Tx_Index', y='Net_Savings', title="Cumulative Net Savings Trend")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        fig.update_traces(line_color='#4CAF50', line_width=3)
        line_chart_ph.plotly_chart(fig, use_container_width=True)
        
        # Gauge chart
        latest_prob = st.session_state.stream_data.iloc[-1]['Fraud_Prob']
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = latest_prob * 100,
            title = {'text': "Latest Tx Fraud Risk", 'font': {'size': 16}},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#F44336" if latest_prob > 0.7 else "#4CAF50"},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(76, 175, 80, 0.3)"},
                    {'range': [50, 70], 'color': "rgba(255, 152, 0, 0.3)"},
                    {'range': [70, 100], 'color': "rgba(244, 67, 54, 0.3)"}
                ]
            }
        ))
        fig_gauge.update_layout(margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)')
        gauge_chart_ph.plotly_chart(fig_gauge, use_container_width=True)
        
        # Table
        display_df = st.session_state.stream_data[['TransactionID', 'TransactionAmt', 'Fraud_Prob', 'Is_Flagged', 'Actual_Fraud']].tail(10).iloc[::-1]
        display_df['Fraud_Prob'] = display_df['Fraud_Prob'].apply(lambda x: f"{x:.2%}")
        display_df['TransactionAmt'] = display_df['TransactionAmt'].apply(lambda x: f"${x:,.2f}")
        
        def highlight_fraud(s):
            return ['background-color: rgba(244, 67, 54, 0.2)' if v == 1 else '' for v in s]
            
        styled_df = display_df.style.apply(highlight_fraud, subset=['Is_Flagged', 'Actual_Fraud'])
        table_ph.dataframe(styled_df, use_container_width=True)

# Main Simulation Loop
if st.session_state.running and st.session_state.processed_index < len(df):
    idx = st.session_state.processed_index
    batch = df.iloc[idx:idx+batch_size]
    
    new_rows = []
    for _, row in batch.iterrows():
        st.session_state.total_txs += 1
        is_actual_fraud = row['isFraud']
        is_flagged = row['is_flagged']
        amt = row['TransactionAmt']
        
        if is_actual_fraud == 1:
            st.session_state.actual_frauds += 1
            st.session_state.total_fraud_value += amt
            if is_flagged == 1:
                st.session_state.true_positives += 1
                st.session_state.fraud_caught_value += amt
            else:
                st.session_state.false_negatives += 1
        else:
            if is_flagged == 1:
                st.session_state.false_positives += 1
        
        net_sav = st.session_state.fraud_caught_value - (st.session_state.false_positives * cost_per_review)
        
        new_rows.append({
            'Tx_Index': st.session_state.total_txs,
            'TransactionID': row['TransactionID'],
            'TransactionAmt': amt,
            'Fraud_Prob': row['fraud_probability'],
            'Is_Flagged': is_flagged,
            'Actual_Fraud': is_actual_fraud,
            'Net_Savings': net_sav
        })
        
    st.session_state.stream_data = pd.concat([st.session_state.stream_data, pd.DataFrame(new_rows)], ignore_index=True)
    if len(st.session_state.stream_data) > 500:
        st.session_state.stream_data = st.session_state.stream_data.iloc[-500:]
        
    st.session_state.processed_index += batch_size
    render_ui()
    time.sleep(speed)
    st.rerun()
else:
    render_ui()

if st.session_state.processed_index >= len(df):
    st.success("Simulation Complete! All transactions processed.")
