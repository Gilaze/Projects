import streamlit as st
import pandas as pd
import plotly.express as px
import os
import kalshi_python
from kalshi_python.models import LoginRequest

# --- 1. SETUP & AUTHENTICATION ---
st.set_page_config(page_title="My Kalshi Dashboard", layout="wide")
st.title("📈 My Kalshi Portfolio")

# Ideally, pull these from st.secrets or os.environ in Leapcell
KEY_ID = os.environ.get("KALSHI_KEY_ID", "YOUR_KEY_ID")
PRIVATE_KEY = os.environ.get("KALSHI_PRIVATE_KEY", "YOUR_PRIVATE_KEY_STRING")

@st.cache_resource
def get_kalshi_client():
    config = kalshi_python.Configuration()
    # Use demo-api for testing, trading-api for live
    config.host = "https://trading-api.kalshi.co/trade-api/v2" 
    
    kalshi_api = kalshi_python.ApiInstance(
        kalshi_python.ApiClient(config)
    )
    # Note: Refer to Kalshi docs for exact RSA auth implementation
    # as you must sign a timestamp with your private key to log in.
    return kalshi_api

client = get_kalshi_client()

# --- 2. FETCH DATA ---
# (Mocking the data logic so you can see how the UI wires up)

def fetch_data():
    # In a real app, you would call:
    # balance_res = client.get_portfolio_balance()
    # history_res = client.get_portfolio_history()
    # orders_res = client.get_orders(status='resting')
    
    # Simulating the processed data for the dashboard based on your example
    history = pd.DataFrame([
        {"date": "2024-01-01", "type": "deposit", "amount": 400.00},
        {"date": "2024-02-15", "type": "withdrawal", "amount": -300.00}
    ])
    
    current_balance = 200.00 
    
    active_bids = pd.DataFrame([
        {"ticker": "INFLATION-24", "side": "yes", "price": 0.45, "contracts": 10},
        {"ticker": "FED-RATE-NOV", "side": "no", "price": 0.20, "contracts": 50}
    ])
    
    return history, current_balance, active_bids

history_df, current_balance, bids_df = fetch_data()

# --- 3. CALCULATE CAREER PROFIT ---
total_deposits = history_df[history_df['type'] == 'deposit']['amount'].sum()
total_withdrawals = abs(history_df[history_df['type'] == 'withdrawal']['amount'].sum())

# Formula: What I have + What I took out - What I put in
career_profit = current_balance + total_withdrawals - total_deposits

st.metric(label="Career Profit", value=f"${career_profit:.2f}")

# --- 4. BUILD THE UI TABS ---
tab1, tab2, tab3 = st.tabs(["📋 Transfer Log", "📊 Profit Graph", "🛒 Current Bids"])

with tab1:
    st.subheader("Deposit & Withdrawal History")
    st.dataframe(history_df, use_container_width=True)

with tab2:
    st.subheader("In vs. Out Over Time")
    # Create cumulative sums for the graph
    history_df['cumulative_in'] = history_df[history_df['amount'] > 0]['amount'].cumsum()
    history_df['cumulative_out'] = history_df[history_df['amount'] < 0]['amount'].abs().cumsum()
    
    # Forward fill missing values for a smooth line chart
    history_df = history_df.fillna(method='ffill').fillna(0)
    
    fig = px.line(history_df, x='date', y=['cumulative_in', 'cumulative_out'], 
                  labels={'value': 'USD', 'variable': 'Flow Type'},
                  title="Cumulative Money Put In vs. Taken Out")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Active Resting Orders")
    if bids_df.empty:
        st.info("You have no active bids at the moment.")
    else:
        st.dataframe(bids_df, use_container_width=True)