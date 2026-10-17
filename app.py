import streamlit as st
import pandas as pd

st.set_page_config(page_title="Carefull Fraud Detection", layout="wide")
st.title("🛡️ Carefull - Transaction Alert System Results")

st.markdown("### LLM Validation Results (Groq gpt-oss-20b & OpenRouter GPT-4o-mini)")
try:
    results = pd.read_csv("results/results_groq.csv")
    st.dataframe(results[['transaction_id', 'user_id', 'risk_level', 'confidence', 'reasoning']], use_container_width=True)
    
    st.markdown("### Filtered High-Risk Transactions")
    high_risk = results[results['risk_level'] == 'high']
    st.dataframe(high_risk, use_container_width=True)
except FileNotFoundError:
    st.error("Results file not found. Please run the pipeline first.")