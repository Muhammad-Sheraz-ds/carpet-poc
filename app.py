import streamlit as st
import pandas as pd
import sys
import os

# Add 'src' to python path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the logic directly from your api file
# (This runs the model loading code inside api.py automatically)
from src import api

st.set_page_config(page_title="Carpet Analysis AI", layout="wide")
st.title("Carpet Analysis AI")
st.markdown("**Enterprise Stock & Recommendation Engine**")

# Load Resources Check
try:
    # api.df_prods should be loaded when we imported api
    if api.df_prods is None:
        st.error("Models not loaded. Ensure 'models/' folder is uploaded to GitHub.")
        st.stop()
        
    prods = api.get_products()
    prod_map = {p['name']: p['id'] for p in prods}
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["Stock Intelligence", "Recommendation System", "Market Basket Analysis"])

# --- TAB 1: STOCK ---
with tab1:
    st.subheader("Inventory Forecast")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        s_name = st.selectbox("Select Product", list(prod_map.keys()))
        if st.button("Analyze Burn Rate"):
            pid = prod_map[s_name]
            
            # DIRECT FUNCTION CALL
            res = api.predict_stock(pid)
            
            st.metric("Current Stock", res['current_stock'])
            st.metric("Burn Rate", f"{res['burn_rate']} units/day")
            
            if res['status'] == "CRITICAL":
                st.error(f"CRITICAL: Stockout in {res['days_left']} days ({res['stockout_date']})")
            else:
                st.success(f"HEALTHY: Stockout in {res['days_left']} days ({res['stockout_date']})")

# --- TAB 2: RECS ---
with tab2:
    st.subheader("Personalized Suggestions")
    
    with st.container():
        st.write("Simulation Controls")
        c1, c2 = st.columns(2)
        with c1:
            uid = st.number_input("Customer ID", 1, 100, 1)
        with c2:
            context_prod = st.selectbox(
                "Simulate Page View (Current Context)", 
                ["Home Page (No Context)"] + list(prod_map.keys())
            )

    if st.button("Get Recommendations", type="primary"):
        # DIRECT FUNCTION CALL
        viewing_id = prod_map[context_prod] if context_prod != "Home Page (No Context)" else None
        
        # We call the python function directly
        res = api.hybrid_recommend(user_id=uid, viewing_product_id=viewing_id)
        
        if res.get('recommendations'):
            st.success("Top Picks For This User:")
            for item in res['recommendations']:
                st.markdown(f"- {item}")
            st.divider()
            with st.expander("View AI Reasoning"):
                for reason in res.get('logic_explanation', []):
                    st.info(f"• {reason}")
        else:
            st.warning("No recommendations found.")

# --- TAB 3: MARKET BASKET ---
with tab3:
    st.subheader("Market Basket Analysis")
    st.markdown("Discover hidden buying patterns (e.g., Door Mat -> Tape).")
    
    if st.button("Run Apriori Analysis"):
        # DIRECT FUNCTION CALL
        data = api.get_patterns()
        df = pd.DataFrame(data)
        st.table(df)