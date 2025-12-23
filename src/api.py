from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib
import os

app = FastAPI(title="CarpetAI Engine")

# --- LOAD RESOURCES ---
try:
    df_stock = pd.read_csv("models/stock_forecast.csv")
    df_rules = pd.read_csv("models/market_basket.csv")
    df_prods = pd.read_csv("data/products.csv")
    
    # Load Matrices
    user_matrix = joblib.load("models/user_item_matrix.pkl")
    user_sim = joblib.load("models/user_similarity.pkl")
    content_sim = joblib.load("models/content_similarity.pkl")
except:
    print("⚠️ WARNING: Models missing. Run train_pipeline.py first!")

# Helper to look up product names
def get_names(id_list):
    return df_prods[df_prods['id'].isin(id_list)]['name'].tolist()

@app.get("/")
def home(): return {"status": "AI Online"}

@app.get("/products")
def get_products():
    return df_prods[['id', 'name']].to_dict(orient="records")

# 1. STOCK PREDICTION
@app.get("/predict/stock/{product_id}")
def predict_stock(product_id: int):
    row = df_stock[df_stock['product_id'] == product_id]
    if row.empty: raise HTTPException(404, "Product not found")
    return row.to_dict(orient="records")[0]

# 2. MARKET BASKET
@app.get("/market-basket")
def get_patterns():
    return df_rules.to_dict(orient="records")

# 3. HYBRID RECOMMENDATION (THE NEW LOGIC)
@app.get("/recommend/hybrid")
def hybrid_recommend(user_id: int, viewing_product_id: Optional[int] = None):
    """
    Combines User-History (Collaborative) AND Item-Context (Content-Based)
    into a single weighted list.
    """
    final_recommendations = set()
    debug_info = []

    # --- STRATEGY A: COLLABORATIVE (Who are you?) ---
    # Logic: Find a similar user and see what they bought.
    if user_id in user_matrix.index:
        u_idx = user_matrix.index.get_loc(user_id)
        # Find most similar user (excluding self)
        sim_scores = list(enumerate(user_sim[u_idx]))
        sim_idx = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1][0]
        similar_user_id = user_matrix.index[sim_idx]
        
        # Get items they liked (>3 stars) that YOU haven't bought
        their_ratings = user_matrix.iloc[sim_idx]
        my_ratings = user_matrix.iloc[u_idx]
        collab_ids = their_ratings[(their_ratings > 3) & (my_ratings == 0)].index.tolist()
        
        # Add Top 2 Collaborative items
        for pid in collab_ids[:2]:
            final_recommendations.add(pid)
            
        debug_info.append(f"Because you are similar to Customer #{similar_user_id}")

    # --- STRATEGY B: CONTENT-BASED (What do you like?) ---
    # Logic: Look at 'viewing_product_id'. If None, look at user's last purchase.
    target_pid = viewing_product_id
    
    # If no product selected, try to find one from user history
    if not target_pid and user_id in user_matrix.index:
        u_idx = user_matrix.index.get_loc(user_id)
        my_ratings = user_matrix.iloc[u_idx]
        # Get user's highest rated product
        liked_products = my_ratings[my_ratings > 3].index.tolist()
        if liked_products:
            target_pid = liked_products[0] # Pick the first liked item
            
    if target_pid and target_pid in df_prods['id'].values:
        idx = df_prods[df_prods['id'] == target_pid].index[0]
        # Get top 3 similar items
        scores = sorted(list(enumerate(content_sim[idx])), key=lambda x: x[1], reverse=True)[1:4]
        content_ids = [df_prods.iloc[i[0]]['id'] for i in scores]
        
        # Add Top 2 Content items
        for pid in content_ids[:2]:
            final_recommendations.add(pid)
            
        product_name = df_prods[df_prods['id'] == target_pid].iloc[0]['name']
        debug_info.append(f"Because you are interested in '{product_name}'")

    # --- FINAL MERGE ---
    rec_list = list(final_recommendations)
    if not rec_list:
        return {"recs": [], "reason": "New User & No Context"}
        
    names = get_names(rec_list)
    return {
        "user_id": user_id,
        "recommendations": names,
        "logic_explanation": debug_info
    }