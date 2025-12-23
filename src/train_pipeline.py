import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from mlxtend.frequent_patterns import apriori, association_rules
from datetime import datetime, timedelta
import joblib
import os

os.makedirs("models", exist_ok=True)

def train_all():
    print("⏳ Starting Offline Training...")
    
    try:
        df_trans = pd.read_csv("data/transactions.csv")
        df_prods = pd.read_csv("data/products.csv")
    except:
        print("❌ Error: Data not found. Run src/data_generator.py first.")
        return

    df_trans['date'] = pd.to_datetime(df_trans['date'])

    # --- 1. STOCK PREDICTION ---
    stock_forecasts = []
    for _, p in df_prods.iterrows():
        pid = p['id']
        sales = df_trans[df_trans['product_id'] == pid].copy()
        
        daily = sales.groupby('date')['quantity'].sum().reset_index()
        daily['cumulative'] = daily['quantity'].cumsum()
        start = daily['date'].min()
        
        if pd.isna(start): continue

        daily['day_idx'] = (daily['date'] - start).dt.days
        
        if len(daily) > 2:
            model = LinearRegression()
            model.fit(daily[['day_idx']], daily['cumulative'])
            burn_rate = model.coef_[0]
        else:
            burn_rate = 0.1

        if burn_rate <= 0: burn_rate = 0.01
        
        days_left = p['stock'] / burn_rate
        out_date = datetime.now() + timedelta(days=days_left)
        
        stock_forecasts.append({
            "product_id": pid,
            "product_name": p['name'],
            "current_stock": p['stock'],
            "burn_rate": round(burn_rate, 2),
            "days_left": int(days_left),
            "stockout_date": out_date.strftime("%Y-%m-%d"),
            "status": "CRITICAL" if days_left < 14 else "OK"
        })
    
    pd.DataFrame(stock_forecasts).to_csv("models/stock_forecast.csv", index=False)
    print("✅ Stock Models Trained.")

    # --- 2. RECOMMENDATION ---
    user_matrix = df_trans.pivot_table(index='customer_id', columns='product_id', values='rating').fillna(0)
    user_sim = cosine_similarity(user_matrix)
    joblib.dump(user_matrix, "models/user_item_matrix.pkl")
    joblib.dump(user_sim, "models/user_similarity.pkl")
    
    col_cat = 'cat' if 'cat' in df_prods.columns else 'category'
    df_prods['features'] = df_prods[col_cat] + " " + df_prods['name']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_prods['features'])
    content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    joblib.dump(content_sim, "models/content_similarity.pkl")
    print("✅ RecSys Models Saved.")

    # --- 3. MARKET BASKET (FIXED) ---
    print("⏳ Running Market Basket Analysis...")
    merged = df_trans.merge(df_prods, left_on='product_id', right_on='id')
    basket = (merged.groupby(['order_id', 'name'])['quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('order_id'))
    
    # Fix 1: Use .map() instead of .applymap()
    # Fix 2: Return Boolean (True/False) to satisfy mlxtend warning
    basket_sets = basket.map(lambda x: True if x >= 1 else False)
    
    frequent_sets = apriori(basket_sets, min_support=0.005, use_colnames=True)
    
    if frequent_sets.empty:
        print("⚠️ No patterns found.")
    else:
        rules = association_rules(frequent_sets, metric="lift", min_threshold=1)
        
        # --- CLEANING STEP ---
        rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x)[0])
        rules['consequents'] = rules['consequents'].apply(lambda x: list(x)[0])
        
        rules = rules.rename(columns={
            "antecedents": "If Buying This...",
            "consequents": "...Recommend This",
            "confidence": "Probability"
        })
        
        rules['Probability'] = (rules['Probability'] * 100).astype(int).astype(str) + "%"
        
        final_rules = rules[["If Buying This...", "...Recommend This", "Probability"]]
        final_rules = final_rules.sort_values(by="Probability", ascending=False).head(15)
        
        final_rules.to_csv("models/market_basket.csv", index=False)
        print("✅ Market Basket Rules Cleaned & Saved.")

    print("🚀 TRAINING COMPLETE.")

if __name__ == "__main__":
    train_all()