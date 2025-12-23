import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
import os

fake = Faker()
np.random.seed(42)
random.seed(42)

# Config
NUM_CUSTOMERS = 100
NUM_TRANSACTIONS = 4000
START_DATE = datetime.now() - timedelta(days=365)

# --- 1. PRODUCT CATALOG (With Business Profiles) ---
products = [
    # PROFILE: CRITICAL (High Demand, Low Stock -> Needs Alert)
    {"id": 101, "name": "Double-Sided Carpet Tape", "cat": "Installation", "profile": "critical", "price": 15},
    {"id": 102, "name": "Heavy Duty Entrance Mat", "cat": "Mat", "profile": "critical", "price": 45},
    
    # PROFILE: HEALTHY (Balanced Stock)
    {"id": 103, "name": "Modern Grey Shag Rug", "cat": "Rug", "profile": "healthy", "price": 200},
    {"id": 104, "name": "Non-Slip Rug Pad", "cat": "Installation", "profile": "healthy", "price": 50},
    {"id": 105, "name": "Kitchen Anti-Fatigue Mat", "cat": "Mat", "profile": "healthy", "price": 35},

    # PROFILE: SLOW/LUXURY (Low Sales, Low Stock is OK)
    {"id": 106, "name": "Royal Persian Silk Rug", "cat": "Luxury", "profile": "slow", "price": 1500},
    {"id": 107, "name": "Hand-Woven Wool Carpet", "cat": "Luxury", "profile": "slow", "price": 600},
    
    # PROFILE: OVERSTOCK (Low Sales, Huge Stock -> Bad Investment)
    {"id": 108, "name": "Artificial Grass Rug", "cat": "Outdoor", "profile": "overstock", "price": 90},
]

# Apply Real-World Logic
final_products = []
for p in products:
    profile = p.pop("profile")
    
    if profile == "critical":
        p['stock'] = random.randint(5, 20)      # Low Stock
        p['velocity_factor'] = 8.0              # High Sales
    elif profile == "healthy":
        p['stock'] = random.randint(100, 150)   # Medium Stock
        p['velocity_factor'] = 2.0              # Medium Sales
    elif profile == "slow":
        p['stock'] = random.randint(3, 8)       # Low Stock
        p['velocity_factor'] = 0.2              # Rare Sales
    elif profile == "overstock":
        p['stock'] = 250                        # Huge Stock
        p['velocity_factor'] = 0.5              # Low Sales
        
    final_products.append(p)

# --- 2. CUSTOMERS ---
customers = [{"id": i, "name": fake.name()} for i in range(1, NUM_CUSTOMERS + 1)]

# --- 3. TRANSACTIONS (Enforcing Velocity & Patterns) ---
transactions = []
order_id = 5000

for day in range(365):
    curr_date = START_DATE + timedelta(days=day)
    
    # Daily Sales Loop
    for p in final_products:
        # Poisson distribution creates realistic daily variations
        num_sales = np.random.poisson(p['velocity_factor'])

        if num_sales > 0:
            for _ in range(num_sales):
                order_id += 1
                cust_id = random.randint(1, NUM_CUSTOMERS)
                
                # Main Purchase
                transactions.append({
                    "date": curr_date.strftime("%Y-%m-%d"),
                    "order_id": order_id,
                    "customer_id": cust_id,
                    "product_id": p['id'],
                    "quantity": 1,
                    "rating": random.randint(3, 5)
                })
                
                # --- MARKET BASKET INJECTIONS (Hidden Patterns) ---
                # 1. Mat -> Tape (80%)
                if p['id'] == 102 and random.random() < 0.8:
                     transactions.append({"date": curr_date.strftime("%Y-%m-%d"), "order_id": order_id, "customer_id": cust_id, "product_id": 101, "quantity": 1, "rating": 4})

                # 2. Luxury Rug -> Pad (60%)
                if p['id'] in [106, 107] and random.random() < 0.6:
                     transactions.append({"date": curr_date.strftime("%Y-%m-%d"), "order_id": order_id, "customer_id": cust_id, "product_id": 104, "quantity": 1, "rating": 5})

# --- SAVE ---
os.makedirs("data", exist_ok=True)
pd.DataFrame(final_products).to_csv("data/products.csv", index=False)
pd.DataFrame(customers).to_csv("data/customers.csv", index=False)
pd.DataFrame(transactions).to_csv("data/transactions.csv", index=False)

print(f"✅ Data Generated: {len(transactions)} realistic transactions created.")