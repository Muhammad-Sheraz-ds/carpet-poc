
# 🧶 Carpet Business AI: Intelligent Retail POC

## 📌 Project Overview
This project is a Proof of Concept (POC). It utilizes Machine Learning to solve three critical retail problems:
1.  **Stock Prediction:** forecasting when inventory will run out based on sales velocity.
2.  **Market Basket Analysis:** Discovering hidden cross-selling patterns (e.g., "Customers who buy Stair Treads also buy Carpet Tape").
3.  **Recommendation System:** Suggesting products based on user history (Collaborative) and item features (Content-Based).

---

## 🏗️ System Architecture (How it Works)
This project follows a professional **Production Pipeline** approach, separating "Heavy Training" from "Fast Inference."

1.  **Data Layer (`data/`)**: Synthetic data is generated with specific business rules injected.
2.  **Training Pipeline (`src/train_pipeline.py`)**: Runs offline. It loads data, trains ML models (Regression, Apriori, Cosine Similarity), and saves the results/models to the `models/` or `data/processed/` folder.
3.  **Inference Layer (`app.py`)**: The Gradio UI. It simply loads the pre-trained models/results to provide instant answers to the user.

---

## 📂 The Dataset & Product Logic
We generate a synthetic dataset focused strictly on the **Flooring Ecosystem**. The data is not random; it contains injected "buying behaviors" to test the algorithms.

### 1. Product Catalog (12 Items)
| Category | Product Name | Logic / Relation |
| :--- | :--- | :--- |
| **Luxury Rugs** | Royal Persian Silk Rug | High value, low stock. |
| | Hand-Woven Wool Carpet | Often bought with **Rug Pads**. |
| **Everyday** | Modern Grey Shag Rug | High volume seller. |
| | Bohemian Jute Runner | - |
| **Specialty Mats** | Heavy Duty Entrance Mat | Often bought with **Grippers**. |
| | Non-Slip Stair Treads | **Hard Rule:** Requires **Tape**. |
| | Kitchen Anti-Fatigue Mat | - |
| | Artificial Grass Rug | - |
| **Installation** | Thick Felt Rug Pad (Underlay) | The "Upsell" for luxury rugs. |
| | Double-Sided Carpet Tape | Essential for Stair Treads. |
| | Anti-Curl Corner Grippers | Essential for Door Mats. |

### 2. Injected Patterns (The "Eggs & Bread" of Carpets)
The data generator creates **3,000 transactions** containing these specific hidden rules for the AI to find:
* **The Staircase Project:** If a user buys *Stair Treads*, there is a 90% probability they also buy *Carpet Tape*.
* **The Luxury Protection:** If a user buys a *Silk/Wool Rug*, there is a 70% probability they buy a *Rug Pad*.
* **The Entryway Setup:** If a user buys an *Entrance Mat*, there is a 60% probability they buy *Corner Grippers*.

---

## 🚀 Setup & Installation

### 1. Environment Setup
```bash
# Create Virtual Environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install Dependencies
pip install -r requirements.txt

```

### 2. Execution Order (Crucial)

**Step 1: Generate Data**
Creates `products.csv`, `customers.csv`, and `transactions.csv`.

```bash
python src/data_generator.py

```

**Step 2: Train Models (The Batch Process)**
Runs Linear Regression and Association Rule Mining. Saves `.pkl` and processed `.csv` files.

```bash
python src/train_pipeline.py

```

**Step 3: Run the Application**
Launches the Gradio Web Interface.

```bash
python app.py

```