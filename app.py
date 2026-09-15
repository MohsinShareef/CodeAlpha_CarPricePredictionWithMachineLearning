"""
Streamlit Car Price Predictor
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------- Page config ----------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)

# ---------- Load artifacts (cached) ----------
@st.cache_resource
def load_artifacts():
    model          = joblib.load("model.pkl")
    brand_goodwill = joblib.load("brand_goodwill.pkl")
    global_mean    = joblib.load("global_mean.pkl")
    return model, brand_goodwill, global_mean

model, brand_goodwill, global_mean = load_artifacts()

# ---------- Reference data (for dropdowns) ----------
# Sorted lists taken from the training data
BRANDS = sorted(brand_goodwill.keys())
FUEL_TYPES   = ["Petrol", "Diesel", "CNG"]
SELLING_TYPES = ["Dealer", "Individual"]
TRANSMISSIONS = ["Manual", "Automatic"]

# ---------- UI ----------
st.title("🚗 Car Price Predictor")
st.markdown(
    "Enter the car details below and get an **estimated selling price** "
    "(in lakhs INR) powered by a Random Forest model."
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", BRANDS)
        present_price = st.number_input(
            "Present Showroom Price (lakhs)",
            min_value=0.0, max_value=100.0, value=6.0, step=0.1,
        )
        driven_kms = st.number_input(
            "Kilometers Driven",
            min_value=0, max_value=500_000, value=40_000, step=500,
        )
        fuel_type = st.selectbox("Fuel Type", FUEL_TYPES)

    with col2:
        car_age = st.slider("Car Age (years)", 0, 30, 5)
        transmission = st.selectbox("Transmission", TRANSMISSIONS)
        selling_type = st.selectbox("Seller Type", SELLING_TYPES)
        owner = st.selectbox("Previous Owners", [0, 1, 2, 3], index=0)

    submitted = st.form_submit_button("💰 Predict Price")

# ---------- Prediction ----------
if submitted:
    # Build single-row input matching training schema
    goodwill = brand_goodwill.get(brand, global_mean)

    input_df = pd.DataFrame([{
        "Present_Price":   present_price,
        "Driven_kms":      driven_kms,
        "Owner":           owner,
        "Car_Age":         car_age,
        "Brand_Goodwill":  goodwill,
        "Fuel_Type":       fuel_type,
        "Selling_type":    selling_type,
        "Transmission":    transmission,
        "Brand":           brand,
    }])

    pred = float(model.predict(input_df)[0])
    pred = max(pred, 0)  # no negative prices

    st.success(f"### Estimated Selling Price: ₹ {pred:,.2f} lakhs")
    st.caption(
        f"≈ ₹ {pred * 100_000:,.0f} INR  ·  "
        f"Brand goodwill factor: {goodwill:.2f}"
    )

    # Confidence band from individual trees (Random Forest only)
    try:
        prep   = model.named_steps["prep"]
        rf     = model.named_steps["model"]
        X_prep = prep.transform(input_df)
        tree_preds = np.array([t.predict(X_prep)[0] for t in rf.estimators_])
        st.info(
            f"Tree spread: ₹{tree_preds.min():.2f}L – ₹{tree_preds.max():.2f}L "
            f"(σ = {tree_preds.std():.2f})"
        )
    except Exception:
        pass  # Skip if best model isn't Random Forest

    st.divider()
    st.markdown("#### 📋 Input Summary")
    st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        "This app predicts **used-car selling prices** using a "
        "Random Forest regressor trained on the Kaggle "
        "*Car Price Prediction (Used Cars)* dataset.\n\n"
        "**Features used:**\n"
        "- Brand goodwill (avg. brand price)\n"
        "- Car age\n"
        "- Present showroom price\n"
        "- Kilometers driven\n"
        "- Fuel type, transmission, seller type, owners\n\n"
        "Prices are in **lakhs INR** (1 lakh = 100,000)."
    )