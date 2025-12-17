"""
Regression Predictor Page

Predicts TotalAmount for Amazon sales orders.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from amazon_sales_ml.ui.api_client import PredictorClient


# Feature options (based on training data)
CATEGORIES = ["Books", "Clothing", "Electronics", "Home & Kitchen", "Sports & Outdoors", "Toys & Games"]
BRANDS = ["Apex", "BrightLux", "HomeEase", "KiddoFun", "ReadMore", "UrbanStyle", "Zenith"]
PAYMENT_METHODS = ["Amazon Pay", "Cash on Delivery", "Credit Card", "Debit Card", "Net Banking", "UPI"]
ORDER_STATUSES = ["Cancelled", "Delivered", "Pending", "Returned", "Shipped"]
COUNTRIES = ["Australia", "Canada", "India", "United States"]
DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def render_model_selector(client: PredictorClient):
    """Render model selection and status in sidebar."""
    st.sidebar.markdown("### Model Selection")
    
    if not client.health_check():
        st.sidebar.error("API Disconnected")
        st.sidebar.caption("Start the API with:")
        st.sidebar.code("uv run python scripts/run_api.py", language="bash")
        return
    
    st.sidebar.success("API Connected")
    
    # Get available models
    available = client.get_available_models()
    current_info = client.get_model_info()
    
    if available and available.registered_models:
        # Build selection options
        options = []
        option_labels = {}
        
        for m in available.registered_models:
            options.append(m.alias)
            rmse = m.metrics.get("test_RMSE", m.metrics.get("test_rmse", 0))
            r2 = m.metrics.get("test_R2", m.metrics.get("test_r2", 0))
            label = f"{m.alias.upper()} (v{m.version})"
            option_labels[m.alias] = label
        
        if available.fallback_available:
            options.append("fallback")
            option_labels["fallback"] = "FALLBACK (best_model.yaml)"
        
        # Current selection
        current_alias = current_info.alias if current_info and current_info.alias else "fallback"
        if current_alias not in options:
            current_alias = options[0] if options else None
        
        selected = st.sidebar.selectbox(
            "Select Model",
            options=options,
            index=options.index(current_alias) if current_alias in options else 0,
            format_func=lambda x: option_labels.get(x, x),
            key="model_selector",
        )
        
        # Load button
        if st.sidebar.button("Load Selected Model", use_container_width=True):
            with st.sidebar:
                with st.spinner("Loading model..."):
                    success, message = client.load_model(selected)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(f"Failed: {message}")
    
    elif available and available.fallback_available:
        st.sidebar.info("No registered models found. Using fallback.")
    else:
        st.sidebar.warning("No models available")
    
    # Show current model details
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Current Model")
    
    if current_info:
        if current_info.alias:
            st.sidebar.markdown(f"**Type:** `{current_info.alias.upper()}`")
        else:
            st.sidebar.markdown("**Type:** `FALLBACK`")
        
        if current_info.version:
            st.sidebar.markdown(f"**Version:** `{current_info.version}`")
        
        if current_info.run_id:
            st.sidebar.markdown(f"**Run ID:** `{current_info.run_id[:8]}...`")
        
        # Metrics
        if current_info.metrics:
            st.sidebar.markdown("**Metrics:**")
            for key, value in current_info.metrics.items():
                display_key = key.replace("test_", "").replace("_", " ")
                if isinstance(value, float):
                    st.sidebar.markdown(f"- {display_key}: `{value:.4f}`")
                else:
                    st.sidebar.markdown(f"- {display_key}: `{value}`")
    else:
        st.sidebar.warning("Could not load model info")


def get_input_features() -> Dict[str, Any]:
    """Render input form and return features."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Product Details")
        category = st.selectbox("Category", CATEGORIES, index=2)
        brand = st.selectbox("Brand", BRANDS, index=6)
        quantity = st.number_input("Quantity", min_value=1, max_value=100, value=2)
        unit_price = st.number_input("Unit Price ($)", min_value=0.01, max_value=10000.0, value=299.99, format="%.2f")
    
    with col2:
        st.markdown("##### Pricing")
        discount = st.slider("Discount", min_value=0.0, max_value=0.5, value=0.1, step=0.05, format="%.2f")
        tax = st.number_input("Tax ($)", min_value=0.0, max_value=1000.0, value=48.0, format="%.2f")
        shipping_cost = st.number_input("Shipping Cost ($)", min_value=0.0, max_value=100.0, value=5.99, format="%.2f")
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("##### Order Info")
        payment_method = st.selectbox("Payment Method", PAYMENT_METHODS, index=2)
        order_status = st.selectbox("Order Status", ORDER_STATUSES, index=1)
    
    with col4:
        st.markdown("##### Location")
        country = st.selectbox("Country", COUNTRIES, index=3)
        city = st.text_input("City", value="New York")
        state = st.text_input("State", value="NY")
    
    st.markdown("---")
    
    st.markdown("##### Order Date")
    col5, col6, col7 = st.columns(3)
    
    with col5:
        order_year = st.selectbox("Year", [2020, 2021, 2022, 2023, 2024], index=4)
    with col6:
        order_month = st.selectbox("Month", list(range(1, 13)), index=5)
    with col7:
        order_day = st.selectbox("Day of Week", DAYS_OF_WEEK, index=0)
    
    return {
        "Category": category,
        "Brand": brand,
        "Quantity": quantity,
        "UnitPrice": unit_price,
        "Discount": discount,
        "Tax": tax,
        "ShippingCost": shipping_cost,
        "PaymentMethod": payment_method,
        "OrderStatus": order_status,
        "City": city,
        "State": state,
        "Country": country,
        "OrderYear": order_year,
        "OrderMonth": order_month,
        "OrderDayOfWeek": order_day,
    }


def render_prediction_result(prediction: float, features: Dict[str, Any]):
    """Render prediction result."""
    st.markdown("### Prediction Result")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Predicted Total",
            value=f"${prediction:,.2f}",
        )
    
    with col2:
        base = features["Quantity"] * features["UnitPrice"]
        st.metric(
            label="Base Amount",
            value=f"${base:,.2f}",
            help="Quantity × Unit Price"
        )
    
    with col3:
        discount_amount = base * features["Discount"]
        st.metric(
            label="Discount Applied",
            value=f"-${discount_amount:,.2f}",
            help=f"{features['Discount']*100:.0f}% discount"
        )


def render_batch_upload(client: PredictorClient):
    """Render batch prediction via CSV upload."""
    st.markdown("### Batch Prediction")
    st.caption("Upload a CSV file with the same columns as the input form above.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="batch_csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.markdown(f"**Loaded {len(df)} rows**")
            
            with st.expander("Preview Data", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("Run Batch Prediction", type="primary", key="batch_predict"):
                with st.spinner("Making predictions..."):
                    records = df.to_dict(orient="records")
                    result = client.predict_batch(records)
                    
                    if result.success and result.predictions:
                        df["PredictedTotalAmount"] = result.predictions
                        
                        st.success(f"Predicted {len(result.predictions)} records!")
                        st.dataframe(df, use_container_width=True)
                        
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv",
                        )
                    else:
                        st.error(f"Prediction failed: {result.error}")
        except Exception as e:
            st.error(f"Error reading file: {e}")


def render(client: PredictorClient):
    """Main render function for the regression page."""
    
    st.title("Sales Amount Predictor")
    st.caption("Predict the total amount for Amazon sales orders using machine learning.")
    
    # Sidebar: Model selection and status
    render_model_selector(client)
    
    # Tabs for single vs batch prediction
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    
    with tab1:
        st.markdown("### Input Features")
        
        features = get_input_features()
        
        st.markdown("---")
        
        if st.button("Predict Total Amount", type="primary", use_container_width=True):
            if not client.health_check():
                st.error("API is not available. Please start the prediction server.")
            else:
                with st.spinner("Making prediction..."):
                    result = client.predict(features)
                    
                    if result.success and result.prediction is not None:
                        render_prediction_result(result.prediction, features)
                    else:
                        st.error(f"Prediction failed: {result.error}")
    
    with tab2:
        render_batch_upload(client)
