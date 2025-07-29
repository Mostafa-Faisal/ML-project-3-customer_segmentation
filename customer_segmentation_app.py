import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models
@st.cache_resource
def load_models():
    try:
        kmeans_model = joblib.load('models/kmeans_customer_segmentation.pkl')
        scaler = joblib.load('models/scaler.pkl')
        pca_model = joblib.load('models/pca_model.pkl')
        return kmeans_model, scaler, pca_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Prediction function
def predict_customer_segment(customer_data, kmeans_model, scaler):
    features = ['Age', 'Annual_Income', 'Spending_Score', 
                'Purchase_Frequency', 'Average_Order_Value', 'Years_Customer']
    
    # Create DataFrame
    df = pd.DataFrame([customer_data])
    
    # Select and scale features
    X = df[features]
    X_scaled = scaler.transform(X)
    
    # Predict cluster
    cluster = kmeans_model.predict(X_scaled)[0]
    
    # Get cluster probabilities (distances to centroids)
    distances = kmeans_model.transform(X_scaled)[0]
    probabilities = 1 / (1 + distances)
    probabilities = probabilities / probabilities.sum()
    
    return cluster, probabilities

# Cluster interpretation
def interpret_cluster(cluster_id):
    cluster_profiles = {
        0: "Budget-conscious customers with moderate spending patterns",
        1: "High-value customers with strong purchasing power", 
        2: "Frequent buyers with consistent engagement",
        3: "Premium customers with high order values",
        4: "New or occasional customers with growth potential"
    }
    return cluster_profiles.get(cluster_id, f"Customer segment {cluster_id}")

# Main app
def main():
    st.title("🎯 Customer Segmentation Prediction")
    st.markdown("Predict customer segments using K-Means clustering")
    
    # Load models
    kmeans_model, scaler, pca_model = load_models()
    
    if kmeans_model is None:
        st.error("Models not found. Please ensure model files are in the 'models' directory.")
        return
    
    # Sidebar for input
    st.sidebar.header("Customer Information")
    
    # Input fields
    age = st.sidebar.slider("Age", 18, 80, 35)
    annual_income = st.sidebar.slider("Annual Income ($)", 20000, 150000, 60000)
    spending_score = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)
    purchase_frequency = st.sidebar.slider("Purchase Frequency", 1, 50, 15)
    avg_order_value = st.sidebar.slider("Average Order Value ($)", 20, 500, 150)
    years_customer = st.sidebar.slider("Years as Customer", 0.1, 10.0, 2.0)
    
    # Create customer data
    customer_data = {
        'Age': age,
        'Annual_Income': annual_income,
        'Spending_Score': spending_score,
        'Purchase_Frequency': purchase_frequency,
        'Average_Order_Value': avg_order_value,
        'Years_Customer': years_customer
    }
    
    # Predict button
    if st.sidebar.button("Predict Segment", type="primary"):
        cluster, probabilities = predict_customer_segment(customer_data, kmeans_model, scaler)
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Prediction Results")
            st.success(f"**Predicted Cluster: {cluster}**")
            st.info(f"**Profile:** {interpret_cluster(cluster)}")
            
            # Customer summary
            st.subheader("Customer Summary")
            summary_df = pd.DataFrame([customer_data]).T
            summary_df.columns = ['Value']
            st.dataframe(summary_df, use_container_width=True)
            
        with col2:
            st.subheader("Cluster Probabilities")
            prob_df = pd.DataFrame({
                'Cluster': [f'Cluster {i}' for i in range(len(probabilities))],
                'Probability': probabilities
            })
            
            fig = px.bar(prob_df, x='Cluster', y='Probability', 
                        title='Cluster Assignment Confidence')
            st.plotly_chart(fig, use_container_width=True)
    
    # Batch prediction
    st.subheader("📊 Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Uploaded data:", batch_df.head())
            
            if st.button("Predict All"):
                # Predict for all customers
                predictions = []
                for _, row in batch_df.iterrows():
                    cluster, _ = predict_customer_segment(row.to_dict(), kmeans_model, scaler)
                    predictions.append(cluster)
                
                batch_df['Predicted_Cluster'] = predictions
                batch_df['Cluster_Profile'] = batch_df['Predicted_Cluster'].apply(interpret_cluster)
                
                st.success("Predictions completed!")
                st.dataframe(batch_df)
                
                # Download results
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="customer_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
