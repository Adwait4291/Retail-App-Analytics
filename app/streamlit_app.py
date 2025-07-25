# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime
import json
from pymongo import MongoClient
from dotenv import load_dotenv

# Set page configuration
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# MongoDB connection
def get_mongodb_connection():
    """Create a new MongoDB connection"""
    username = os.getenv("MONGODB_USERNAME")
    password = os.getenv("MONGODB_PASSWORD")
    cluster = os.getenv("MONGODB_CLUSTER")
    database = os.getenv("MONGODB_DATABASE")
    
    connection_string = f"mongodb+srv://{username}:{password}@{cluster}/"
    client = MongoClient(connection_string)
    db = client.get_database(database)
    
    return db, client

# Data retrieval functions
def fetch_from_mongodb(limit=1000):
    """Fetch raw data from MongoDB"""
    db, client = get_mongodb_connection()
    try:
        collection = db.products
        cursor = collection.find({}, {'_id': 0}, limit=limit)
        df = pd.DataFrame(list(cursor))
        return df
    finally:
        client.close()

def load_processed_data(latest=True, processing_version=None, limit=1000):
    """Load processed data from MongoDB"""
    db, client = get_mongodb_connection()
    try:
        collection = db.processed_retail_data
        
        # Query based on parameters
        if processing_version:
            query = {"processing_version": processing_version}
        elif latest:
            metadata_collection = db.processing_metadata
            latest_metadata = metadata_collection.find_one({"domain": "retail"}, sort=[("processed_at", -1)])
            
            if not latest_metadata:
                return pd.DataFrame()
                
            latest_version = latest_metadata["processing_version"]
            query = {"processing_version": latest_version}
        else:
            query = {}
        
        # Fetch data with limit
        cursor = collection.find(query, {'_id': 0}, limit=limit)
        df = pd.DataFrame(list(cursor))
        return df
    finally:
        client.close()

# Get data stats
def get_data_stats():
    """Get statistics about the MongoDB data"""
    db, client = get_mongodb_connection()
    try:
        # Product collection counts
        raw_count = db.products.count_documents({})
        
        # Processed data collection counts
        processed_count = db.processed_retail_data.count_documents({})
        
        # Processing metadata
        metadata = list(db.processing_metadata.find(
            {}, 
            sort=[("processed_at", -1)],
            limit=5
        ))
        
        return {
            "raw_count": raw_count,
            "processed_count": processed_count,
            "recent_processing": metadata
        }
    finally:
        client.close()

# Get latest model info
def get_latest_model_info():
    """Get latest model metadata"""
    db, client = get_mongodb_connection()
    try:
        model_metadata = db.model_metadata.find_one(
            {"status": "active"}, 
            sort=[("trained_at", -1)]
        )
        return model_metadata
    finally:
        client.close()

# Get detailed model metrics
def get_detailed_model_metrics(model_version):
    """Get detailed metrics for a model version"""
    db, client = get_mongodb_connection()
    try:
        metrics = db.model_metrics.find_one(
            {"model_version": model_version},
            {"_id": 0}
        )
        return metrics
    finally:
        client.close()

# Load model files
@st.cache_resource
def load_model(model_path, scaler_path, feature_names_path):
    """Load model files for prediction"""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
        
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None, None

# Title and description
st.title("Retail Analytics Dashboard")
st.write("""
This dashboard provides insights into the retail app data pipeline and model performance.
Monitor data ingestion, processing status, and make predictions with the latest ML model.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Data Overview", "Data Processing", "Model Performance"]
)

# Display data overview page
if page == "Data Overview":
    st.header("Data Overview")
    
    with st.spinner("Loading data statistics..."):
        data_stats = get_data_stats()
    
    # Data Counts
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Raw Records", data_stats["raw_count"])
    with col2:
        st.metric("Processed Records", data_stats["processed_count"])
    
    # Recent Processing Jobs
    st.subheader("Recent Processing Jobs")
    if data_stats["recent_processing"]:
        processing_df = pd.DataFrame([
            {
                "Timestamp": meta.get("processed_at", "Unknown"),
                "Source File": meta.get("source_file", "Unknown") if "source_file" in meta else "N/A",
                "Records Processed": meta.get("record_count", 0),
                "Version": meta.get("processing_version", "Unknown")
            }
            for meta in data_stats["recent_processing"]
        ])
        st.dataframe(processing_df)
    else:
        st.write("No processing jobs found.")
    
    # Raw Data Sample
    st.subheader("Raw Data Sample")
    if st.button("Load Raw Data Sample"):
        with st.spinner("Loading raw data sample..."):
            raw_data = fetch_from_mongodb(limit=10)
            if not raw_data.empty:
                st.dataframe(raw_data)
            else:
                st.error("Failed to load raw data.")

# Display data processing page
elif page == "Data Processing":
    st.header("Data Processing")
    
    # Load processed data
    with st.spinner("Loading processed data..."):
        processed_data = load_processed_data(limit=1000)
    
    if processed_data.empty:
        st.error("No processed data available.")
    else:
        st.write(f"Loaded processed data with {len(processed_data)} records.")
        
        # Data Quality Metrics
        st.subheader("Data Quality Metrics")
        
        # Create columns for key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'purchase_24h' in processed_data.columns:
                purchase_rate = processed_data['purchase_24h'].mean() * 100
                st.metric("Purchase Rate (24h)", f"{purchase_rate:.2f}%")
        
        with col2:
            null_rate = processed_data.isnull().sum().sum() / (processed_data.shape[0] * processed_data.shape[1]) * 100
            st.metric("Missing Data Rate", f"{null_rate:.2f}%")
        
        with col3:
            if 'engagement_score' in processed_data.columns:
                avg_engagement = processed_data['engagement_score'].mean()
                st.metric("Avg. Engagement Score", f"{avg_engagement:.2f}")
        
        # Feature Distribution Plots
        st.subheader("Feature Distributions")
        
        # Select numeric columns for visualization
        numeric_cols = processed_data.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            selected_feature = st.selectbox("Select feature to visualize", numeric_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(processed_data[selected_feature], kde=True, ax=ax)
            ax.set_title(f"Distribution of {selected_feature}")
            st.pyplot(fig)
        
        # Correlation Heatmap
        st.subheader("Feature Correlations")
        
        if st.checkbox("Show Correlation Heatmap"):
            # Select top 15 features by correlation with target (if target exists)
            numeric_cols = processed_data.select_dtypes(include=['number']).columns
            if 'purchase_24h' in numeric_cols:
                corr_matrix = processed_data[numeric_cols].corr()
                target_corr = corr_matrix['purchase_24h'].abs().sort_values(ascending=False)
                top_features = target_corr.head(15).index.tolist()
                
                # Generate heatmap
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(
                    processed_data[top_features].corr(), 
                    annot=True, 
                    cmap='coolwarm', 
                    vmin=-1, 
                    vmax=1,
                    ax=ax
                )
                plt.title("Correlation Matrix of Top Features")
                st.pyplot(fig)
            else:
                st.write("Target variable not found in the dataset.")
        
        # Data Sample
        st.subheader("Processed Data Sample")
        st.dataframe(processed_data.head(10))

# Display model performance page
elif page == "Model Performance":
    st.header("Model Performance")
    
    # Get latest model info
    with st.spinner("Loading model information..."):
        model_info = get_latest_model_info()
    
    if not model_info:
        st.error("No active model found.")
    else:
        # Display model metadata
        st.subheader("Current Active Model")
        st.write(f"**Model Version:** {model_info.get('model_version', 'Unknown')}")
        st.write(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")
        st.write(f"**Trained At:** {model_info.get('trained_at', 'Unknown')}")
        st.write(f"**Data Version:** {model_info.get('data_version', 'Unknown')}")
        
        # Performance Metrics
        metrics = model_info.get('metrics', {})
        if metrics:
            st.subheader("Performance Metrics")
            
            # Create columns for metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
            with col4:
                st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
            
            # ROC AUC
            st.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.4f}")
        
        # Detailed Model Metrics
        st.subheader("Detailed Model Metrics")
        
        with st.spinner("Loading detailed metrics..."):
            detailed_metrics = get_detailed_model_metrics(model_info.get('model_version'))
        
        if detailed_metrics and 'detailed_metrics' in detailed_metrics:
            # Feature Importance
            if 'feature_importance' in detailed_metrics['detailed_metrics']:
                st.write("**Top 10 Feature Importance**")
                imp_df = pd.DataFrame(detailed_metrics['detailed_metrics']['feature_importance'])
                
                # Plot feature importance
                fig, ax = plt.subplots(figsize=(10, 8))
                top_10 = imp_df.head(10)
                sns.barplot(x='importance', y='feature', data=top_10, ax=ax)
                plt.title('Top 10 Feature Importance')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Confusion Matrix
            if 'confusion_matrix' in detailed_metrics['detailed_metrics']:
                st.write("**Confusion Matrix**")
                cm = np.array(detailed_metrics['detailed_metrics']['confusion_matrix'])
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted Labels')
                ax.set_ylabel('True Labels')
                ax.set_title('Confusion Matrix')
                ax.set_xticklabels(['No Purchase', 'Purchase'])
                ax.set_yticklabels(['No Purchase', 'Purchase'])
                st.pyplot(fig)
            
            # Cross-validation scores
            if 'cross_validation' in detailed_metrics['detailed_metrics']:
                cv_scores = detailed_metrics['detailed_metrics']['cross_validation']['scores']
                mean_cv = detailed_metrics['detailed_metrics']['cross_validation']['mean_f1']
                
                st.write(f"**Cross-validation Mean F1:** {mean_cv:.4f}")
                
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.boxplot(x=cv_scores, ax=ax)
                ax.axvline(x=mean_cv, color='r', linestyle='--')
                ax.set_title('F1 Score Cross-validation Distribution')
                st.pyplot(fig)
        else:
            st.write("Detailed metrics not available for this model.")

# Add footer
st.sidebar.markdown("---")
st.sidebar.write("Last data update:")
with st.spinner("Loading last update time..."):
    data_stats = get_data_stats()
    if data_stats["recent_processing"]:
        last_update = data_stats["recent_processing"][0].get("processed_at", "Unknown")
        st.sidebar.write(f"{last_update}")
    else:
        st.sidebar.write("No recent updates")