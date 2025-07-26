# app.py - Streamlit app for Retail ML Pipeline

import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import json
from datetime import datetime, timedelta
import single_prediction

# Set page configuration
st.set_page_config(
    page_title="Retail ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Helper function to convert dataframe to excel for download (NEW)
@st.cache_data
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
    processed_data = output.getvalue()
    return processed_data

# Helper functions (Original code restored)
def connect_to_mongodb():
    """Connect to MongoDB and return database connection."""
    load_dotenv()
    username = os.getenv("MONGODB_USERNAME")
    password = os.getenv("MONGODB_PASSWORD")
    cluster = os.getenv("MONGODB_CLUSTER")
    database = os.getenv("MONGODB_DATABASE")
    connection_string = f"mongodb+srv://{username}:{password}@{cluster}/"
    try:
        client = MongoClient(connection_string)
        db = client.get_database(database)
        return db, client
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {str(e)}")
        return None, None

def get_active_model_metadata():
    """Get metadata for the active model."""
    db, client = connect_to_mongodb()
    if db is None:
        return None
    try:
        model_metadata_collection = db.model_metadata
        active_model = model_metadata_collection.find_one(
            {"status": "active"},
            sort=[("trained_at", -1)]
        )
        client.close()
        return active_model
    except Exception as e:
        st.error(f"Error fetching model metadata: {str(e)}")
        if 'client' in locals():
            client.close()
        return None

def get_model_metrics(model_version):
    """Get detailed metrics for a specific model version."""
    db, client = connect_to_mongodb()
    if db is None:
        return None
    try:
        metrics_collection = db.model_metrics
        model_metrics = metrics_collection.find_one({"model_version": model_version})
        client.close()
        return model_metrics
    except Exception as e:
        st.error(f"Error fetching model metrics: {str(e)}")
        if 'client' in locals():
            client.close()
        return None

def get_all_models():
    """Get a list of all models with their status."""
    db, client = connect_to_mongodb()
    if db is None:
        return []
    try:
        model_metadata_collection = db.model_metadata
        all_models = list(model_metadata_collection.find(
            {},
            {"model_version": 1, "trained_at": 1, "status": 1, "metrics": 1, "_id": 0}
        ).sort("trained_at", -1))
        client.close()
        return all_models
    except Exception as e:
        st.error(f"Error fetching model list: {str(e)}")
        if 'client' in locals():
            client.close()
        return []

def get_recent_predictions(limit=1000):
    """Get recent prediction results."""
    db, client = connect_to_mongodb()
    if db is None:
        return None
    try:
        prediction_collection = db.predicted_results
        recent_predictions = list(prediction_collection.find(
            {},
            {"_id": 0}
        ).sort("prediction_timestamp", -1).limit(limit))
        client.close()
        if recent_predictions:
            return pd.DataFrame(recent_predictions)
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching prediction results: {str(e)}")
        if 'client' in locals():
            client.close()
        return None

def get_prediction_metadata():
    """Get metadata for all prediction runs."""
    db, client = connect_to_mongodb()
    if db is None:
        return None
    try:
        metadata_collection = db.prediction_metadata
        metadata = list(metadata_collection.find(
            {},
            {"_id": 0}
        ).sort("prediction_time", -1))
        client.close()
        if metadata:
            return pd.DataFrame(metadata)
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching prediction metadata: {str(e)}")
        if 'client' in locals():
            client.close()
        return None

# App header
st.title("📊 Retail ML Dashboard")
st.write("Real-time monitoring for the retail machine learning pipeline")

# Create tabs (Added a 4th tab)
tab1, tab2, tab3, tab4 = st.tabs(["📈 Model Metrics", "🔮 Predictions", "🧪 Single Prediction", "📂 Batch Prediction"])

# Model Metrics tab (Original code restored)
with tab1:
    st.header("Model Performance & Metrics")
    active_model = get_active_model_metadata()
    if active_model:
        st.subheader("Active Model")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Model Version**: {active_model['model_version']}")
            trained_at = active_model['trained_at']
            st.info(f"**Trained At**: {trained_at.strftime('%Y-%m-%d %H:%M:%S')}")
        with col2:
            metrics = active_model.get('metrics', {})
            st.info(f"**Accuracy**: {metrics.get('accuracy', 'N/A'):.4f}")
            st.info(f"**F1 Score**: {metrics.get('f1_score', 'N/A'):.4f}")
        with col3:
            st.info(f"**Precision**: {metrics.get('precision', 'N/A'):.4f}")
            st.info(f"**Recall**: {metrics.get('recall', 'N/A'):.4f}")
        detailed_metrics = get_model_metrics(active_model['model_version'])
        if detailed_metrics and 'detailed_metrics' in detailed_metrics:
            st.subheader("Feature Importance")
            feature_importance = detailed_metrics['detailed_metrics'].get('feature_importance', [])
            if feature_importance:
                fi_df = pd.DataFrame(feature_importance).sort_values('importance', ascending=False)
                top_features = fi_df.head(15)
                fig = px.bar(top_features, x='importance', y='feature', orientation='h', title="Top 15 Most Important Features")
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("View All Feature Importance"):
                    st.dataframe(fi_df)
            st.subheader("Confusion Matrix")
            confusion_matrix = detailed_metrics['detailed_metrics'].get('confusion_matrix', [])
            if confusion_matrix:
                labels = ['Not Purchased (0)', 'Purchased (1)']
                fig = go.Figure(data=go.Heatmap(z=confusion_matrix, x=labels, y=labels, colorscale='Blues', text=confusion_matrix, texttemplate="%{text}"))
                fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted Label", yaxis_title="True Label")
                st.plotly_chart(fig, use_container_width=True)
                tn, fp = confusion_matrix[0]
                fn, tp = confusion_matrix[1]
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                with metrics_col1: st.metric("True Positives", tp)
                with metrics_col2: st.metric("True Negatives", tn)
                with metrics_col3: st.metric("False Positives", fp)
                with metrics_col4: st.metric("False Negatives", fn)
        st.subheader("Model History")
        models = get_all_models()
        if models:
            models_df = pd.DataFrame(models)
            models_df['trained_at'] = pd.to_datetime(models_df['trained_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
            models_df['accuracy'] = models_df['metrics'].apply(lambda x: x.get('accuracy', 'N/A') if x else 'N/A')
            models_df['f1_score'] = models_df['metrics'].apply(lambda x: x.get('f1_score', 'N/A') if x else 'N/A')
            st.dataframe(models_df[['model_version', 'trained_at', 'status', 'accuracy', 'f1_score']], use_container_width=True)
            if len(models) > 1:
                st.subheader("Model Performance Over Time")
                perf_df = pd.DataFrame([m for m in models if 'metrics' in m and m.get('metrics') is not None])
                if not perf_df.empty:
                    perf_df = perf_df.sort_values('trained_at').reset_index(drop=True)
                    fig = go.Figure()
                    metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall']
                    for metric in metrics_to_plot:
                        # Extract metric safely
                        metric_values = perf_df['metrics'].apply(lambda x: x.get(metric) if isinstance(x, dict) else np.nan)
                        fig.add_trace(go.Scatter(x=perf_df.index, y=metric_values, mode='lines+markers', name=metric.capitalize().replace('_', ' ')))
                    fig.update_layout(title="Model Performance Metrics Over Time", xaxis_title="Model Sequence", yaxis_title="Metric Value", xaxis=dict(tickmode='array', tickvals=list(perf_df.index), ticktext=perf_df['model_version']), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No active model found in the database.")

# Predictions tab (Original code restored)
with tab2:
    st.header("Prediction Results")
    prediction_meta = get_prediction_metadata()
    if prediction_meta is not None:
        st.subheader("Prediction Run History")
        prediction_meta['prediction_time'] = pd.to_datetime(prediction_meta['prediction_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(prediction_meta[['prediction_id', 'model_version', 'prediction_time', 'record_count', 'positive_rate', 'high_probability_rate']], use_container_width=True)
        if len(prediction_meta) > 1:
            st.subheader("Prediction Distribution Over Time")
            plot_data = prediction_meta.copy()
            plot_data['prediction_time'] = pd.to_datetime(plot_data['prediction_time'])
            plot_data = plot_data.sort_values('prediction_time')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_data['prediction_time'], y=plot_data['positive_rate'], mode='lines+markers', name='Positive Rate', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=plot_data['prediction_time'], y=plot_data['high_probability_rate'], mode='lines+markers', name='High Probability Rate', line=dict(color='green')))
            fig.update_layout(title="Prediction Rates Over Time", xaxis_title="Prediction Time", yaxis_title="Rate", yaxis=dict(tickformat=".0%"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
    predictions = get_recent_predictions()
    if predictions is not None:
        st.subheader("Recent Predictions")
        st.subheader("Prediction Probability Distribution")
        if 'purchase_24h_probability' in predictions.columns:
            fig = px.histogram(predictions, x='purchase_24h_probability', nbins=20, labels={'purchase_24h_probability': 'Purchase Probability'}, title="Distribution of Purchase Probabilities")
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Filter Predictions")
        col1, col2 = st.columns(2)
        with col1:
            min_prob = st.slider("Minimum Probability", 0.0, 1.0, 0.0, 0.05)
        with col2:
            prediction_class = st.radio("Prediction Class", ["All", "Positive (1)", "Negative (0)"])
        filtered_predictions = predictions.copy()
        if min_prob > 0:
            filtered_predictions = filtered_predictions[filtered_predictions['purchase_24h_probability'] >= min_prob]
        if prediction_class == "Positive (1)":
            filtered_predictions = filtered_predictions[filtered_predictions['purchase_24h_prediction'] == 1]
        elif prediction_class == "Negative (0)":
            filtered_predictions = filtered_predictions[filtered_predictions['purchase_24h_prediction'] == 0]
        st.dataframe(filtered_predictions, use_container_width=True)
        csv = filtered_predictions.to_csv(index=False)
        st.download_button(label="Download Filtered Predictions as CSV", data=csv, file_name="retail_predictions.csv", mime="text/csv")
    else:
        st.warning("No prediction results found.")

# Single Prediction tab (Original code restored)
with tab3:
    st.header("Single Record Prediction")
    st.write("Enter feature values to get a prediction for a single record")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("User Information")
            user_id = st.text_input("User ID", value="user_" + datetime.now().strftime("%Y%m%d%H%M%S"))
            platform = st.selectbox("Platform", options=["iOS", "Android"])
            age = st.number_input("Age", min_value=13, max_value=100, value=30)
            region = st.selectbox("Region", options=["NorthAmerica", "Europe", "Asia", "LatinAmerica", "Oceania", "Africa", "MiddleEast"])
            acquisition_channel = st.selectbox("Acquisition Channel", options=["Organic", "Paid", "Referral", "Social", "Email"])
            user_segment = st.selectbox("User Segment", options=["Young Professional", "Student", "Parent", "Senior", "Teen"])
            st.subheader("App Information")
            app_version = st.text_input("App Version", value="3.2.1")
            first_visit_date = st.date_input("First Visit Date", value=datetime.now() - timedelta(days=7))
        with col2:
            st.subheader("Engagement Metrics")
            session_count = st.number_input("Session Count", min_value=1, max_value=100, value=5)
            total_screens_viewed = st.number_input("Total Screens Viewed", min_value=1, max_value=500, value=15)
            used_search_feature = st.checkbox("Used Search Feature", value=True)
            wrote_review = st.checkbox("Wrote Review", value=False)
            added_to_wishlist = st.checkbox("Added to Wishlist", value=True)
            st.subheader("Screens Visited")
            screen_options = ["ProductList", "ProductDetail", "CategoryBrowse", "Search", "ShoppingCart", "Checkout", "PaymentMethods", "DeliveryOptions", "WishList", "Reviews", "Promotions", "Account", "AddressBook", "OrderTracking"]
            selected_screens = st.multiselect("Select Screens Visited", options=screen_options, default=["ProductList", "ProductDetail", "ShoppingCart"])
        screen_list = ",".join(selected_screens)
        used_search_feature_int = 1 if used_search_feature else 0
        wrote_review_int = 1 if wrote_review else 0
        added_to_wishlist_int = 1 if added_to_wishlist else 0
        submitted = st.form_submit_button("Make Prediction")
    if submitted:
        feature_dict = { "user_id": user_id, "platform": platform, "age": age, "session_count": session_count, "total_screens_viewed": total_screens_viewed, "used_search_feature": used_search_feature_int, "wrote_review": wrote_review_int, "added_to_wishlist": added_to_wishlist_int, "screen_list": screen_list, "region": region, "acquisition_channel": acquisition_channel, "user_segment": user_segment, "app_version": app_version, "first_visit_date": first_visit_date.strftime("%Y-%m-%d") }
        with st.spinner("Processing prediction..."):
            result = single_prediction.make_single_prediction(feature_dict)
        if result["success"]:
            st.subheader("Prediction Result")
            if result["prediction"] == 1:
                st.success("Prediction: **Will Purchase** within 24 hours")
            else:
                st.error("Prediction: **Will Not Purchase** within 24 hours")
            st.metric("Purchase Probability", f"{result['probability']:.2%}")
        else:
            st.error(f"Error making prediction: {result.get('error', 'Unknown error')}")

# --- NEW BATCH PREDICTION TAB ---
with tab4:
    st.header("📂 Batch Prediction from File")
    st.write("Upload a CSV or Excel file with user data to get predictions in bulk.")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx'],
        help="The file should contain columns like: user_id, platform, age, session_count, etc."
    )
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                input_df = pd.read_csv(uploaded_file)
            else:
                input_df = pd.read_excel(uploaded_file)
            st.subheader("Uploaded Data Preview")
            st.dataframe(input_df.head())
            if st.button("Run Batch Prediction", type="primary"):
                with st.spinner("Processing and predicting..."):
                    model, scaler, feature_names, scaling_info = single_prediction.get_active_model_artifacts()
                    if model is None:
                        st.error("Failed to load the active model. Cannot proceed.")
                    else:
                        # Ensure all necessary columns exist, adding defaults if they don't
                        default_values = {
                            'wrote_review': 0, 'acquisition_channel': 'Organic',
                            'user_segment': 'Young Professional', 'first_visit_date': datetime.now().strftime("%Y-%m-%d"),
                            'platform': 'iOS', 'age': 30, 'region': 'NorthAmerica', 'app_version': '1.0.0',
                            'session_count': 1, 'total_screens_viewed': 1, 'used_search_feature': 0,
                            'added_to_wishlist': 0, 'screen_list': ''
                        }
                        for col, default in default_values.items():
                            if col not in input_df.columns:
                                input_df[col] = default
                        # Process the dataframe using the existing function
                        X_processed = single_prediction.process_single_record(input_df.copy(), feature_names, scaling_info, scaler)
                        predictions = model.predict(X_processed)
                        probabilities = model.predict_proba(X_processed)[:, 1]
                        results_df = input_df.copy()
                        results_df['purchase_prediction'] = predictions
                        results_df['purchase_probability'] = probabilities
                        st.session_state['batch_results'] = results_df
                        st.success("Batch prediction complete!")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    if 'batch_results' in st.session_state:
        st.subheader("Prediction Results")
        results_df_display = st.session_state['batch_results']
        st.dataframe(results_df_display)
        excel_data = to_excel(results_df_display)
        st.download_button(
            label="📥 Download Results as Excel",
            data=excel_data,
            file_name="batch_prediction_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )