# app.py - Streamlit app for Retail ML Pipeline

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Helper functions
def connect_to_mongodb():
    """Connect to MongoDB and return database connection."""
    # Load environment variables
    load_dotenv()
    
    # Get MongoDB connection details
    username = os.getenv("MONGODB_USERNAME")
    password = os.getenv("MONGODB_PASSWORD")
    cluster = os.getenv("MONGODB_CLUSTER")
    database = os.getenv("MONGODB_DATABASE")
    
    # Create connection string
    connection_string = f"mongodb+srv://{username}:{password}@{cluster}/"
    
    try:
        # Connect to MongoDB
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
        
        # Get the active model metadata
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
        
        # Get metrics for the specific model
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
        
        # Get all models, sorted by trained date
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
        
        # Get recent predictions
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
        
        # Get all prediction run metadata
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

# Detect if running in Docker
is_docker = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER", "false").lower() == "true"
if is_docker:
    st.info("🐳 Running in Docker environment", icon="🐳")
else:
    st.info("💻 Running in local environment", icon="💻")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📈 Model Metrics", "🔮 Predictions", "🧪 Single Prediction"])

# Model Metrics tab
with tab1:
    st.header("Model Performance & Metrics")
    
    # Get active model
    active_model = get_active_model_metadata()
    
    if active_model:
        # Display active model info
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
            
        # Get detailed metrics
        detailed_metrics = get_model_metrics(active_model['model_version'])
        
        if detailed_metrics and 'detailed_metrics' in detailed_metrics:
            # Feature importance
            st.subheader("Feature Importance")
            
            feature_importance = detailed_metrics['detailed_metrics'].get('feature_importance', [])
            if feature_importance:
                # Convert to dataframe and sort
                fi_df = pd.DataFrame(feature_importance).sort_values('importance', ascending=False)
                
                # Take top 15 features for better visualization
                top_features = fi_df.head(15)
                
                # Create bar chart with Plotly
                fig = px.bar(
                    top_features, 
                    x='importance', 
                    y='feature', 
                    orientation='h',
                    title="Top 15 Most Important Features",
                    labels={'importance': 'Importance', 'feature': 'Feature'},
                    color='importance',
                    color_continuous_scale='viridis'
                )
                
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("View All Feature Importance"):
                    st.dataframe(fi_df)
                
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            
            confusion_matrix = detailed_metrics['detailed_metrics'].get('confusion_matrix', [])
            if confusion_matrix:
                # Create confusion matrix plot
                labels = ['Not Purchased (0)', 'Purchased (1)']
                
                fig = go.Figure(data=go.Heatmap(
                    z=confusion_matrix,
                    x=labels,
                    y=labels,
                    colorscale='Blues',
                    showscale=False,
                    text=confusion_matrix,
                    texttemplate="%{text}",
                    textfont={"size": 16},
                ))
                
                fig.update_layout(
                    title="Confusion Matrix",
                    xaxis_title="Predicted Label",
                    yaxis_title="True Label",
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate metrics from confusion matrix
                tn, fp = confusion_matrix[0]
                fn, tp = confusion_matrix[1]
                
                # Display metrics
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                with metrics_col1:
                    st.metric("True Positives", tp)
                
                with metrics_col2:
                    st.metric("True Negatives", tn)
                
                with metrics_col3:
                    st.metric("False Positives", fp)
                
                with metrics_col4:
                    st.metric("False Negatives", fn)
        
        # Show model history
        st.subheader("Model History")
        models = get_all_models()
        
        if models:
            # Create DataFrame
            models_df = pd.DataFrame(models)
            
            # Format dates
            models_df['trained_at'] = pd.to_datetime(models_df['trained_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Extract metrics
            models_df['accuracy'] = models_df['metrics'].apply(lambda x: x.get('accuracy', 'N/A') if x else 'N/A')
            models_df['f1_score'] = models_df['metrics'].apply(lambda x: x.get('f1_score', 'N/A') if x else 'N/A')
            
            # Display as table
            st.dataframe(
                models_df[['model_version', 'trained_at', 'status', 'accuracy', 'f1_score']],
                use_container_width=True
            )
            
            # Plot model performance over time
            if len(models) > 1:
                st.subheader("Model Performance Over Time")
                
                # Create performance DataFrame
                perf_df = pd.DataFrame([
                    {
                        'model_version': m['model_version'],
                        'trained_at': m['trained_at'],
                        'accuracy': m['metrics'].get('accuracy', None) if 'metrics' in m else None,
                        'f1_score': m['metrics'].get('f1_score', None) if 'metrics' in m else None,
                        'precision': m['metrics'].get('precision', None) if 'metrics' in m else None,
                        'recall': m['metrics'].get('recall', None) if 'metrics' in m else None
                    }
                    for m in models if 'metrics' in m
                ])
                
                if not perf_df.empty:
                    # Sort by trained_at
                    perf_df = perf_df.sort_values('trained_at')
                    
                    # Plot metrics over time
                    fig = go.Figure()
                    
                    metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall']
                    for metric in metrics_to_plot:
                        fig.add_trace(go.Scatter(
                            x=range(len(perf_df)),
                            y=perf_df[metric],
                            mode='lines+markers',
                            name=metric.capitalize().replace('_', ' ')
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title="Model Performance Metrics Over Time",
                        xaxis_title="Model Sequence",
                        yaxis_title="Metric Value",
                        xaxis=dict(
                            tickmode='array',
                            tickvals=list(range(len(perf_df))),
                            ticktext=[m.split('_')[2] for m in perf_df['model_version']]
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("No active model found in the database.")

# Predictions tab
with tab2:
    st.header("Prediction Results")
    
    # Get prediction metadata
    prediction_meta = get_prediction_metadata()
    
    if prediction_meta is not None:
        # Display prediction run stats
        st.subheader("Prediction Run History")
        
        # Format dates for display
        prediction_meta['prediction_time'] = pd.to_datetime(prediction_meta['prediction_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Display recent prediction runs
        st.dataframe(
            prediction_meta[['prediction_id', 'model_version', 'prediction_time', 'record_count', 'positive_rate', 'high_probability_rate']],
            use_container_width=True
        )
        
        # Plot prediction distribution over time
        if len(prediction_meta) > 1:
            st.subheader("Prediction Distribution Over Time")
            
            # Prepare data
            plot_data = prediction_meta.copy()
            plot_data['prediction_time'] = pd.to_datetime(plot_data['prediction_time'])
            plot_data = plot_data.sort_values('prediction_time')
            
            # Create plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=plot_data['prediction_time'],
                y=plot_data['positive_rate'],
                mode='lines+markers',
                name='Positive Rate',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=plot_data['prediction_time'],
                y=plot_data['high_probability_rate'],
                mode='lines+markers',
                name='High Probability Rate',
                line=dict(color='green')
            ))
            
            fig.update_layout(
                title="Prediction Rates Over Time",
                xaxis_title="Prediction Time",
                yaxis_title="Rate",
                yaxis=dict(tickformat=".0%"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Get detailed predictions
        predictions = get_recent_predictions()
        
        if predictions is not None:
            # Show prediction results
            st.subheader("Recent Predictions")
            
            # Handle datetime columns
            datetime_cols = predictions.select_dtypes(include=['datetime']).columns
            for col in datetime_cols:
                predictions[col] = pd.to_datetime(predictions[col])
            
            # Create prediction distribution chart
            st.subheader("Prediction Probability Distribution")
            
            if 'purchase_24h_probability' in predictions.columns:
                fig = px.histogram(
                    predictions, 
                    x='purchase_24h_probability',
                    nbins=20,
                    labels={'purchase_24h_probability': 'Purchase Probability'},
                    title="Distribution of Purchase Probabilities",
                    color_discrete_sequence=['skyblue']
                )
                
                fig.update_layout(
                    xaxis_title="Purchase Probability",
                    yaxis_title="Count"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Filter options
            st.subheader("Filter Predictions")
            
            # Add filters
            col1, col2 = st.columns(2)
            
            with col1:
                min_prob = st.slider("Minimum Probability", 0.0, 1.0, 0.0, 0.05)
            
            with col2:
                prediction_class = st.radio("Prediction Class", ["All", "Positive (1)", "Negative (0)"])
            
            # Apply filters
            filtered_predictions = predictions.copy()
            
            if min_prob > 0:
                filtered_predictions = filtered_predictions[filtered_predictions['purchase_24h_probability'] >= min_prob]
            
            if prediction_class == "Positive (1)":
                filtered_predictions = filtered_predictions[filtered_predictions['purchase_24h_prediction'] == 1]
            elif prediction_class == "Negative (0)":
                filtered_predictions = filtered_predictions[filtered_predictions['purchase_24h_prediction'] == 0]
            
            # Display filtered data
            st.dataframe(filtered_predictions, use_container_width=True)
            
            # Download button
            csv = filtered_predictions.to_csv(index=False)
            st.download_button(
                label="Download Filtered Predictions as CSV",
                data=csv,
                file_name="retail_predictions.csv",
                mime="text/csv",
            )
        else:
            st.warning("No prediction results found.")
    else:
        st.warning("No prediction data found in the database.")

# Single Prediction tab
with tab3:
    st.header("Single Record Prediction")
    st.write("Enter feature values to get a prediction for a single record")
    
    # Create form for feature input
    with st.form("prediction_form"):
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            # User and session features
            st.subheader("User Information")
            user_id = st.text_input("User ID", value="user_" + datetime.now().strftime("%Y%m%d%H%M%S"))
            platform = st.selectbox("Platform", options=["iOS", "Android"])
            age = st.number_input("Age", min_value=13, max_value=100, value=30)
            region = st.selectbox("Region", options=["NorthAmerica", "Europe", "Asia", "LatinAmerica", "Oceania", "Africa", "MiddleEast"])
            acquisition_channel = st.selectbox("Acquisition Channel", options=["Organic", "Paid", "Referral", "Social", "Email"])
            user_segment = st.selectbox("User Segment", options=["Young Professional", "Student", "Parent", "Senior", "Teen"])
            
            # App version
            st.subheader("App Information")
            app_version = st.text_input("App Version", value="3.2.1")
            first_visit_date = st.date_input("First Visit Date", value=datetime.now() - timedelta(days=7))
        
        with col2:
            # Engagement metrics
            st.subheader("Engagement Metrics")
            session_count = st.number_input("Session Count", min_value=1, max_value=100, value=5)
            total_screens_viewed = st.number_input("Total Screens Viewed", min_value=1, max_value=500, value=15)
            
            # Boolean features
            used_search_feature = st.checkbox("Used Search Feature", value=True)
            wrote_review = st.checkbox("Wrote Review", value=False)
            added_to_wishlist = st.checkbox("Added to Wishlist", value=True)
            
            # Screen list
            st.subheader("Screens Visited")
            screen_options = [
                "ProductList", "ProductDetail", "CategoryBrowse", "Search",
                "ShoppingCart", "Checkout", "PaymentMethods", "DeliveryOptions",
                "WishList", "Reviews", "Promotions", 
                "Account", "AddressBook", "OrderTracking"
            ]
            selected_screens = st.multiselect("Select Screens Visited", options=screen_options, 
                                             default=["ProductList", "ProductDetail", "ShoppingCart"])
        
        # Create a screen_list string from selected screens
        screen_list = ",".join(selected_screens)
        
        # Convert boolean checkboxes to integers (0/1)
        used_search_feature_int = 1 if used_search_feature else 0
        wrote_review_int = 1 if wrote_review else 0
        added_to_wishlist_int = 1 if added_to_wishlist else 0
        
        # Submit button
        submitted = st.form_submit_button("Make Prediction")
    
    # Process the form submission
    if submitted:
        # Prepare feature dictionary
        feature_dict = {
            "user_id": user_id,
            "platform": platform,
            "age": age,
            "session_count": session_count,
            "total_screens_viewed": total_screens_viewed,
            "used_search_feature": used_search_feature_int,
            "wrote_review": wrote_review_int,
            "added_to_wishlist": added_to_wishlist_int,
            "screen_list": screen_list,
            "region": region,
            "acquisition_channel": acquisition_channel,
            "user_segment": user_segment,
            "app_version": app_version,
            "first_visit_date": first_visit_date.strftime("%Y-%m-%d")
        }
        
        # Display a spinner while making prediction
        with st.spinner("Processing prediction..."):
            # Call the prediction function
            result = single_prediction.make_single_prediction(feature_dict)
        
        # Check if prediction was successful
        if result["success"]:
            # Show the prediction result
            st.subheader("Prediction Result")
            
            # Create columns for results
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                # Show prediction class
                if result["prediction"] == 1:
                    st.success("Prediction: **Will Purchase** within 24 hours")
                else:
                    st.error("Prediction: **Will Not Purchase** within 24 hours")
                
                # Show probability
                st.metric("Purchase Probability", f"{result['probability']:.2%}")
                
                # Option to save prediction
                if st.button("Save This Prediction"):
                    if single_prediction.save_prediction_to_mongodb(feature_dict, result):
                        st.success("Prediction saved successfully!")
                    else:
                        st.error("Failed to save prediction.")
            
            with result_col2:
                # Create a gauge chart for probability visualization
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["probability"],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Purchase Probability"},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.3], 'color': "red"},
                            {'range': [0.3, 0.7], 'color': "yellow"},
                            {'range': [0.7, 1], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.7
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Show feature influences if available
            if 'feature_influences' in result and 'top_features' in result['feature_influences']:
                st.subheader("Key Influencing Features")
                
                # Convert to DataFrame for display
                influence_df = pd.DataFrame(result['feature_influences']['top_features'])
                
                # Create horizontal bar chart of influences
                fig = px.bar(
                    influence_df, 
                    x='influence', 
                    y='feature', 
                    orientation='h',
                    labels={'influence': 'Influence', 'feature': 'Feature'},
                    color='influence',
                    color_continuous_scale='RdBu',
                    title="Top 5 Features Influencing This Prediction"
                )
                
                # Set the colorscale to center at zero
                max_abs_influence = max(abs(influence_df['influence'].min()), abs(influence_df['influence'].max()))
                fig.update_layout(
                    coloraxis_colorbar=dict(
                        title="Influence",
                        tickvals=[-max_abs_influence, 0, max_abs_influence],
                        ticktext=["Negative", "Neutral", "Positive"],
                    ),
                    xaxis=dict(range=[-max_abs_influence, max_abs_influence])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("View All Feature Influences"):
                    all_influences = pd.DataFrame(result['feature_influences']['all_features'])
                    st.dataframe(all_influences.sort_values('abs_influence', ascending=False))
        else:
            # Show error
            st.error(f"Error making prediction: {result.get('error', 'Unknown error')}")
            st.warning("Please make sure all fields are filled correctly and try again.")

# Sidebar
st.sidebar.title("Retail ML Pipeline")
st.sidebar.info("This dashboard provides real-time insights into the retail machine learning pipeline.")

# Show active model in sidebar
active_model = get_active_model_metadata()
if active_model:
    st.sidebar.subheader("Active Model")
    st.sidebar.code(active_model['model_version'])
    
    metrics = active_model.get('metrics', {})
    if metrics:
        st.sidebar.metric("F1 Score", f"{metrics.get('f1_score', 'N/A'):.4f}")

# Last update time
st.sidebar.subheader("Dashboard Info")
st.sidebar.text(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Add refresh button
if st.sidebar.button("Refresh Data"):
    st.experimental_rerun()