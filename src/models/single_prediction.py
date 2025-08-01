# single_prediction.py
# Module for handling single record predictions

import pandas as pd
import numpy as np
import joblib
import json
import logging
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('retail_single_prediction')

# Dynamically determine if running in Docker or locally
is_docker = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER", "false").lower() == "true"

# Set MODEL_DIR based on environment
if is_docker:
    MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
else:
    MODEL_DIR = "../models"

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
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        return None, None

def get_active_model_artifacts():
    """Get the active model, scaler, and feature names."""
    try:
        db, client = connect_to_mongodb()
        
        if db is None:
            return None, None, None, None
        
        model_metadata_collection = db.model_metadata
        
        # Get the active model metadata
        active_model = model_metadata_collection.find_one(
            {"status": "active"}, 
            sort=[("trained_at", -1)]
        )
        
        client.close()
        
        if not active_model:
            logger.error("No active model found in database")
            return None, None, None, None
        
        model_version = active_model['model_version']
        
        # Load model - fix path for Docker environment
        model_path = active_model['model_path']
        if is_docker and model_path.startswith('../models/'):
            model_path = model_path.replace('../models/', '/app/models/')
        logger.info(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        
        # Load scaler - fix path for Docker environment  
        scaler_path = active_model['scaler_path']
        if is_docker and scaler_path.startswith('../models/'):
            scaler_path = scaler_path.replace('../models/', '/app/models/')
        logger.info(f"Loading scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path)
        
        # Load feature names - fix path for Docker environment
        feature_names_path = active_model['feature_names_path']
        if is_docker and feature_names_path.startswith('../models/'):
            feature_names_path = feature_names_path.replace('../models/', '/app/models/')
        logger.info(f"Loading feature names from: {feature_names_path}")
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
        
        # Load scaling info - fix path for Docker environment
        scaling_info_path = active_model['scaling_info_path']
        if is_docker and scaling_info_path.startswith('../models/'):
            scaling_info_path = scaling_info_path.replace('../models/', '/app/models/')
        logger.info(f"Loading scaling info from: {scaling_info_path}")
        with open(scaling_info_path, 'r') as f:
            scaling_info = json.load(f)
        
        return model, scaler, feature_names, scaling_info
    
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        return None, None, None, None

def create_feature_df(feature_dict):
    """Convert feature dictionary to a DataFrame."""
    # Create a single row DataFrame
    df = pd.DataFrame([feature_dict])
    return df

def process_single_record(df, feature_names, scaling_info, scaler):
    """Process a single record for prediction."""
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Time-based Processing
    if 'first_visit_date' in df_processed.columns:
        df_processed['first_visit_date'] = pd.to_datetime(df_processed['first_visit_date'], errors='coerce')
        df_processed['hour'] = df_processed['first_visit_date'].dt.hour
        df_processed['dayofweek'] = df_processed['first_visit_date'].dt.dayofweek
        df_processed['is_weekend'] = df_processed['dayofweek'].isin([5,6]).astype(int)
    
    # Screen List Processing
    if 'screen_list' in df_processed.columns:
        # Convert screen list to string and add comma for consistent processing
        df_processed['screen_list'] = df_processed['screen_list'].astype(str) + ','
        
        # Define screen categories
        shopping_screens = ['ProductList', 'ProductDetail', 'CategoryBrowse', 'Search']
        cart_screens = ['ShoppingCart', 'Checkout', 'PaymentMethods', 'DeliveryOptions']
        engagement_screens = ['WishList', 'Reviews', 'Promotions']
        account_screens = ['Account', 'AddressBook', 'OrderTracking']
        
        # Create binary indicators for each screen
        for screen in (shopping_screens + cart_screens + engagement_screens + account_screens):
            df_processed[screen.lower()] = df_processed['screen_list'].str.contains(screen).astype(int)
        
        # Create count features for each category
        df_processed['shopping_count'] = df_processed[[s.lower() for s in shopping_screens]].sum(axis=1)
        df_processed['cart_count'] = df_processed[[s.lower() for s in cart_screens]].sum(axis=1)
        df_processed['engagement_count'] = df_processed[[s.lower() for s in engagement_screens]].sum(axis=1)
        df_processed['account_count'] = df_processed[[s.lower() for s in account_screens]].sum(axis=1)
        
        # Create Other category
        all_tracked_screens = shopping_screens + cart_screens + engagement_screens + account_screens
        df_processed['other_screens'] = df_processed['screen_list'].apply(
            lambda x: len([s for s in x.split(',') if s and s not in all_tracked_screens])
        )
    
    # Feature Engineering
    if all(col in df_processed.columns for col in ['session_count', 'used_search_feature', 'wrote_review', 'added_to_wishlist', 'total_screens_viewed']):
        df_processed['engagement_score'] = (
            df_processed['session_count'] * 0.3 +
            df_processed['used_search_feature'] * 0.2 +
            df_processed['wrote_review'] * 0.15 +
            df_processed['added_to_wishlist'] * 0.15 +
            df_processed['total_screens_viewed'] * 0.2
        )
    
    if all(col in df_processed.columns for col in ['shopping_count', 'cart_count', 'engagement_count', 'account_count']):
        # Create screen diversity score
        df_processed['screen_diversity'] = (
            df_processed[['shopping_count', 'cart_count', 'engagement_count', 'account_count']].gt(0).sum(axis=1)
        )
        
        # Create purchase intent score
        if 'added_to_wishlist' in df_processed.columns:
            df_processed['purchase_intent'] = (
                df_processed['cart_count'] * 0.4 +
                df_processed['shopping_count'] * 0.3 +
                df_processed['engagement_count'] * 0.2 +
                df_processed['added_to_wishlist'] * 0.1
            )
    
    # Categorical Feature Processing
    # Platform encoding
    if 'platform' in df_processed.columns:
        df_processed['platform'] = df_processed['platform'].map({'iOS': 1, 'Android': 0})
    
    # Region encoding
    if 'region' in df_processed.columns:
        region_dummies = pd.get_dummies(df_processed['region'], prefix='region')
        df_processed = pd.concat([df_processed, region_dummies], axis=1)
    
    # Acquisition channel encoding
    if 'acquisition_channel' in df_processed.columns:
        channel_dummies = pd.get_dummies(df_processed['acquisition_channel'], prefix='channel')
        df_processed = pd.concat([df_processed, channel_dummies], axis=1)
    
    # User segment processing
    if 'user_segment' in df_processed.columns:
        df_processed['age_group'] = df_processed['user_segment'].apply(lambda x: x.split()[0])
        df_processed['user_type'] = df_processed['user_segment'].apply(lambda x: ' '.join(x.split()[1:]))
        
        age_group_dummies = pd.get_dummies(df_processed['age_group'], prefix='age_group')
        user_type_dummies = pd.get_dummies(df_processed['user_type'], prefix='user_type')
        df_processed = pd.concat([df_processed, age_group_dummies, user_type_dummies], axis=1)
    
    # App version processing
    if 'app_version' in df_processed.columns:
        df_processed['app_major_version'] = df_processed['app_version'].apply(lambda x: int(x.split('.')[0]))
        
        # Create version recency score
        df_processed['version_score'] = df_processed['app_version'].apply(
            lambda x: sum(float(i)/(10**n) for n, i in enumerate(x.split('.')))
        )
    
    # Clean up and prepare final dataset
    # Drop original columns that have been processed
    columns_to_drop = [
        'screen_list', 'first_visit_date', 
        'region', 'acquisition_channel', 'user_segment', 'app_version',
        'age_group', 'user_type'
    ]
    
    # Only drop columns that exist
    columns_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
    df_processed = df_processed.drop(columns=columns_to_drop)
    
    # Ensure all column names are lowercase
    df_processed.columns = df_processed.columns.str.lower()
    
    # Handle feature alignment with the model's expected features
    missing_features = [f for f in feature_names if f not in df_processed.columns]
    if missing_features:
        for feat in missing_features:
            df_processed[feat] = 0
    
    # Get final features in the correct order
    X = df_processed[feature_names].copy()
    
    # Apply scaling to numeric features
    scaled_features = scaling_info.get("scaled_features", [])
    if scaled_features and scaler is not None:
        # Only scale features that exist in our data and were scaled during training
        scale_cols = [col for col in scaled_features if col in X.columns]
        
        try:
            # Create a copy of the scaled features
            X_scaled = X[scale_cols].copy()
            X_scaled_arr = scaler.transform(X_scaled)
            
            # Put the scaled data back into the dataframe
            for i, col in enumerate(scale_cols):
                X[col] = X_scaled_arr[:, i]
        except Exception as e:
            logger.error(f"Error during feature scaling: {str(e)}")
    
    return X

def make_single_prediction(feature_dict):
    """
    Make a prediction for a single record.
    
    Parameters:
    -----------
    feature_dict : dict
        Dictionary of feature values
    
    Returns:
    --------
    dict
        Prediction results with class and probability
    """
    # Load model and artifacts
    model, scaler, feature_names, scaling_info = get_active_model_artifacts()
    
    if model is None:
        return {
            "error": "Failed to load model",
            "success": False
        }
    
    try:
        # Convert to DataFrame
        df = create_feature_df(feature_dict)
        
        # Process features
        X = process_single_record(df, feature_names, scaling_info, scaler)
        
        # Make prediction
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0, 1])
        
        # Get top influencing features (if model supports it)
        feature_influences = {}
        if hasattr(model, 'feature_importances_'):
            # Get feature importance from the model
            importance = model.feature_importances_
            
            # Create a dictionary of feature importance
            features_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })
            
            # Get the values for the current prediction
            for feature in feature_names:
                if feature in X.columns:
                    features_df.loc[features_df['feature'] == feature, 'value'] = X[feature].values[0]
                else:
                    features_df.loc[features_df['feature'] == feature, 'value'] = 0
            
            # Calculate influence (importance * value)
            features_df['influence'] = features_df['importance'] * features_df['value']
            
            # Sort by absolute influence
            features_df['abs_influence'] = features_df['influence'].abs()
            features_df = features_df.sort_values('abs_influence', ascending=False)
            
            # Get top influencing features
            top_features = features_df.head(5).to_dict('records')
            feature_influences = {
                'top_features': top_features,
                'all_features': features_df.to_dict('records')
            }
        
        # Return prediction results
        return {
            "prediction": prediction,
            "probability": probability,
            "feature_influences": feature_influences,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return {
            "error": str(e),
            "success": False
        }

def save_prediction_to_mongodb(feature_dict, prediction_result):
    """
    Save the prediction to MongoDB.
    
    Parameters:
    -----------
    feature_dict : dict
        The original feature values
    prediction_result : dict
        The prediction results
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        db, client = connect_to_mongodb()
        
        if db is None:
            return False
        
        # Add prediction result to features
        record = {**feature_dict}
        record['purchase_24h_prediction'] = prediction_result['prediction']
        record['purchase_24h_probability'] = prediction_result['probability']
        record['prediction_timestamp'] = datetime.now()
        record['prediction_source'] = 'single_prediction_ui'
        
        # Save to predicted_results collection
        db.predicted_results.insert_one(record)
        client.close()
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving prediction: {str(e)}")
        if 'client' in locals():
            client.close()
        return False

# Example usage (for testing)
if __name__ == "__main__":
    # Example feature dictionary
    sample_features = {
        "user_id": "test_user_1",
        "platform": "iOS",
        "age": 35,
        "session_count": 5,
        "total_screens_viewed": 15,
        "used_search_feature": 1,
        "wrote_review": 0,
        "added_to_wishlist": 1,
        "screen_list": "ProductList,ProductDetail,ShoppingCart,Checkout",
        "region": "NorthAmerica",
        "acquisition_channel": "Organic",
        "user_segment": "Young Professional",
        "app_version": "3.2.1",
        "first_visit_date": "2025-05-01"
    }
    
    # Make prediction
    result = make_single_prediction(sample_features)
    print(result)