# predict.py
import os
import pandas as pd
import numpy as np
import logging
import joblib
import json
import hashlib
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('retail_prediction')

# Constants
CURRENT_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BATCH_SIZE = 1000  # Process data in batches to prevent memory issues

# Dynamically determine if running in Docker or locally
is_docker = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER", "false").lower() == "true"

# Set MODEL_DIR based on environment
if is_docker:
    MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
    logger.info("Running in Docker environment, using path: %s", MODEL_DIR)
else:
    MODEL_DIR = "../models"
    logger.info("Running in local environment, using path: %s", MODEL_DIR)

def connect_to_mongodb():
    """Connect to MongoDB and return database connection."""
    logger.info("Connecting to MongoDB database...")
    
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
        logger.info("Successfully connected to MongoDB")
        return db, client
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        raise

def get_latest_model_metadata():
    """Get metadata for the latest trained model."""
    logger.info("Retrieving latest model metadata...")
    
    try:
        db, client = connect_to_mongodb()
        model_metadata_collection = db.model_metadata
        
        # Get the latest active model metadata
        latest_model = model_metadata_collection.find_one(
            {"status": "active"}, 
            sort=[("trained_at", -1)]
        )
        
        client.close()
        
        if not latest_model:
            logger.error("No active model found in database")
            return None
            
        logger.info(f"Found active model: {latest_model['model_version']}")
        return latest_model
        
    except Exception as e:
        logger.error(f"Error retrieving model metadata: {str(e)}")
        if 'client' in locals():
            client.close()
        return None

def load_model_artifacts(model_metadata):
    """Load model, scaler, and feature info from disk."""
    logger.info(f"Loading model artifacts for version: {model_metadata['model_version']}")
    
    try:
        # Load model
        logger.info(f"Loading model from {model_metadata['model_path']}")
        model = joblib.load(model_metadata['model_path'])
        
        # Load scaler
        logger.info(f"Loading scaler from {model_metadata['scaler_path']}")
        scaler = joblib.load(model_metadata['scaler_path'])
        
        # Load feature names
        logger.info(f"Loading feature names from {model_metadata['feature_names_path']}")
        with open(model_metadata['feature_names_path'], 'r') as f:
            feature_names = json.load(f)
        
        # Load scaling info
        logger.info(f"Loading scaling info from {model_metadata['scaling_info_path']}")
        with open(model_metadata['scaling_info_path'], 'r') as f:
            scaling_info = json.load(f)
        
        logger.info(f"Successfully loaded model artifacts")
        
        return model, scaler, feature_names, scaling_info
        
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}", exc_info=True)
        raise

def fetch_prediction_data(batch_size=BATCH_SIZE):
    """Fetch data from prediction_data collection that needs predictions."""
    logger.info("Fetching data for prediction...")
    
    try:
        db, client = connect_to_mongodb()
        prediction_collection = db.prediction_data
        
        # Query for records that haven't been predicted yet
        # This assumes you have a 'predicted' field to track status
        query = {"predicted": {"$ne": True}}
        
        # Count total records to process
        total_records = prediction_collection.count_documents(query)
        logger.info(f"Found {total_records} records requiring prediction")
        
        if total_records == 0:
            logger.info("No data to predict. Exiting.")
            client.close()
            return None, 0
        
        # Initialize counter
        processed_count = 0
        
        # Use aggregation to batch process
        df_list = []
        
        # Process in batches to avoid memory issues
        for batch_start in range(0, total_records, batch_size):
            # Get full documents directly
            batch_cursor = prediction_collection.find(
                query
            ).skip(batch_start).limit(batch_size)
            
            batch_data = list(batch_cursor)
            
            if not batch_data:
                continue
                
            # Convert to DataFrame
            batch_df = pd.DataFrame(batch_data)
            df_list.append(batch_df)
            
            processed_count += len(batch_df)
            logger.info(f"Fetched batch of {len(batch_df)} records. Total processed: {processed_count}/{total_records}")
            
            # If we've fetched all records or reached max size, stop
            if processed_count >= total_records:
                break
        
        client.close()
        
        # Combine all batches
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
            logger.info(f"Total data fetched: {len(df)} records")
            return df, total_records
        else:
            logger.warning("No data returned from query")
            return None, 0
            
    except Exception as e:
        logger.error(f"Error fetching prediction data: {str(e)}", exc_info=True)
        if 'client' in locals():
            client.close()
        return None, 0

def process_features(df, feature_names, scaling_info, scaler):
    """
    Process features to match the model's expected format.
    This mirrors the processing from train.py but for prediction data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data to process
    feature_names : list
        List of feature names expected by the model
    scaling_info : dict
        Dictionary with scaling information
    scaler : object
        Trained scaler object
        
    Returns:
    --------
    tuple (pandas.DataFrame, pandas.DataFrame)
        Processed features ready for prediction and original processed dataframe
    """
    logger.info("Processing features for prediction...")
    
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Drop _id field if it exists but keep it for reference
    original_ids = None
    if '_id' in df_processed.columns:
        original_ids = df_processed['_id'].copy()
        df_processed = df_processed.drop('_id', axis=1)
    
    # Time-based Processing - similar to process_retail_data in processing.py
    if 'first_visit_date' in df_processed.columns:
        df_processed['first_visit_date'] = pd.to_datetime(df_processed['first_visit_date'], errors='coerce')
        df_processed['hour'] = df_processed['first_visit_date'].dt.hour
        df_processed['dayofweek'] = df_processed['first_visit_date'].dt.dayofweek
        df_processed['is_weekend'] = df_processed['dayofweek'].isin([5,6]).astype(int)
    
    # Screen List Processing
    if 'screen_list' in df_processed.columns:
        # Add comma for consistent processing
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
    
    # Feature Engineering - same as in processing.py
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
    logger.info("Aligning features with model expectations...")
    
    # Keep only the features the model knows about, in the right order
    missing_features = [f for f in feature_names if f not in df_processed.columns]
    if missing_features:
        logger.warning(f"Missing {len(missing_features)} features required by the model. Adding with zero values.")
        for feat in missing_features:
            df_processed[feat] = 0
    
    # Get final features in the correct order
    X = df_processed[feature_names].copy()
    
    # Apply scaling to numeric features
    logger.info("Applying feature scaling...")
    scaled_features = scaling_info.get("scaled_features", [])
    if scaled_features and scaler is not None:
        # Only scale features that exist in our data and were scaled during training
        scale_cols = [col for col in scaled_features if col in X.columns]
        logger.info(f"Scaling {len(scale_cols)} numeric features")
        
        try:
            # Create a copy of the scaled features
            X_scaled = X[scale_cols].copy()
            X_scaled_arr = scaler.transform(X_scaled)
            
            # Put the scaled data back into the dataframe
            for i, col in enumerate(scale_cols):
                X[col] = X_scaled_arr[:, i]
                
            logger.info("Feature scaling applied successfully")
        except Exception as e:
            logger.error(f"Error during feature scaling: {str(e)}", exc_info=True)
            # Attempt a fallback approach if columns don't match exactly
            logger.warning("Attempting fallback scaling approach...")
            for col in scale_cols:
                if col in X.columns:
                    # Scale individual columns
                    try:
                        col_values = X[col].values.reshape(-1, 1)
                        col_scaled = scaler.transform(col_values)[:, 0]
                        X[col] = col_scaled
                    except:
                        logger.warning(f"Could not scale column {col}, using original values")
    
    # Add back the original IDs if they existed
    result_df = df_processed.copy()
    if original_ids is not None:
        result_df['_id'] = original_ids
    
    logger.info(f"Feature processing complete. Final shape: {X.shape}")
    
    return X, result_df

def make_predictions(model, X, original_df):
    """Generate predictions using the trained model."""
    logger.info(f"Generating predictions for {len(X)} records...")
    
    try:
        # Generate predictions and probabilities
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]  # Probability of class 1
        
        # Add predictions to original dataframe
        prediction_df = original_df.copy()
        prediction_df['purchase_24h_prediction'] = y_pred
        prediction_df['purchase_24h_probability'] = y_pred_proba
        
        # Generate prediction timestamp
        prediction_df['prediction_timestamp'] = datetime.now()
        
        # Calculate prediction summary
        pos_rate = y_pred.mean() 
        logger.info(f"Prediction complete. Positive rate: {pos_rate:.2%}")
        
        # High probability segments
        high_prob_threshold = 0.7
        high_prob_rate = (y_pred_proba >= high_prob_threshold).mean()
        logger.info(f"High probability rate (>={high_prob_threshold}): {high_prob_rate:.2%}")
        
        return prediction_df, {
            'positive_rate': float(pos_rate),
            'record_count': len(y_pred),
            'high_probability_rate': float(high_prob_rate)
        }
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}", exc_info=True)
        raise

def save_predictions(prediction_df, prediction_metadata, model_version, prediction_id):
    """Save predictions and metadata to MongoDB."""
    logger.info("Saving predictions to MongoDB...")
    
    try:
        db, client = connect_to_mongodb()
        prediction_result_collection = db.predicted_results
        prediction_metadata_collection = db.prediction_metadata
        prediction_data_collection = db.prediction_data
        
        # Save prediction results
        # Convert DataFrame to records
        prediction_records = prediction_df.to_dict('records')
        
        # Add prediction run ID to each record
        for record in prediction_records:
            record['prediction_id'] = prediction_id
            record['model_version'] = model_version
            
            # Convert ObjectId to string if present (for MongoDB storage)
            if '_id' in record and not isinstance(record['_id'], str):
                record['original_id'] = str(record['_id'])
                del record['_id']  # Remove the ObjectId to avoid errors
        
        # Insert in batches
        batch_size = 1000
        total_inserted = 0
        
        logger.info(f"Saving {len(prediction_records)} prediction results in batches of {batch_size}")
        
        for i in range(0, len(prediction_records), batch_size):
            batch = prediction_records[i:i+batch_size]
            result = prediction_result_collection.insert_many(batch)
            total_inserted += len(result.inserted_ids)
            logger.info(f"Saved batch {i//batch_size + 1}. Progress: {total_inserted}/{len(prediction_records)}")
        
        logger.info(f"Successfully saved {total_inserted} prediction results")
        
        # Save prediction metadata
        metadata = {
            'prediction_id': prediction_id,
            'model_version': model_version,
            'prediction_time': datetime.now(),
            'record_count': prediction_metadata['record_count'],
            'positive_rate': prediction_metadata['positive_rate'],
            'high_probability_rate': prediction_metadata['high_probability_rate']
        }
        
        prediction_metadata_collection.insert_one(metadata)
        logger.info(f"Saved prediction metadata with ID: {prediction_id}")
        
        # Update original prediction_data records to mark as predicted
        # Extract original IDs if they exist
        if '_id' in prediction_df.columns:
            original_ids = prediction_df['_id'].tolist()
            update_result = prediction_data_collection.update_many(
                {"_id": {"$in": original_ids}},
                {"$set": {"predicted": True, "prediction_id": prediction_id}}
            )
            logger.info(f"Updated {update_result.modified_count} records in prediction_data")
        
        client.close()
        return True
        
    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}", exc_info=True)
        if 'client' in locals():
            client.close()
        return False

def main():
    """Main function to orchestrate the prediction pipeline."""
    logger.info("Starting prediction pipeline...")
    
    try:
        # Generate a unique ID for this prediction run
        prediction_id = f"pred_{CURRENT_TIMESTAMP}_{hashlib.md5(CURRENT_TIMESTAMP.encode()).hexdigest()[:8]}"
        logger.info(f"Prediction run ID: {prediction_id}")
        
        # Step 1: Get the latest model metadata
        model_metadata = get_latest_model_metadata()
        if not model_metadata:
            logger.error("No active model found. Cannot proceed with predictions.")
            return False
        
        model_version = model_metadata['model_version']
        logger.info(f"Using model version: {model_version}")
        
        # Step 2: Load model artifacts
        model, scaler, feature_names, scaling_info = load_model_artifacts(model_metadata)
        
        # Step 3: Fetch data requiring predictions
        df, total_records = fetch_prediction_data()
        if df is None or len(df) == 0:
            logger.info("No data available for prediction. Exiting.")
            return True
        
        # Step 4: Process features - note that we now pass scaler to the function
        X, processed_df = process_features(df, feature_names, scaling_info, scaler)
        
        # Step 5: Make predictions
        prediction_df, prediction_metadata = make_predictions(model, X, df)
        
        # Step 6: Save predictions
        success = save_predictions(prediction_df, prediction_metadata, model_version, prediction_id)
        
        if success:
            logger.info(f"Prediction pipeline completed successfully for {total_records} records")
            return True
        else:
            logger.error("Failed to save predictions")
            return False
            
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    logger.info(f"Prediction pipeline finished with exit code: {exit_code}")
    exit(exit_code)