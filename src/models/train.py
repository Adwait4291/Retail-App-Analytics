# train.py
import os
import pandas as pd
import numpy as np
import joblib
import logging
import os
import hashlib
import json
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                          roc_auc_score, classification_report, confusion_matrix)

# Set up logging
logging.basicConfig(level=logging.INFO,
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('retail_ml_training')

# Constants
MIN_PERFORMANCE_IMPROVEMENT = 0.01  # Minimum improvement in F1 score to accept new model

# Dynamically determine if running in Docker or locally
is_docker = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER", "false").lower() == "true"

# Set MODEL_DIR based on environment
if is_docker:
    MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
    logger.info("Running in Docker environment, using path: %s", MODEL_DIR)
else:
    MODEL_DIR = "../models"
    logger.info("Running in local environment, using path: %s", MODEL_DIR)

CURRENT_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def connect_to_mongodb():
   """Connect to MongoDB and return database connection."""
   logger.info("Connecting to MongoDB database...")
   load_dotenv()
   username = os.getenv("MONGODB_USERNAME")
   password = os.getenv("MONGODB_PASSWORD")
   cluster = os.getenv("MONGODB_CLUSTER")
   database = os.getenv("MONGODB_DATABASE")
   connection_string = f"mongodb+srv://{username}:{password}@{cluster}/"
   try:
       client = MongoClient(connection_string)
       db = client.get_database(database)
       logger.info("Successfully connected to MongoDB")
       return db, client
   except Exception as e:
       logger.error(f"Error connecting to MongoDB: {str(e)}")
       raise

def get_all_processed_data():
   """Get all processed data from MongoDB regardless of version."""
   logger.info("Retrieving all processed data...")
   db, client = connect_to_mongodb()
   try:
       processed_collection = db.processed_retail_data
       cursor = processed_collection.find({}, {'_id': 0})
       df = pd.DataFrame(list(cursor))
       record_count = len(df)
       logger.info(f"Retrieved {record_count} total processed records across all versions")
       
       metadata_collection = db.processing_metadata
       latest_metadata = metadata_collection.find_one(
           {"domain": "retail"},
           sort=[("processed_at", -1)]
       )
       latest_version = latest_metadata["processing_version"] if latest_metadata else "unknown"
       logger.info(f"Latest processing version: {latest_version}")
       
       client.close()
       if record_count == 0:
           logger.warning("No processed data found")
           return None, None
       return df, latest_version
   except Exception as e:
       logger.error(f"Error retrieving processed data: {str(e)}")
       client.close()
       raise

def get_active_model_metadata():
   """Get metadata for the currently active model."""
   logger.info("Checking for an active model...")
   try:
       db, client = connect_to_mongodb()
       model_metadata_collection = db.model_metadata
       active_model = model_metadata_collection.find_one(
           {"status": "active"},
           sort=[("trained_at", -1)]
       )
       client.close()
       if not active_model:
           logger.info("No active model found")
           return None
       logger.info(f"Found active model: {active_model['model_version']}")
       return active_model
   except Exception as e:
       logger.error(f"Error retrieving model metadata: {str(e)}")
       client.close()
       return None

def prepare_data_for_training(df):
   """Prepare data for model training."""
   logger.info("Preparing data for model training...")
   if 'user_id' in df.columns:
       df = df.drop('user_id', axis=1)
   if 'record_hash' in df.columns:
       df = df.drop('record_hash', axis=1)
   
   timestamp_columns = [col for col in df.columns if 'timestamp' in col or 'date' in col]
   if timestamp_columns:
       logger.info(f"Dropping timestamp-like columns: {timestamp_columns}")
       df = df.drop(columns=timestamp_columns)

   object_columns = df.select_dtypes(include=['object']).columns
   if len(object_columns) > 0:
       df = df.drop(columns=object_columns)
       logger.info(f"Dropped non-numeric columns: {object_columns}")

   if df.isnull().sum().sum() > 0:
       numeric_cols = df.select_dtypes(include=np.number).columns
       for col in numeric_cols:
           if df[col].isnull().any():
               df[col] = df[col].fillna(df[col].median())
   
   if 'purchase_24h' not in df.columns:
       raise ValueError("Target column 'purchase_24h' not found in dataset")

   X = df.drop('purchase_24h', axis=1)
   y = df['purchase_24h']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
   
   num_features = X.select_dtypes(include=np.number).columns.tolist()
   scaler = StandardScaler().fit(X_train[num_features])
   
   X_train.loc[:, num_features] = scaler.transform(X_train[num_features])
   X_test.loc[:, num_features] = scaler.transform(X_test[num_features])
   
   scaling_info = {
       "scaled_features": num_features,
       "all_features": X.columns.tolist()
   }
   return X_train, X_test, y_train, y_test, scaler, scaling_info

def train_random_forest(X_train, y_train):
   """Train a Random Forest model."""
   logger.info("Training Random Forest model...")
   model = RandomForestClassifier(
       n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'
   )
   model.fit(X_train, y_train)
   logger.info("Model training completed")
   return model

def evaluate_model(model, X_test, y_test, feature_names):
   """Evaluate the model and return performance metrics."""
   logger.info("Evaluating model performance...")
   y_pred = model.predict(X_test)
   y_pred_proba = model.predict_proba(X_test)[:, 1]
   
   metrics = {
       'accuracy': float(accuracy_score(y_test, y_pred)),
       'precision': float(precision_score(y_test, y_pred)),
       'recall': float(recall_score(y_test, y_pred)),
       'f1_score': float(f1_score(y_test, y_pred)),
       'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
   }
   for metric, value in metrics.items():
       logger.info(f"  {metric}: {value:.4f}")
   
   importance_df = pd.DataFrame({
       'feature': feature_names,
       'importance': model.feature_importances_
   }).sort_values('importance', ascending=False)

   detailed_metrics = {
       'summary_metrics': metrics,
       'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
       'feature_importance': importance_df.to_dict(orient='records')
   }
   return metrics, detailed_metrics

def compare_and_decide_save(new_metrics, active_model_metadata):
    """Compare new model with the active one and decide whether to save."""
    if not active_model_metadata:
        logger.info("No active model to compare against. Saving new model.")
        return True, "Initial model"

    existing_f1 = active_model_metadata.get('metrics', {}).get('f1_score', 0)
    new_f1 = new_metrics.get('f1_score', 0)
    improvement = new_f1 - existing_f1

    logger.info(f"Model Comparison - Active F1: {existing_f1:.4f}, New F1: {new_f1:.4f}")

    if improvement >= MIN_PERFORMANCE_IMPROVEMENT:
        logger.info(f"New model is better by {improvement:.4f}. It will be saved.")
        return True, f"F1 improvement: +{improvement:.4f}"
    else:
        logger.warning(f"New model F1 score did not meet the improvement threshold of {MIN_PERFORMANCE_IMPROVEMENT}.")
        logger.warning("The new model will NOT be saved. Keeping the existing active model.")
        return False, f"Insufficient F1 improvement: +{improvement:.4f}"

def save_model_artifacts(model, scaler, model_version, metrics, detailed_metrics, data_version, scaling_info):
   """Save model, scaler, and metadata using filenames."""
   logger.info(f"Saving new active model version: {model_version}")
   os.makedirs(MODEL_DIR, exist_ok=True)
   
   # --- NECESSARY CHANGE: Define filenames ---
   model_filename = f"rf_model_{model_version}.pkl"
   scaler_filename = f"scaler_{model_version}.pkl"
   feature_names_filename = f"feature_names_{model_version}.json"
   scaling_info_filename = f"scaling_info_{model_version}.json"

   # --- Build full paths for saving to disk ---
   model_path = os.path.join(MODEL_DIR, model_filename)
   scaler_path = os.path.join(MODEL_DIR, scaler_filename)
   feature_names_path = os.path.join(MODEL_DIR, feature_names_filename)
   scaling_info_path = os.path.join(MODEL_DIR, scaling_info_filename)
   
   try:
       joblib.dump(model, model_path)
       joblib.dump(scaler, scaler_path)
       with open(feature_names_path, 'w') as f:
           json.dump(scaling_info["all_features"], f)
       with open(scaling_info_path, 'w') as f:
           json.dump(scaling_info, f)
   except Exception as e:
       logger.error(f"Error saving model files: {str(e)}", exc_info=True)
       raise

   try:
       db, client = connect_to_mongodb()
       model_metadata_collection = db.model_metadata
       metrics_collection = db.model_metrics
       
       model_metadata_collection.update_many(
           {"status": "active"}, {"$set": {"status": "inactive"}}
       )
       
       # --- NECESSARY CHANGE: Store FILENAMES in database, not paths ---
       metadata = {
           "model_version": model_version,
           "model_type": "RandomForestClassifier",
           "trained_at": datetime.now(),
           "data_version": data_version,
           "metrics": metrics,
           "model_filename": model_filename,
           "scaler_filename": scaler_filename,
           "feature_names_filename": feature_names_filename,
           "scaling_info_filename": scaling_info_filename,
           "status": "active"
       }
       model_metadata_collection.insert_one(metadata)
       
       metrics_doc = {
           "model_version": model_version,
           "recorded_at": datetime.now(),
           "detailed_metrics": detailed_metrics
       }
       metrics_collection.insert_one(metrics_doc)
       
       logger.info(f"Successfully saved and activated new model version: {model_version}")
       client.close()
   except Exception as e:
       logger.error(f"Error saving model metadata: {str(e)}")
       if 'client' in locals():
           client.close()
       raise

def main():
   """Main function to train and conditionally save a model."""
   logger.info("Starting model training pipeline...")
   try:
       df, latest_data_version = get_all_processed_data()
       if df is None:
           logger.error("No processed data available. Exiting training.")
           return

       active_model_metadata = get_active_model_metadata()

       X_train, X_test, y_train, y_test, scaler, scaling_info = prepare_data_for_training(df)
       model = train_random_forest(X_train, y_train)
       
       new_metrics, detailed_metrics = evaluate_model(model, X_test, y_test, scaling_info["all_features"])
       
       should_save, reason = compare_and_decide_save(new_metrics, active_model_metadata)
       logger.info(f"Decision: {'Save new model' if should_save else 'Keep existing model'}. Reason: {reason}")
       
       if should_save:
           data_hash = hashlib.md5(str(new_metrics).encode()).hexdigest()[:8]
           model_version = f"rf_{CURRENT_TIMESTAMP}_{data_hash}"
           save_model_artifacts(
               model=model,
               scaler=scaler,
               model_version=model_version,
               metrics=new_metrics,
               detailed_metrics=detailed_metrics,
               data_version=latest_data_version,
               scaling_info=scaling_info
           )
       
   except Exception as e:
       logger.critical(f"A critical error occurred in the training pipeline: {str(e)}", exc_info=True)
       raise

if __name__ == "__main__":
   main()