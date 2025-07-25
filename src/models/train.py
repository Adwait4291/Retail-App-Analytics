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
MIN_NEW_RECORDS_THRESHOLD = 200  # Minimum new records needed to retrain
MIN_PERFORMANCE_IMPROVEMENT = 0.02  # Minimum improvement in F1 score to accept new model

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

def get_all_processed_data():
   """Get all processed data from MongoDB regardless of version."""
   logger.info("Retrieving all processed data...")
   
   db, client = connect_to_mongodb()
   
   try:
       # Get collection
       processed_collection = db.processed_retail_data
       
       # Fetch all processed data
       cursor = processed_collection.find({}, {'_id': 0})
       df = pd.DataFrame(list(cursor))
       
       record_count = len(df)
       logger.info(f"Retrieved {record_count} total processed records across all versions")
       
       # Get latest processing version for reference
       metadata_collection = db.processing_metadata
       latest_metadata = metadata_collection.find_one(
           {"domain": "retail"}, 
           sort=[("processed_at", -1)]
       )
       
       latest_version = latest_metadata["processing_version"] if latest_metadata else "unknown"
       logger.info(f"Latest processing version: {latest_version}")
       
       # Close connection
       client.close()
       
       if record_count == 0:
           logger.warning("No processed data found")
           return None, None
           
       return df, latest_version
   
   except Exception as e:
       logger.error(f"Error retrieving processed data: {str(e)}")
       client.close()
       raise

def get_latest_model_metadata():
   """Get metadata for the latest trained model."""
   logger.info("Checking for existing model metadata...")
   
   try:
       db, client = connect_to_mongodb()
       model_metadata_collection = db.model_metadata
       
       # Get the latest model metadata
       latest_model = model_metadata_collection.find_one(
           {"status": "active"}, 
           sort=[("trained_at", -1)]
       )
       
       client.close()
       
       if not latest_model:
           logger.info("No existing model found")
           return None
           
       logger.info(f"Found existing model: {latest_model['model_version']}")
       return latest_model
       
   except Exception as e:
       logger.error(f"Error retrieving model metadata: {str(e)}")
       client.close()
       return None

def should_train_model(latest_data_version, latest_model_metadata):
   """Determine if a new model should be trained based on data updates."""
   logger.info("Checking if model training is needed...")
   
   # If no model exists, we should definitely train
   if latest_model_metadata is None:
       logger.info("No existing model found. Training new model.")
       return True, "Initial model training"
   
   # Check if data has been updated since last model training
   if latest_model_metadata['data_version'] != latest_data_version:
       logger.info(f"Data version mismatch - Model: {latest_model_metadata['data_version']}, Data: {latest_data_version}")
       
       # Check if we have enough new data with additional debugging
       try:
           db, client = connect_to_mongodb()
           
           # Debug query parameters
           trained_at = latest_model_metadata['trained_at']
           logger.info(f"Looking for records processed after: {trained_at}")
           
           # Find and count all processing metadata entries after model training
           metadata_collection = db.processing_metadata
           cursor = metadata_collection.find({"processed_at": {"$gt": trained_at}, "domain": "retail"})
           
           # Log each found entry for debugging
           entries = list(cursor)
           logger.info(f"Found {len(entries)} processing entries after last model training")
           
           for entry in entries:
               logger.info(f"Processing entry: {entry.get('processing_version')}, Records: {entry.get('record_count')}")
           
           total_new_records = sum(meta.get('record_count', 0) for meta in entries)
           logger.info(f"Total new records counted: {total_new_records} (threshold: {MIN_NEW_RECORDS_THRESHOLD})")
           
           client.close()
           
           if total_new_records >= MIN_NEW_RECORDS_THRESHOLD:
               logger.info(f"Found {total_new_records} new records since last training.")
               return True, f"Data updated with {total_new_records} new records"
           else:
               logger.info(f"Only {total_new_records} new records. Not enough for retraining.")
               return False, f"Insufficient new data: {total_new_records} records (threshold: {MIN_NEW_RECORDS_THRESHOLD})"
               
       except Exception as e:
           logger.error(f"Error checking for new data: {str(e)}")
           return True, "Error checking data updates, proceeding with training"
   
   # Check time since last training
   days_since_training = (datetime.now() - latest_model_metadata['trained_at']).days
   if days_since_training > 30:  # Retrain monthly at minimum
       logger.info(f"Model is {days_since_training} days old. Retraining for freshness.")
       return True, f"Model age: {days_since_training} days"
       
   logger.info("No significant changes detected. Using existing model.")
   return False, "No significant changes to data"

def prepare_data_for_training(df):
   """Prepare data for model training."""
   logger.info("Preparing data for model training...")
   
   # Drop user_id column
   if 'user_id' in df.columns:
       logger.info("Dropping user_id column")
       df = df.drop('user_id', axis=1)
   
   # Drop record_hash column if present (defensive check)
   if 'record_hash' in df.columns:
       logger.info("Dropping record_hash column")
       df = df.drop('record_hash', axis=1)
   
   # Drop timestamp and date columns that can't be used as features
   timestamp_columns = []
   for col in df.columns:
       # Check if column contains timestamp-like strings by examining the first non-null value
       if df[col].dtype == 'object':
           sample_values = df[col].dropna().head(5).tolist()
           if any(isinstance(val, str) and ('/' in val or '-' in val or ':' in val) for val in sample_values):
               timestamp_columns.append(col)
   
   if timestamp_columns:
       logger.info(f"Dropping timestamp columns: {timestamp_columns}")
       df = df.drop(columns=timestamp_columns)
   
   # Check for and handle any remaining non-numeric columns
   object_columns = df.select_dtypes(include=['object']).columns
   if len(object_columns) > 0:
       logger.warning(f"Found non-numeric columns that need to be handled: {object_columns}")
       # Drop these columns as they can't be directly used in the model
       df = df.drop(columns=object_columns)
       logger.info(f"Dropped non-numeric columns: {object_columns}")
   
   # Check for and handle any remaining null values
   null_counts = df.isnull().sum()
   if null_counts.sum() > 0:
       logger.warning(f"Found null values: {null_counts[null_counts > 0]}")
       # For numeric columns, fill with median
       numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
       for col in numeric_cols:
           if df[col].isnull().sum() > 0:
               df[col] = df[col].fillna(df[col].median())
               logger.info(f"Filled nulls in {col} with median")
       
       # For boolean columns, fill with False
       bool_cols = df.select_dtypes(include=['bool']).columns
       for col in bool_cols:
           if df[col].isnull().sum() > 0:
               df[col] = df[col].fillna(False)
               logger.info(f"Filled nulls in {col} with False")
   
   # Check if target column exists
   if 'purchase_24h' not in df.columns:
       logger.error("Target column 'purchase_24h' not found in dataset")
       raise ValueError("Target column 'purchase_24h' not found in dataset")
   
   # Separate features and target
   X = df.drop('purchase_24h', axis=1)
   y = df['purchase_24h']
   logger.info(f"Target distribution: {y.value_counts(normalize=True).to_dict()}")
   
   # Split data
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )
   logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
   
   # Scale numerical features
   logger.info("Applying StandardScaler to numerical features")
   num_features = X.select_dtypes(include=['int64', 'float64']).columns
   scaler = StandardScaler()
   X_train[num_features] = scaler.fit_transform(X_train[num_features])
   X_test[num_features] = scaler.transform(X_test[num_features])
   
   return X_train, X_test, y_train, y_test, scaler, list(X.columns)

def train_random_forest(X_train, y_train):
   """Train a Random Forest model."""
   logger.info("Training Random Forest model...")
   
   # Initialize and train the model
   model = RandomForestClassifier(
       n_estimators=100,
       max_depth=None,
       min_samples_split=2,
       min_samples_leaf=1,
       max_features='sqrt',
       random_state=42,
       n_jobs=-1,
       class_weight='balanced'
   )
   
   model.fit(X_train, y_train)
   logger.info("Model training completed")
   
   return model

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
   """Evaluate the model and return performance metrics."""
   logger.info("Evaluating model performance...")
   
   # Make predictions
   y_pred = model.predict(X_test)
   y_pred_proba = model.predict_proba(X_test)[:, 1]
   
   # Calculate main metrics
   metrics = {
       'accuracy': float(accuracy_score(y_test, y_pred)),
       'precision': float(precision_score(y_test, y_pred)),
       'recall': float(recall_score(y_test, y_pred)),
       'f1_score': float(f1_score(y_test, y_pred)),
       'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
   }
   
   logger.info(f"Model performance metrics:")
   for metric, value in metrics.items():
       logger.info(f"  {metric}: {value:.4f}")
   
   # Generate classification report
   class_report = classification_report(y_test, y_pred, output_dict=True)
   logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
   
   # Generate confusion matrix
   conf_matrix = confusion_matrix(y_test, y_pred).tolist()
   logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
   
   # Cross-validation scores
   logger.info("Performing cross-validation...")
   cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
   logger.info(f"5-fold CV F1 scores: {cv_scores}")
   logger.info(f"Mean CV F1 score: {cv_scores.mean():.4f}")
   
   # Calculate feature importance
   logger.info("Calculating feature importance...")
   feature_importance = model.feature_importances_
   
   # Create a DataFrame of feature importances
   importance_df = pd.DataFrame({
       'feature': feature_names,
       'importance': feature_importance
   }).sort_values('importance', ascending=False)
   
   # Log top 10 features
   logger.info("Top 10 most important features:")
   for idx, row in importance_df.head(10).iterrows():
       logger.info(f"  {row['feature']}: {row['importance']:.4f}")
   
   # Assemble detailed metrics dictionary
   detailed_metrics = {
       'summary_metrics': metrics,
       'classification_report': class_report,
       'confusion_matrix': conf_matrix,
       'cross_validation': {
           'scores': cv_scores.tolist(),
           'mean_f1': float(cv_scores.mean())
       },
       'feature_importance': importance_df.to_dict(orient='records')
   }
   
   return metrics, detailed_metrics, importance_df.to_dict(orient='records')

def compare_with_existing_model(new_metrics, latest_model_metadata):
   """Compare new model performance with existing model. Always return True to save the new model,
   but log a warning if performance is worse."""
   if not latest_model_metadata:
       logger.info("No existing model to compare with")
       return True, "Initial model"
   
   existing_metrics = latest_model_metadata.get('metrics', {})
   existing_f1 = existing_metrics.get('f1_score', 0)
   new_f1 = new_metrics.get('f1_score', 0)
   
   improvement = new_f1 - existing_f1
   
   logger.info(f"Model comparison - Existing F1: {existing_f1:.4f}, New F1: {new_f1:.4f}")
   logger.info(f"Improvement: {improvement:.4f} (threshold: {MIN_PERFORMANCE_IMPROVEMENT})")
   
   if improvement < 0:
       # Add a highly visible warning when performance is worse
       warning_message = f"WARNING: NEW MODEL PERFORMANCE IS SIGNIFICANTLY WORSE THAN EXISTING MODEL (F1 change: {improvement:.4f})"
       logger.warning("!" * 80)
       logger.warning(warning_message)
       logger.warning("!" * 80)
       return True, f"Saving despite lower F1 score: {improvement:.4f}"
   elif improvement >= MIN_PERFORMANCE_IMPROVEMENT:
       return True, f"F1 improvement: +{improvement:.4f}"
   else:
       return True, f"Slight F1 improvement: +{improvement:.4f}"

def save_model_metrics(model_version, detailed_metrics):
   """Save detailed model metrics to dedicated collection."""
   logger.info(f"Saving detailed metrics for model: {model_version}")
   
   try:
       db, client = connect_to_mongodb()
       metrics_collection = db.model_metrics
       
       # Create metrics document
       metrics_doc = {
           "model_version": model_version,
           "recorded_at": datetime.now(),
           "detailed_metrics": detailed_metrics
       }
       
       # Insert metrics
       result = metrics_collection.insert_one(metrics_doc)
       logger.info(f"Saved detailed metrics with ID: {result.inserted_id}")
       
       client.close()
       return True
       
   except Exception as e:
       logger.error(f"Error saving model metrics: {str(e)}")
       if 'client' in locals():
           client.close()
       return False

def save_model(model, scaler, model_version, metrics, detailed_metrics, feature_importance, data_version, feature_names):
   """Save model, scaler, and metadata."""
   logger.info(f"Saving model version: {model_version}")
   
   # Create model directory if it doesn't exist
   if not os.path.exists(MODEL_DIR):
       os.makedirs(MODEL_DIR)
       logger.info(f"Created directory: {MODEL_DIR}")
   
   # Save model and scaler
   model_path = os.path.join(MODEL_DIR, f"rf_model_{model_version}.pkl")
   scaler_path = os.path.join(MODEL_DIR, f"scaler_{model_version}.pkl")
   
   joblib.dump(model, model_path)
   joblib.dump(scaler, scaler_path)
   logger.info(f"Model saved to {model_path}")
   logger.info(f"Scaler saved to {scaler_path}")
   
   # Save feature names
   feature_names_path = os.path.join(MODEL_DIR, f"feature_names_{model_version}.json")
   with open(feature_names_path, 'w') as f:
       json.dump(feature_names, f)
   logger.info(f"Feature names saved to {feature_names_path}")
   
   # Save model metadata to MongoDB
   try:
       db, client = connect_to_mongodb()
       model_metadata_collection = db.model_metadata
       
       # Update existing active models to inactive
       model_metadata_collection.update_many(
           {"status": "active"},
           {"$set": {"status": "inactive"}}
       )
       
       # Create metadata
       metadata = {
           "model_version": model_version,
           "model_type": "RandomForest",
           "trained_at": datetime.now(),
           "data_version": data_version,
           "metrics": metrics,  # Store basic metrics for quick access
           "model_path": model_path,
           "scaler_path": scaler_path,
           "feature_names_path": feature_names_path,
           "status": "active"
       }
       
       # Insert new metadata
       model_metadata_collection.insert_one(metadata)
       logger.info("Model metadata saved to MongoDB")
       
       # Save detailed metrics to separate collection
       save_model_metrics(model_version, detailed_metrics)
       
       client.close()
       
   except Exception as e:
       logger.error(f"Error saving model metadata: {str(e)}")
       if 'client' in locals():
           client.close()

def main():
   """Main function to train and evaluate a Random Forest model."""
   logger.info("Starting model training process")
   
   try:
       # Get all processed data instead of just the latest version
       df, latest_data_version = get_all_processed_data()
       if df is None:
           logger.error("No processed data available. Exiting.")
           return
       
       # Get latest model metadata
       latest_model_metadata = get_latest_model_metadata()
       
       # Check if we should train a new model
       should_train, reason = should_train_model(latest_data_version, latest_model_metadata)
       logger.info(f"Training decision: {should_train} - {reason}")
       
       if not should_train:
           logger.info("Using existing model. Training skipped.")
           return
       
       # Prepare data for training
       X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data_for_training(df)
       
       # Train the model
       model = train_random_forest(X_train, y_train)
       
       # Evaluate the model (now with more detailed metrics)
       metrics, detailed_metrics, feature_importance = evaluate_model(
           model, X_train, X_test, y_train, y_test, feature_names
       )
       
       # Compare with existing model (will always return True with the modified function)
       is_better, comparison_reason = compare_with_existing_model(metrics, latest_model_metadata)
       logger.info(f"Model comparison result: {is_better} - {comparison_reason}")
       
       # Generate a unique model version
       data_hash = hashlib.md5(str(metrics).encode()).hexdigest()[:8]
       model_version = f"rf_{CURRENT_TIMESTAMP}_{data_hash}"
       
       # Save the model and metadata
       save_model(
           model=model,
           scaler=scaler, 
           model_version=model_version,
           metrics=metrics,
           detailed_metrics=detailed_metrics,
           feature_importance=feature_importance,
           data_version=latest_data_version,
           feature_names=feature_names
       )
       logger.info(f"Model training complete. New model version: {model_version}")
       
   except Exception as e:
       logger.error(f"Error in model training process: {str(e)}", exc_info=True)
       raise  # Re-raise the exception so the pipeline knows there was an error

if __name__ == "__main__":
   main()