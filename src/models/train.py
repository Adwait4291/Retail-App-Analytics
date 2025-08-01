# src/train.py (MLflow Aliases & Promotion Logic Version)

import os
import pandas as pd
import numpy as np
import joblib
import logging
import json
import shutil
import git
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# --- 1. Centralized Configuration ---
class Config:
    # Environment and Paths
    IS_DOCKER = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER", "false").lower() == "true"
    MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models") if IS_DOCKER else "../models"

    # MLflow Settings
    MLFLOW_TRACKING_URI = "file:./mlruns"
    MLFLOW_EXPERIMENT_NAME = "Retail24_Training_Experiment"
    MLFLOW_REGISTERED_MODEL_NAME = "Retail24RandomForestClassifier"
    PRODUCTION_ALIAS = "production" # Using alias instead of stage

    # Model Training Parameters
    N_ESTIMATORS = 150
    MAX_DEPTH = 8
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('retail_ml_training')
load_dotenv()

# --- Database and Data Handling ---
def connect_to_mongodb():
    """Connect to MongoDB and return the database object."""
    try:
        client = MongoClient(f"mongodb+srv://{os.getenv('MONGODB_USERNAME')}:{os.getenv('MONGODB_PASSWORD')}@{os.getenv('MONGODB_CLUSTER')}/")
        db = client[os.getenv("MONGODB_DATABASE")]
        logger.info("Successfully connected to MongoDB.")
        return db
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        raise

def get_all_processed_data(db):
    """Get all processed data from MongoDB."""
    logger.info("Retrieving all processed data...")
    cursor = db.processed_retail_data.find({}, {'_id': 0})
    df = pd.DataFrame(list(cursor))
    if df.empty:
        logger.warning("No processed data found.")
        return None, None
    metadata = db.processing_metadata.find_one({"domain": "retail"}, sort=[("processed_at", -1)])
    latest_version = metadata["processing_version"] if metadata else "unknown"
    logger.info(f"Retrieved {len(df)} records. Latest data version: {latest_version}")
    return df, latest_version

def prepare_data_for_training(df):
    """Prepares raw data for model training."""
    logger.info("Preparing data for model training...")
    df = df.copy()
    if 'user_id' in df.columns: df = df.drop('user_id', axis=1)
    if 'record_hash' in df.columns: df = df.drop('record_hash', axis=1)
    timestamp_cols = [col for col in df.columns if df[col].dtype == 'object' and any(isinstance(val, str) and ('/' in val or '-' in val or ':' in val) for val in df[col].dropna().head(5))]
    if timestamp_cols: df = df.drop(columns=timestamp_cols)
    object_cols = df.select_dtypes(include=['object']).columns
    if len(object_cols) > 0: df = df.drop(columns=object_cols)
    if 'purchase_24h' not in df.columns: raise ValueError("Target column 'purchase_24h' not found.")

    X = df.drop('purchase_24h', axis=1)
    y = df['purchase_24h']
    num_features = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y)
    scaler = StandardScaler().fit(X_train[num_features])
    X_train[num_features] = scaler.transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])
    return X_train, X_test, y_train, y_test, scaler, num_features

# --- Model Lifecycle ---
def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """Trains a new model and evaluates its performance."""
    logger.info("Training and evaluating model...")
    model = RandomForestClassifier(n_estimators=Config.N_ESTIMATORS, max_depth=Config.MAX_DEPTH, random_state=Config.RANDOM_STATE, n_jobs=-1, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Ensure all metrics are standard Python floats
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
    }
    logger.info(f"New model evaluation complete. F1 Score: {metrics['f1_score']:.4f}")
    return model, metrics

def get_production_model_via_alias(client, model_name):
    """Gets the current production model and its F1 score using its alias."""
    logger.info(f"Searching for model with alias '{Config.PRODUCTION_ALIAS}' for '{model_name}'...")
    try:
        # Get model version by alias
        prod_version = client.get_model_version_by_alias(model_name, Config.PRODUCTION_ALIAS)
        run = client.get_run(prod_version.run_id)
        f1_score = run.data.metrics.get("f1_score", 0.0)
        logger.info(f"Found production model: Version {prod_version.version} with F1 score: {f1_score:.4f}")
        return prod_version, f1_score
    except MlflowException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            logger.info(f"No model currently has the '{Config.PRODUCTION_ALIAS}' alias.")
        else:
            logger.warning(f"Could not retrieve production model: {e}")
        return None, 0.0 # No best model, so any new model is better

def update_mongodb_record(db, run_id, model_version, metrics, data_version):
    """Deactivates old model and saves new active model metadata to MongoDB."""
    logger.info(f"Updating MongoDB: Deactivating old models and setting '{model_version}' as active.")
    collection = db.model_metadata
    collection.update_many({"status": "active"}, {"$set": {"status": "inactive", "deactivated_at": datetime.now()}})
    collection.insert_one({
        "model_version": model_version, "mlflow_run_id": run_id,
        "trained_at": datetime.now(), "data_version": data_version,
        "metrics": metrics, "status": "active"
    })
    logger.info(f"MongoDB updated successfully. Active model is now '{model_version}'.")

# --- Main Orchestration Logic ---
def main():
    """Main function to orchestrate the model training and promotion pipeline."""
    logger.info("Starting model training pipeline with unconditional training logic.")
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)

    db = connect_to_mongodb()
    df, latest_data_version = get_all_processed_data(db)
    if df is None: return

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run started with ID: {run_id}")
        
        try:
            X_train, X_test, y_train, y_test, scaler, num_features = prepare_data_for_training(df)
            model, new_metrics = train_and_evaluate_model(X_train, y_train, X_test, y_test)
            new_f1_score = new_metrics.get('f1_score', 0.0)

            mlflow.log_params({"n_estimators": Config.N_ESTIMATORS, "max_depth": Config.MAX_DEPTH, "data_version": latest_data_version})
            mlflow.log_metrics(new_metrics)

            client = MlflowClient()
            _, best_f1_score = get_production_model_via_alias(client, Config.MLFLOW_REGISTERED_MODEL_NAME)
            
            if new_f1_score > best_f1_score:
                logger.info(f"PROMOTION: New model F1 ({new_f1_score:.4f}) is better than production F1 ({best_f1_score:.4f}).")
                
                signature = infer_signature(X_train, model.predict(X_train))
                model_info = mlflow.sklearn.log_model(model, "model", signature=signature)
                joblib.dump(scaler, "scaler.pkl")
                mlflow.log_artifact("scaler.pkl", "auxiliary")

                registered_model = mlflow.register_model(model_info.model_uri, Config.MLFLOW_REGISTERED_MODEL_NAME)
                new_version = registered_model.version
                
                logger.info(f"Setting alias '{Config.PRODUCTION_ALIAS}' on new model version {new_version}.")
                client.set_registered_model_alias(Config.MLFLOW_REGISTERED_MODEL_NAME, Config.PRODUCTION_ALIAS, new_version)
                
                model_version_name = f"rf_v{new_version}_{datetime.now().strftime('%Y%m%d')}"
                update_mongodb_record(db, run_id, model_version_name, new_metrics, latest_data_version)

            else:
                logger.info(f"NO PROMOTION: New model F1 ({new_f1_score:.4f}) is not better than production F1 ({best_f1_score:.4f}).")
                mlflow.sklearn.log_model(model, "model_not_promoted")
                logger.info("Logged experiment without promoting model.")

        except Exception as e:
            logger.critical(f"An unrecoverable error occurred in the training pipeline: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    main()