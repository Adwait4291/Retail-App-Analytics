# src/train.py (MLflow Integrated Version - Fixed)

import os
import pandas as pd
import numpy as np
import joblib
import logging
import hashlib
import json
import shutil
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                          roc_auc_score, classification_report, confusion_matrix)

# --- MLflow imports ---
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.exceptions import MlflowException
# --- End MLflow imports ---

# Set up logging
logging.basicConfig(level=logging.INFO,
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('retail_ml_training')

# Constants
MIN_NEW_RECORDS_THRESHOLD = 200
MIN_PERFORMANCE_IMPROVEMENT = 0.02

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
    
    try:
        db, client = connect_to_mongodb()
        
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
            logger.warning("No processed data found. Generating dummy data for training.")
            # Generate dummy data for demonstration if no data in MongoDB
            from sklearn.datasets import make_classification
            X_dummy, y_dummy = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=42)
            df_dummy = pd.DataFrame(X_dummy, columns=[f'feature_{i}' for i in range(X_dummy.shape[1])])
            df_dummy['purchase_24h'] = y_dummy
            return df_dummy, "dummy_data_v1"
        
        return df, latest_version
    
    except Exception as e:
        logger.error(f"Error retrieving processed data: {str(e)}")
        if 'client' in locals():
            client.close()
        # Fallback to dummy data on MongoDB error
        logger.warning("MongoDB data retrieval failed. Generating dummy data for training.")
        from sklearn.datasets import make_classification
        X_dummy, y_dummy = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=42)
        df_dummy = pd.DataFrame(X_dummy, columns=[f'feature_{i}' for i in range(X_dummy.shape[1])])
        df_dummy['purchase_24h'] = y_dummy
        return df_dummy, "dummy_data_v1_error_fallback"

def get_latest_model_metadata():
    """Get metadata for the latest trained model from MongoDB."""
    logger.info("Checking for existing model metadata in MongoDB...")
    
    try:
        db, client = connect_to_mongodb()
        model_metadata_collection = db.model_metadata
        
        # Get the latest active model metadata from MongoDB
        latest_model = model_metadata_collection.find_one(
            {"status": "active"},
            sort=[("trained_at", -1)]
        )
        
        client.close()
        
        if not latest_model:
            logger.info("No existing model found in MongoDB.")
            return None
        
        logger.info(f"Found existing model in MongoDB: {latest_model['model_version']}")
        return latest_model
    
    except Exception as e:
        logger.error(f"Error retrieving model metadata from MongoDB: {str(e)}")
        if 'client' in locals():
            client.close()
        return None

def should_train_model(latest_data_version, latest_model_metadata):
    """Determine if a new model should be trained based on data updates."""
    logger.info("Checking if model training is needed...")
    
    # Check if there's any registered model in MLflow
    try:
        client = mlflow.tracking.MlflowClient()
        try:
            registered_models = client.search_registered_models(filter_string="name='Retail24RandomForestClassifier'")
            mlflow_latest_model_version = None
            if registered_models and registered_models[0].latest_versions:
                mlflow_latest_model_version = registered_models[0].latest_versions[0]
                logger.info(f"Found latest model in MLflow Registry: Version {mlflow_latest_model_version.version}, Stage: {mlflow_latest_model_version.current_stage}")
        except Exception as e:
            logger.info(f"No registered model found in MLflow or error accessing registry: {e}")
            mlflow_latest_model_version = None
        
        # If no model in MLflow, or no active model in MongoDB, we should train
        if not mlflow_latest_model_version and not latest_model_metadata:
            logger.info("No existing model found in MLflow or MongoDB. Training new model.")
            return True, "Initial model training"
        
        # Use the MLflow model's run_id to get its data_version, if available
        mlflow_data_version = None
        if mlflow_latest_model_version and mlflow_latest_model_version.run_id:
            try:
                run_data = client.get_run(mlflow_latest_model_version.run_id).data
                mlflow_data_version = run_data.params.get('data_version')
            except Exception as e:
                logger.warning(f"Could not get MLflow run data: {e}")
        
        if mlflow_data_version and mlflow_data_version != latest_data_version:
            logger.info(f"Data version mismatch - MLflow Model Data: {mlflow_data_version}, Current Data: {latest_data_version}")
            return True, f"Data version changed from {mlflow_data_version} to {latest_data_version}"
    
    except Exception as e:
        logger.warning(f"Error checking MLflow registry: {e}. Proceeding with MongoDB-only check.")
    
    # Check time since last training (using MongoDB model if available)
    last_training_time = None
    if latest_model_metadata:
        last_training_time = latest_model_metadata.get('trained_at')
    
    if last_training_time:
        days_since_training = (datetime.now() - last_training_time).days
        if days_since_training > 30:  # Retrain monthly at minimum
            logger.info(f"Model is {days_since_training} days old. Retraining for freshness.")
            return True, f"Model age: {days_since_training} days"
    
    logger.info("No significant changes detected. Using existing model.")
    return False, "No significant changes to data"

def prepare_data_for_training(df):
    """Prepare data for model training."""
    logger.info("Preparing data for model training...")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Drop user_id column
    if 'user_id' in df.columns:
        logger.info("Dropping user_id column")
        df = df.drop('user_id', axis=1)
    
    # Drop record_hash column if present
    if 'record_hash' in df.columns:
        logger.info("Dropping record_hash column")
        df = df.drop('record_hash', axis=1)
    
    # Drop timestamp and date columns that can't be used as features
    timestamp_columns = []
    for col in df.columns:
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
    
    # Get numerical features
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    logger.info(f"Numerical features to scale: {len(num_features)}")
    
    # Create a scaler and fit it on numerical features only
    scaler = StandardScaler()
    if num_features:  # Only fit if there are numerical features
        scaler.fit(X_train[num_features])
        
        # Transform numerical features
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[num_features] = scaler.transform(X_train[num_features])
        X_test_scaled[num_features] = scaler.transform(X_test[num_features])
    else:
        # No numerical features to scale
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
    
    # Log feature counts for debugging
    logger.info(f"Total features for model: {X.shape[1]}")
    logger.info(f"Numerical features scaled: {len(num_features)}")
    
    # Create scaling info dict to store with model
    scaling_info = {
        "scaled_features": num_features,
        "all_features": X.columns.tolist()
    }
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, scaling_info

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """Train a Random Forest model."""
    logger.info("Training Random Forest model...")
    
    # Initialize and train the model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=random_state,
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
    
    # Debug information about lengths
    logger.info(f"Number of feature importances from model: {len(feature_importance)}")
    logger.info(f"Number of feature names provided: {len(feature_names)}")
    
    # Create a DataFrame of feature importances - Make sure arrays are the same length
    if len(feature_importance) != len(feature_names):
        logger.warning(f"Feature count mismatch! Model has {len(feature_importance)} feature importances "
                      f"but {len(feature_names)} feature names were provided.")
        
        # Use only the available feature names (truncate if needed)
        if len(feature_importance) < len(feature_names):
            logger.warning("Using only the first portion of provided feature names to match model.")
            feature_names_used = feature_names[:len(feature_importance)]
        else:
            # Create generic feature names for any extra features
            logger.warning("Creating generic feature names for extra features.")
            feature_names_used = list(feature_names)
            for i in range(len(feature_names), len(feature_importance)):
                feature_names_used.append(f"feature_{i}")
    else:
        feature_names_used = feature_names
    
    # Now create the DataFrame with matching arrays
    importance_df = pd.DataFrame({
        'feature': feature_names_used,
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
    """Compare new model performance with existing model."""
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
        warning_message = f"WARNING: NEW MODEL PERFORMANCE IS WORSE THAN EXISTING MODEL (F1 change: {improvement:.4f})"
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

def save_model(model, scaler, model_version, metrics, detailed_metrics, feature_importance, data_version, scaling_info, X_train_sample):
    """Save model, scaler, and metadata."""
    logger.info(f"Saving model version: {model_version}")
    
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info(f"Using model directory: {os.path.abspath(MODEL_DIR)}")
    
    # --- MLflow Logging ---
    temp_dir = None
    try:
        # Log model hyperparameters
        # Log model hyperparameters
        # Log model hyperparameters
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("random_state", model.random_state)
        mlflow.log_param("data_version", data_version)

# ✅ Log metrics (guaranteed fresh values)
        if   mlflow.active_run():
              for metric_name, value in metrics.items():
               mlflow.log_metric(metric_name, value)


        
        # Infer signature and create input example for MLflow model logging
        signature = infer_signature(X_train_sample, model.predict(X_train_sample))
        
        # Log the scikit-learn model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="retail24_random_forest_model",
            registered_model_name="Retail24RandomForestClassifier",
            signature=signature,
            input_example=X_train_sample
        )
        
        # Create temporary files to log them as artifacts
        temp_dir = "temp_mlflow_artifacts"
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_scaler_path = os.path.join(temp_dir, "scaler.pkl")
        temp_feature_names_path = os.path.join(temp_dir, "feature_names.json")
        temp_scaling_info_path = os.path.join(temp_dir, "scaling_info.json")
        temp_feature_importance_path = os.path.join(temp_dir, "feature_importance.json")
        
        joblib.dump(scaler, temp_scaler_path)
        mlflow.log_artifact(temp_scaler_path, artifact_path="artifacts")
        
        with open(temp_feature_names_path, 'w') as f:
            json.dump(scaling_info["all_features"], f)
        mlflow.log_artifact(temp_feature_names_path, artifact_path="artifacts")
        
        with open(temp_scaling_info_path, 'w') as f:
            json.dump(scaling_info, f)
        mlflow.log_artifact(temp_scaling_info_path, artifact_path="artifacts")
        
        with open(temp_feature_importance_path, 'w') as f:
            json.dump(feature_importance, f)
        mlflow.log_artifact(temp_feature_importance_path, artifact_path="artifacts")
        
        logger.info("Model, scaler, feature metadata, and feature importance logged to MLflow.")
    
    except MlflowException as e:
        logger.error(f"MLflow logging failed: {e}. Continuing with MongoDB save and local file save.")
    except Exception as e:
        logger.error(f"Unexpected error during MLflow logging: {e}. Continuing with MongoDB save and local file save.")
    finally:
        # Clean up the temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Could not clean up temp directory: {e}")
    
    # --- MongoDB Saving ---
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
            "metrics": metrics,
            "model_path": os.path.join(MODEL_DIR, f"rf_model_{model_version}.pkl"),
            "scaler_path": os.path.join(MODEL_DIR, f"scaler_{model_version}.pkl"),
            "feature_names_path": os.path.join(MODEL_DIR, f"feature_names_{model_version}.json"),
            "scaling_info_path": os.path.join(MODEL_DIR, f"scaling_info_{model_version}.json"),
            "status": "active",
            "feature_count": len(scaling_info["all_features"]),
            "scaled_feature_count": len(scaling_info["scaled_features"]),
            "mlflow_run_id": mlflow.active_run().info.run_id if mlflow.active_run() else None
        }
        
        # Insert new metadata
        model_metadata_collection.insert_one(metadata)
        logger.info("Model metadata saved to MongoDB")
        
        # Save detailed metrics to separate collection
        save_model_metrics(model_version, detailed_metrics)
        
        client.close()
        
        # Save model and scaler to local disk
        joblib.dump(model, metadata["model_path"])
        joblib.dump(scaler, metadata["scaler_path"])
        with open(metadata["feature_names_path"], 'w') as f:
            json.dump(scaling_info["all_features"], f)
        with open(metadata["scaling_info_path"], 'w') as f:
            json.dump(scaling_info, f)
        logger.info("Model files also saved locally for MongoDB consistency.")
    
    except Exception as e:
        logger.error(f"Error saving model metadata to MongoDB or local files: {str(e)}", exc_info=True)

def main():
    """Main function to train and evaluate a Random Forest model."""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Set MLflow experiment
    mlflow.set_experiment("Retail24_Training_Experiment_v2")

    
    logger.info("Starting model training process")
    
    # Start MLflow Run
    with mlflow.start_run():
        # Log git commit hash if available
        try:
            import git
            repo = git.Repo(search_parent_directories=True)
            mlflow.log_param("git_commit", repo.head.commit.hexsha)
        except Exception as e:
            logger.warning(f"Could not log git commit: {e}")
        
        if mlflow.active_run():
            logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        
        try:
            # Get all processed data
            df, latest_data_version = get_all_processed_data()
            if df is None:
                logger.error("No processed data available. Exiting.")
                return
            
            # Get latest model metadata from MongoDB
            latest_model_metadata = get_latest_model_metadata()
            
            # Check if we should train a new model
            should_train, reason = should_train_model(latest_data_version, latest_model_metadata)
            
            if not should_train:
                logger.info(f"Training decision: False - {reason}")
                logger.info("Using existing model. Training skipped.")
                return
            
            logger.info(f"Training decision: True - {reason}")
            
            # Prepare data for training
            X_train, X_test, y_train, y_test, scaler, scaling_info = prepare_data_for_training(df)
            
            # Train the model
            n_estimators = 150
            max_depth = 8
            random_state = 42
            
            model = train_random_forest(
                X_train=X_train, 
                y_train=y_train, 
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                random_state=random_state
            )
            
            # Evaluate the model
            metrics, detailed_metrics, feature_importance = evaluate_model(
                model, X_train, X_test, y_train, y_test, scaling_info["all_features"]
            )
            
            # Compare with existing model
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
                scaling_info=scaling_info,
                X_train_sample=X_train.head(5)
            )
            logger.info(f"Model training complete. New model version: {model_version}")
            
        except Exception as e:
            logger.error(f"Error in model training process: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    main()