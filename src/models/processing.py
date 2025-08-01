# processing.py
# retail_data_processor.py

import numpy as np
import pandas as pd
import os
import uuid
import hashlib
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
import logging
try:
    # Try local import first (when running from src directory)
    from config import PRESERVE_HASH_IN_PROCESSED_DATA
except ImportError:
    # Fall back to absolute import (when running from project root or in Docker)
    from src.config import PRESERVE_HASH_IN_PROCESSED_DATA
# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('retail_pipeline')

# Constants
CURRENT_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
REPROCESSING_DAYS_THRESHOLD = 7  # Reprocess after 7 days
MIN_NEW_RECORDS_THRESHOLD = 100  # Minimum new records required for processing

def main():
    """
    Main function that orchestrates the data processing pipeline.
    """
    logger.info("Starting retail data processing pipeline...")
    
    # Load data from MongoDB
    raw_data = fetch_from_mongodb()
    
    if raw_data.empty:
        logger.error("Could not fetch data from MongoDB. Exiting...")
        return
    
    # Get last processed timestamp from metadata
    last_processed_timestamp = get_last_processed_timestamp()
    
    # Determine what data needs to be processed based on timestamp
    if last_processed_timestamp:
        logger.info(f"Last processed timestamp: {last_processed_timestamp}")
        
        # Ensure timestamp column is in datetime format
        if 'timestamp' not in raw_data.columns:
            logger.error("Source data is missing timestamp column for incremental processing")
            return
            
        # Convert timestamp to datetime for comparison
        raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'])
        
        # Filter data for new records only
        new_data = raw_data[raw_data['timestamp'] > last_processed_timestamp]
        logger.info(f"Found {len(new_data)} new records since last processing")
        
        # Check if we have enough new records to process
        if len(new_data) < MIN_NEW_RECORDS_THRESHOLD:
            logger.info(f"Not enough new records to process. Found {len(new_data)}, need {MIN_NEW_RECORDS_THRESHOLD}")
            logger.info("No processing needed. Loading latest processed data...")
            processed_data = load_processed_data(latest=True)
            print_data_quality_report(processed_data)
            return
            
        # We have enough new records, process only these
        data_to_process = new_data
        logger.info(f"Processing {len(data_to_process)} new records")
    else:
        # First run - process all records
        logger.info("No previous processing found. Processing all records.")
        data_to_process = raw_data
        
    # Process the data and get version info
    processed_data, processing_version = process_retail_data(data_to_process)
    
    # Print data quality report
    print_data_quality_report(processed_data)
    
    # Get all existing processed data
    existing_processed_data = load_processed_data(latest=True)
    
    # If existing data exists, append new processed data to it
    if not existing_processed_data.empty:
        # Check if we need to handle duplicate user_ids or other keys
        if 'user_id' in processed_data.columns and 'user_id' in existing_processed_data.columns:
            # Remove any duplicate user_ids that might exist in both datasets
            existing_user_ids = set(existing_processed_data['user_id'])
            processed_data = processed_data[~processed_data['user_id'].isin(existing_user_ids)]
        
        # Update the processing version for all data to be consistent
        existing_processed_data['processing_version'] = processing_version
        
        # Combine existing and new processed data
        combined_data = pd.concat([existing_processed_data, processed_data], ignore_index=True)
        logger.info(f"Combined {len(existing_processed_data)} existing records with {len(processed_data)} new records")
        
        # Use the combined data for saving
        processed_data = combined_data
    
    # Save processed data with version info back to MongoDB
    save_processed_data(processed_data, processing_version, data_to_process)
    
    # Save the latest timestamp to metadata
    if 'timestamp' in data_to_process.columns:
        try:
            # Ensure timestamps are in datetime format before finding max
            timestamps = pd.to_datetime(data_to_process['timestamp'])
            latest_timestamp = timestamps.max()
            update_last_processed_timestamp(latest_timestamp, processing_version)
        except Exception as e:
            logger.warning(f"Error processing timestamp: {str(e)}. Using current time.")
            update_last_processed_timestamp(datetime.now(), processing_version)
    
    logger.info("Data processing pipeline completed.")

def get_last_processed_timestamp():
    """
    Get the timestamp of the last processed record from metadata.
    
    Returns:
    --------
    datetime or None
        The timestamp of the last processed record, or None if no previous processing
    """
    logger.info("Retrieving last processed timestamp...")
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Get MongoDB connection details
        username = os.getenv("MONGODB_USERNAME")
        password = os.getenv("MONGODB_PASSWORD")
        cluster = os.getenv("MONGODB_CLUSTER")
        database = os.getenv("MONGODB_DATABASE")
        
        # Create connection string
        connection_string = f"mongodb+srv://{username}:{password}@{cluster}/"
        
        # Connect to MongoDB
        client = MongoClient(connection_string)
        db = client.get_database(database)
        metadata_collection = db.processing_metadata
        
        # Get the most recent processing record
        last_processing = metadata_collection.find_one(
            sort=[("processed_at", -1)]  # Sort by processed_at descending
        )
        
        if not last_processing or "last_processed_timestamp" not in last_processing:
            client.close()
            return None
        
        # Get the timestamp
        last_timestamp = last_processing["last_processed_timestamp"]
        client.close()
        
        return last_timestamp
        
    except Exception as e:
        logger.error(f"Error retrieving last processed timestamp: {str(e)}", exc_info=True)
        return None

def update_last_processed_timestamp(timestamp, processing_version):
    """
    Update the last processed timestamp in metadata.
    
    Parameters:
    -----------
    timestamp : datetime
        The latest timestamp processed
    processing_version : str
        The version identifier for this processing run
    """
    logger.info(f"Updating last processed timestamp to {timestamp}...")
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Get MongoDB connection details
        username = os.getenv("MONGODB_USERNAME")
        password = os.getenv("MONGODB_PASSWORD")
        cluster = os.getenv("MONGODB_CLUSTER")
        database = os.getenv("MONGODB_DATABASE")
        
        # Create connection string
        connection_string = f"mongodb+srv://{username}:{password}@{cluster}/"
        
        # Connect to MongoDB
        client = MongoClient(connection_string)
        db = client.get_database(database)
        metadata_collection = db.processing_metadata
        
        # Find if there's an existing metadata entry for this processing version
        existing_metadata = metadata_collection.find_one({"processing_version": processing_version})
        
        if existing_metadata:
            # Update the existing entry
            metadata_collection.update_one(
                {"processing_version": processing_version},
                {"$set": {"last_processed_timestamp": timestamp}}
            )
        else:
            # Create a new entry
            metadata = {
                "processing_version": processing_version,
                "processed_at": datetime.now(),
                "last_processed_timestamp": timestamp,
                "domain": "retail"
            }
            metadata_collection.insert_one(metadata)
        
        client.close()
        logger.info(f"Successfully updated last processed timestamp")
        
    except Exception as e:
        logger.error(f"Error updating last processed timestamp: {str(e)}", exc_info=True)

def should_process_data(df):
    """
    Determine if the data needs processing based on metadata
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw data to check
        
    Returns:
    --------
    tuple (boolean, str)
        (True/False whether processing is needed, reason for the decision)
    """
    logger.info("Checking if data needs processing...")
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Get MongoDB connection details
        username = os.getenv("MONGODB_USERNAME")
        password = os.getenv("MONGODB_PASSWORD")
        cluster = os.getenv("MONGODB_CLUSTER")
        database = os.getenv("MONGODB_DATABASE")
        
        # Create connection string
        connection_string = f"mongodb+srv://{username}:{password}@{cluster}/"
        
        # Connect to MongoDB
        client = MongoClient(connection_string)
        db = client.get_database(database)
        metadata_collection = db.processing_metadata
        
        # Get the most recent processing record
        last_processing = metadata_collection.find_one(
            sort=[("processed_at", -1)]  # Sort by processed_at descending
        )
        
        if not last_processing:
            client.close()
            return True, "No previous processing found"
        
        # Compare basic statistics
        current_stats = {
            "record_count": len(df),
            "columns": sorted(df.columns.tolist())
        }
        
        # Add user ID count if available
        if "user_id" in df.columns:
            current_stats["id_count"] = df["user_id"].nunique()
        
        # Check if record count differs
        if current_stats["record_count"] != last_processing.get("record_count"):
            client.close()
            return True, f"Record count changed: {last_processing.get('record_count')} -> {current_stats['record_count']}"
        
        # Check if ID count differs (if available)
        if current_stats.get("id_count") and current_stats["id_count"] != last_processing.get("id_count"):
            client.close()
            return True, f"User count changed: {last_processing.get('id_count')} -> {current_stats['id_count']}"
        
        # Check if schema changed
        if current_stats["columns"] != last_processing.get("columns"):
            client.close()
            return True, f"Schema changed. Columns differ."
        
        # Check time since last processing
        last_processed_time = last_processing["processed_at"]
        days_since_processing = (datetime.now() - last_processed_time).days
        
        if days_since_processing > REPROCESSING_DAYS_THRESHOLD:
            client.close()
            return True, f"Last processing was {days_since_processing} days ago (threshold: {REPROCESSING_DAYS_THRESHOLD})"
        
        client.close()
        return False, f"Data appears unchanged since last processing ({last_processing['processing_version']})"
        
    except Exception as e:
        logger.error(f"Error checking if data needs processing: {str(e)}", exc_info=True)
        # If we can't determine, better to process the data
        return True, f"Error checking processing status: {str(e)}"

def fetch_from_mongodb():
    """
    Fetch data from MongoDB collection.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the raw retail data
    """
    logger.info("Fetching data from MongoDB...")
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get MongoDB connection details from environment variables
    username = os.getenv("MONGODB_USERNAME")
    password = os.getenv("MONGODB_PASSWORD")
    cluster = os.getenv("MONGODB_CLUSTER")
    database = os.getenv("MONGODB_DATABASE")
    
    # Create connection string
    connection_string = f"mongodb+srv://{username}:{password}@{cluster}/"
    
    try:
        # Create a client connection
        client = MongoClient(connection_string)
        
        # Connect to the database
        db = client.get_database(database)
        collection = db.products  # Collection name as in your notebook
        
        # Fetch data (excluding _id field)
        cursor = collection.find({}, {'_id': 0})
        df = pd.DataFrame(list(cursor))
        
        # Check if record_hash exists in the data
        has_record_hash = 'record_hash' in df.columns
        logger.info(f"Data contains record_hash column: {has_record_hash}")
        
        logger.info(f"Successfully fetched {len(df)} records from MongoDB")
        
        # Close the connection
        client.close()
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching data from MongoDB: {e}")
        return pd.DataFrame()

def process_retail_data(df):
    """
    Process retail app data for analysis and ML modeling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing raw retail data
        
    Returns:
    --------
    tuple (pandas.DataFrame, str)
        Processed dataframe and processing version
    """
    logger.info("Processing retail data...")
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    logger.info("Processing time-based features...")
    # Time-based Processing
    # Convert datetime columns - with error handling
    df_processed['first_visit_date'] = pd.to_datetime(df_processed['first_visit_date'], errors='coerce')
    df_processed['purchase_date'] = pd.to_datetime(df_processed['purchase_date'], errors='coerce')
    
    # Calculate time difference and create target
    df_processed['time_to_purchase'] = (df_processed['purchase_date'] - 
                                      df_processed['first_visit_date']).dt.total_seconds() / 3600
    
    # Create 24-hour purchase target
    df_processed['purchase_24h'] = np.where(df_processed['time_to_purchase'] <= 24, 1, 0)
    
    # Extract time features
    df_processed['hour'] = df_processed['first_visit_date'].dt.hour
    df_processed['dayofweek'] = df_processed['first_visit_date'].dt.dayofweek
    df_processed['is_weekend'] = df_processed['dayofweek'].isin([5,6]).astype(int)
    
    logger.info("Processing screen list data...")
    # Screen List Processing
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
    
    logger.info("Creating advanced features...")
    # Feature Engineering
    # Create engagement score
    df_processed['engagement_score'] = (
        df_processed['session_count'] * 0.3 +
        df_processed['used_search_feature'] * 0.2 +
        df_processed['wrote_review'] * 0.15 +
        df_processed['added_to_wishlist'] * 0.15 +
        df_processed['total_screens_viewed'] * 0.2
    )
    
    # Create screen diversity score
    df_processed['screen_diversity'] = (
        df_processed[['shopping_count', 'cart_count', 
                     'engagement_count', 'account_count']].gt(0).sum(axis=1)
    )
    
    # Create purchase intent score
    df_processed['purchase_intent'] = (
        df_processed['cart_count'] * 0.4 +
        df_processed['shopping_count'] * 0.3 +
        df_processed['engagement_count'] * 0.2 +
        df_processed['added_to_wishlist'] * 0.1
    )
    
    logger.info("Processing categorical features...")
    # Categorical Feature Processing
    # Platform encoding
    df_processed['platform'] = df_processed['platform'].map({'iOS': 1, 'Android': 0})
    
    # Region encoding
    region_dummies = pd.get_dummies(df_processed['region'], prefix='region')
    df_processed = pd.concat([df_processed, region_dummies], axis=1)
    
    # Acquisition channel encoding
    channel_dummies = pd.get_dummies(df_processed['acquisition_channel'], prefix='channel')
    df_processed = pd.concat([df_processed, channel_dummies], axis=1)
    
    # User segment processing
    df_processed['age_group'] = df_processed['user_segment'].apply(lambda x: x.split()[0])
    df_processed['user_type'] = df_processed['user_segment'].apply(lambda x: ' '.join(x.split()[1:]))
    
    age_group_dummies = pd.get_dummies(df_processed['age_group'], prefix='age_group')
    user_type_dummies = pd.get_dummies(df_processed['user_type'], prefix='user_type')
    df_processed = pd.concat([df_processed, age_group_dummies, user_type_dummies], axis=1)
    
    # App version processing
    df_processed['app_major_version'] = df_processed['app_version'].apply(lambda x: int(x.split('.')[0]))
    
    # Create version recency score
    df_processed['version_score'] = df_processed['app_version'].apply(
        lambda x: sum(float(i)/(10**n) for n, i in enumerate(x.split('.')))
    )
    
    logger.info("Cleaning up final dataset...")
    # Clean up and prepare final dataset
    # Drop original columns that have been processed
    columns_to_drop = [
        'screen_list', 'purchase_date', 'first_visit_date', 
        'time_to_purchase', 'made_purchase', 'region', 
        'acquisition_channel', 'user_segment', 'app_version',
        'age_group', 'user_type'
    ]
    
    # Conditionally add record_hash to columns to drop based on configuration
    if not PRESERVE_HASH_IN_PROCESSED_DATA and 'record_hash' in df_processed.columns:
        columns_to_drop.append('record_hash')
        logger.info("Removing record_hash from processed data (as per configuration)")
    
    # Only drop columns that exist
    columns_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
    df_processed = df_processed.drop(columns=columns_to_drop)
    
    # Ensure all column names are lowercase
    df_processed.columns = df_processed.columns.str.lower()
    
    # Preserve timestamp column if it exists (for incremental processing)
    if 'timestamp' in df_processed.columns:
        # First ensure it's in datetime format
        df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
        # Then convert to string format for storage
        df_processed['timestamp'] = df_processed['timestamp'].astype(str)
    
    # Generate a processing version identifier
    processing_version = f"retail_v{CURRENT_TIMESTAMP}_{uuid.uuid4().hex[:8]}"
    logger.info(f"Processing version: {processing_version}")
    
    # Add the processing version to the dataframe
    df_processed["processing_version"] = processing_version
    
    logger.info(f"Data processing completed with shape: {df_processed.shape}")
    return df_processed, processing_version

def print_data_quality_report(df):
    """
    Print a data quality report for the processed dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed dataframe
    """
    logger.info("\nGenerating Data Quality Report...")
    print("\nData Quality Report")
    print("-" * 50)
    print(f"Shape: {df.shape}")
    print(f"\nNull values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    
    if 'purchase_24h' in df.columns:
        print(f"\nPurchase rate (24h): {df['purchase_24h'].mean():.2%}")
    
        # Feature correlations - only include numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if 'purchase_24h' in numeric_cols:
            correlation_matrix = df[numeric_cols].corr()['purchase_24h'].sort_values(ascending=False)
            print("\nTop 10 Features by Correlation with Purchase:")
            print(correlation_matrix[:10])
    
    processing_version = df['processing_version'].iloc[0] if 'processing_version' in df.columns else "N/A"
    print(f"\nProcessing Version: {processing_version}")

def save_processed_data(processed_df, processing_version, original_df=None):
    """
    Save processed data and metadata to MongoDB.
    
    Parameters:
    -----------
    processed_df : pandas.DataFrame
        Processed dataframe to save
    processing_version : str
        Version identifier for this processing run
    original_df : pandas.DataFrame, optional
        Original dataframe (for metadata)
        
    Returns:
    --------
    int
        Number of documents inserted
    """
    logger.info(f"Saving processed data to MongoDB (version: {processing_version})...")
    
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
        
        # Save to processed data collection
        collection = db.processed_retail_data
        
        # Clear existing data - this is crucial when updating all documents
        collection.drop()
        logger.info("Dropped existing processed data collection")
        
        # Convert DataFrame to dictionary records
        records = processed_df.to_dict("records")
        
        # Insert records
        result = collection.insert_many(records)
        docs_inserted = len(result.inserted_ids)
        logger.info(f"Successfully inserted {docs_inserted} processed records to MongoDB")
        
        # Add metadata saving
        metadata_collection = db.processing_metadata
        
        # Prepare metadata - using original_df columns for comparison
        metadata = {
            "processing_version": processing_version,
            "processed_at": datetime.now(),
            "record_count": len(processed_df),
            "domain": "retail"
        }
        
        # Store the ORIGINAL dataframe columns, not the processed ones
        if original_df is not None:
            metadata["columns"] = sorted(original_df.columns.tolist())
        else:
            metadata["columns"] = sorted(processed_df.columns.tolist())
        
        # Add user ID count if available in original df
        if original_df is not None and "user_id" in original_df.columns:
            metadata["id_count"] = original_df["user_id"].nunique()
        
        # Calculate data quality metrics
        if "purchase_24h" in processed_df.columns:
            metadata["purchase_rate"] = float(processed_df["purchase_24h"].mean())
        
        # Get processing stats (advanced features)
        metadata["feature_count"] = processed_df.shape[1]
        
        # Add data hash to detect changes
        if original_df is not None:
            # Create a hash of the data to detect changes
            data_sample = original_df.head(100).to_json()
            metadata["data_hash"] = hashlib.md5(data_sample.encode()).hexdigest()
        
        # Store last processed timestamp if available
        if original_df is not None and 'timestamp' in original_df.columns:
            try:
                # Convert to datetime first to ensure consistent format
                timestamps = pd.to_datetime(original_df['timestamp'])
                metadata["last_processed_timestamp"] = timestamps.max()
            except Exception as e:
                # Fallback if there's an error with timestamp processing
                logger.warning(f"Error processing timestamp: {str(e)}. Using current time.")
                metadata["last_processed_timestamp"] = datetime.now()
        
        # Save metadata
        metadata_collection.insert_one(metadata)
        logger.info(f"Saved processing metadata for version {processing_version}")
        
        client.close()
        
        return docs_inserted
        
    except Exception as e:
        logger.error(f"Error saving processed data: {str(e)}", exc_info=True)
        raise

def load_processed_data(latest=True, processing_version=None):
    """
    Load processed data from MongoDB
    
    Parameters:
    -----------
    latest : bool, default=True
        If True, get the latest processed data
    processing_version : str, optional
        Specific processing version to load
        
    Returns:
    --------
    pandas.DataFrame
        Processed data
    """
    logger.info("Loading processed data from MongoDB...")
    
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
        collection = db.processed_retail_data
        
        # Query based on parameters
        if processing_version:
            # Get specific version
            query = {"processing_version": processing_version}
            logger.info(f"Loading specific processing version: {processing_version}")
        elif latest:
            # Get latest version
            metadata_collection = db.processing_metadata
            latest_metadata = metadata_collection.find_one({"domain": "retail"}, sort=[("processed_at", -1)])
            
            if not latest_metadata:
                logger.info("No processing metadata found - this appears to be the first run")
                return pd.DataFrame()  # Return empty DataFrame for first run
                
            latest_version = latest_metadata["processing_version"]
            query = {"processing_version": latest_version}
            logger.info(f"Loading latest processing version: {latest_version}")
        else:
            # Get all processed data
            query = {}
            logger.info("Loading all processed data")
        
        # Fetch data
        data = list(collection.find(query))
        
        if not data:
            logger.warning("No processed data found")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Remove MongoDB _id field if present
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
            
        # Close connection
        client.close()
        
        logger.info(f"Loaded processed data with shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading processed data: {str(e)}")
        # Return empty DataFrame on error rather than raising exception
        return pd.DataFrame()

if __name__ == "__main__":
    main()