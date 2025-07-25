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

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('retail_pipeline')

# Constants
CURRENT_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
REPROCESSING_DAYS_THRESHOLD = 7  # Reprocess after 7 days

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
    
    # Check if processing is needed
    should_process, reason = should_process_data(raw_data)
    logger.info(f"Processing decision: {should_process} - {reason}")
    
    if not should_process:
        logger.info("No processing needed. Loading latest processed data...")
        processed_data = load_processed_data(latest=True)
        print_data_quality_report(processed_data)
        return

    # Process the data and get version info
    processed_data, processing_version = process_retail_data(raw_data)
    
    # Print data quality report
    print_data_quality_report(processed_data)
    
    # Save processed data with version info back to MongoDB
    save_processed_data(processed_data, processing_version, raw_data)
    
    logger.info("Data processing pipeline completed.")

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
    # Convert datetime columns
    df_processed['first_visit_date'] = pd.to_datetime(df_processed['first_visit_date'])
    df_processed['purchase_date'] = pd.to_datetime(df_processed['purchase_date'])
    
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
    df_processed = df_processed.drop(columns=columns_to_drop)
    
    # Ensure all column names are lowercase
    df_processed.columns = df_processed.columns.str.lower()
    
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
                raise ValueError("No processing metadata found")
                
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
        logger.error(f"Error loading processed data: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()