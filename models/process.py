# retail_data_processor.py

import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# --- Configuration ---
RAW_DATA_COLLECTION = "products"
PROCESSED_DATA_COLLECTION = "processed_data"

def get_mongo_connection():
    """
    Establishes and returns a connection to a MongoDB database.
    """
    load_dotenv()
    username = os.getenv("MONGODB_USERNAME", "").strip()
    password = os.getenv("MONGODB_PASSWORD", "").strip()
    cluster = os.getenv("MONGODB_CLUSTER", "").strip()
    database_name = os.getenv("MONGODB_DATABASE", "").strip()

    if not all([username, password, cluster, database_name]):
        print("Error: MongoDB environment variables not fully set. Please check your .env file.")
        return None

    connection_string = f"mongodb+srv://{username}:{password}@{cluster}/"
    client = MongoClient(connection_string)

    try:
        client.admin.command('ismaster')
        db = client.get_database(database_name)
        return db
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def main():
    print("Starting retail data processing pipeline...")
    
    db = get_mongo_connection()
    if db is None:
        return

    raw_data = fetch_from_mongodb(db)
    if raw_data is None or raw_data.empty:
        print("Halting: Could not fetch data from MongoDB.")
        return

    processed_data = process_retail_data(raw_data)
    print_data_quality_report(processed_data)
    save_to_mongodb(processed_data, db)
    print("Data processing pipeline completed.")

def fetch_from_mongodb(db):
    print(f"Fetching data from collection: '{RAW_DATA_COLLECTION}'...")
    try:
        collection = db[RAW_DATA_COLLECTION]
        cursor = collection.find({}, {'_id': 0})
        df = pd.DataFrame(list(cursor))
        print(f"Successfully fetched {len(df)} records from MongoDB.")
        return df
    except Exception as e:
        print(f"Error fetching data from MongoDB: {e}")
        return None

def process_retail_data(df):
    print("Processing retail data...")
    df_processed = df.copy()

    print("Processing time-based features...")
    df_processed['first_visit_date'] = pd.to_datetime(df_processed['first_visit_date'])
    df_processed['purchase_date'] = pd.to_datetime(df_processed['purchase_date'], errors='coerce')

    df_processed['time_to_purchase'] = (df_processed['purchase_date'] - df_processed['first_visit_date']).dt.total_seconds() / 3600
    df_processed['purchase_24h'] = np.where(df_processed['time_to_purchase'] <= 24, 1, 0)

    df_processed['hour'] = df_processed['first_visit_date'].dt.hour
    df_processed['dayofweek'] = df_processed['first_visit_date'].dt.dayofweek
    df_processed['is_weekend'] = df_processed['dayofweek'].isin([5, 6]).astype(int)

    print("Processing screen list data...")
    df_processed['screen_list'] = df_processed['screen_list'].astype(str) + ','

    shopping_screens = ['ProductList', 'ProductDetail', 'CategoryBrowse', 'Search']
    cart_screens = ['ShoppingCart', 'Checkout', 'PaymentMethods', 'DeliveryOptions']
    engagement_screens = ['WishList', 'Reviews', 'Promotions']
    account_screens = ['Account', 'AddressBook', 'OrderTracking']
    all_tracked_screens = shopping_screens + cart_screens + engagement_screens + account_screens

    for screen in all_tracked_screens:
        df_processed[screen.lower()] = df_processed['screen_list'].str.contains(screen + ',', regex=False).astype(int)

    df_processed['shopping_count'] = df_processed[[s.lower() for s in shopping_screens]].sum(axis=1)
    df_processed['cart_count'] = df_processed[[s.lower() for s in cart_screens]].sum(axis=1)
    df_processed['engagement_count'] = df_processed[[s.lower() for s in engagement_screens]].sum(axis=1)
    df_processed['account_count'] = df_processed[[s.lower() for s in account_screens]].sum(axis=1)

    df_processed['other_screens'] = df_processed['screen_list'].apply(
        lambda x: len([s for s in x.split(',') if s and s not in all_tracked_screens])
    )

    print("Creating advanced features...")
    df_processed['engagement_score'] = (
        df_processed['session_count'] * 0.3 +
        df_processed['used_search_feature'] * 0.2 +
        df_processed['wrote_review'] * 0.15 +
        df_processed['added_to_wishlist'] * 0.15 +
        df_processed['total_screens_viewed'] * 0.2
    )

    df_processed['screen_diversity'] = df_processed[
        ['shopping_count', 'cart_count', 'engagement_count', 'account_count']
    ].gt(0).sum(axis=1)

    df_processed['purchase_intent'] = (
        df_processed['cart_count'] * 0.4 +
        df_processed['shopping_count'] * 0.3 +
        df_processed['engagement_count'] * 0.2 +
        df_processed['added_to_wishlist'] * 0.1
    )

    print("Processing categorical features...")
    df_processed['platform'] = df_processed['platform'].map({'iOS': 1, 'Android': 0})

    df_processed = pd.get_dummies(df_processed, columns=['region', 'acquisition_channel'], prefix=['region', 'channel'])

    df_processed['age_group'] = df_processed['user_segment'].apply(lambda x: x.split()[0])
    df_processed['user_type'] = df_processed['user_segment'].apply(lambda x: ' '.join(x.split()[1:]))
    df_processed = pd.get_dummies(df_processed, columns=['age_group', 'user_type'], prefix=['age_group', 'user_type'])

    df_processed['app_major_version'] = df_processed['app_version'].apply(lambda x: int(x.split('.')[0]))
    df_processed['version_score'] = df_processed['app_version'].apply(
        lambda x: sum(float(i)/(10**n) for n, i in enumerate(x.split('.')))
    )

    print("Cleaning up final dataset...")
    columns_to_drop = [
        'screen_list', 'purchase_date', 'first_visit_date', 'time_to_purchase', 
        'made_purchase', 'user_segment', 'app_version', 'age_group', 'user_type'
    ]
    existing_cols_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
    df_processed = df_processed.drop(columns=existing_cols_to_drop)

    df_processed.columns = df_processed.columns.str.lower().str.replace(' ', '_')
    return df_processed

def print_data_quality_report(df):
    print("\nData Quality Report")
    print("-" * 50)
    print(f"Shape: {df.shape}")

    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if nulls.empty:
        print("\nNull values: None found.")
    else:
        print(f"\nNull values:\n{nulls}")

    if 'purchase_24h' in df.columns:
        print(f"\nPurchase rate (24h): {df['purchase_24h'].mean():.2%}")
        correlation = df.corr(numeric_only=True)['purchase_24h'].sort_values(ascending=False)
        print("\nTop 10 Features by Correlation with Purchase:")
        print(correlation.head(10))

def save_to_mongodb(df, db):
    print(f"\nSaving processed data to collection: '{PROCESSED_DATA_COLLECTION}'...")
    try:
        collection = db[PROCESSED_DATA_COLLECTION]
        records = df.to_dict("records")

        print("Clearing existing records...")
        collection.delete_many({})

        print(f"Inserting {len(records)} new records...")
        result = collection.insert_many(records)

        print(f"Successfully saved {len(result.inserted_ids)} processed records to MongoDB.")

        print("\nSample of saved processed data (first 3 records):")
        for doc in collection.find().limit(3):
            print(f"  User ID: {doc.get('user_id', 'N/A')}, "
                  f"Purchase 24h: {doc.get('purchase_24h', 'N/A')}, "
                  f"Engagement Score: {doc.get('engagement_score', 0):.2f}")
    except Exception as e:
        print(f"Error saving data to MongoDB: {e}")
    finally:
        if db is not None and db.client is not None:
            db.client.close()
            print("MongoDB connection closed.")

if __name__ == "__main__":
    main()
