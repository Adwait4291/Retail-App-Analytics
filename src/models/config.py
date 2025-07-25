# config.py

# src/config.py
import os
from pathlib import Path

# Project base directory - finds the project root regardless of where the script is run from
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Default files
DEFAULT_DATA_FILE = "retail_app_data.csv"

# MongoDB collections
PRODUCTS_COLLECTION = "products"
METADATA_COLLECTION = "source_metadata"

# Ingestion parameters
MIN_BATCH_SIZE = 200


# Data processing configurations
PRESERVE_HASH_IN_PROCESSED_DATA = False  # Whether to keep record_hash in processed data
