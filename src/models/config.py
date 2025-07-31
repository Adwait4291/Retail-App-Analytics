# src/config.py
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

# --- Define other directories based on the dynamic BASE_DIR ---
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Default files
DEFAULT_DATA_FILE = "retail_app_data.csv" # Make sure this matches your actual file name

# MongoDB collections
PRODUCTS_COLLECTION = "products"
METADATA_COLLECTION = "processing_metadata"


# Ingestion parameters
MIN_BATCH_SIZE = 200

# Data processing configurations
PRESERVE_HASH_IN_PROCESSED_DATA = False # Whether to keep record_hash in processed data