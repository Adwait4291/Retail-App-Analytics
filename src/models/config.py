# src/config.py
import os
from pathlib import Path

# Project base directory - updated to reflect the root directory inside the Docker container.
BASE_DIR = Path("/app")

# Data directories - these will now correctly resolve to /app/data/...
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
