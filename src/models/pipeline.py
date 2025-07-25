# pipeline.py

import os
import sys
import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('retail_pipeline')

def run_pipeline():
    """
    Main pipeline function that orchestrates the entire ML workflow:
    1. Process data from MongoDB source
    2. Train model on processed data 
    """
    pipeline_start_time = time.time()
    pipeline_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Starting ML pipeline run {pipeline_id}")
    
    # Step 1: Data Processing
    processing_success = run_data_processing()
    if not processing_success:
        logger.error("Data processing step failed. Pipeline aborted.")
        return False
    
    # Step 2: Model Training
    training_success = run_model_training()
    if not training_success:
        logger.error("Model training step failed. Pipeline aborted.")
        return False
    
    # Pipeline completed successfully
    pipeline_duration = (time.time() - pipeline_start_time) / 60
    logger.info(f"Pipeline run {pipeline_id} completed successfully in {pipeline_duration:.2f} minutes")
    return True

def run_data_processing():
    """
    Run the data processing step by importing and calling main function from processing.py
    """
    logger.info("Starting data processing step...")
    
    try:
        # Import main function from processing module
        from processing import main as process_main
        
        # Call the main processing function
        process_main()
        
        logger.info("Data processing step completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in data processing step: {str(e)}", exc_info=True)
        return False

def run_model_training():
    """
    Run the model training step by calling the main function from train.py
    """
    logger.info("Starting model training step...")
    
    try:
        # Import main function from train module
        from train import main as train_main
        
        # Call the main training function
        train_main()
        
        logger.info("Model training step completed")
        return True
        
    except Exception as e:
        logger.error(f"Error in model training step: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    try:
        success = run_pipeline()
        # Exit with appropriate code
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        logger.critical(f"Unhandled exception in pipeline: {str(e)}", exc_info=True)
        sys.exit(1)