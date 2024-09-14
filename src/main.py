import os
import logging
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import main as preprocess_data
from feature_engineering import main as engineer_features
import model_development
import model_evaluation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting Martian Dust Storm Prediction pipeline")

        logger.info("Step 1: Data Preprocessing")
        preprocess_data()

        logger.info("Step 2: Feature Engineering")
        engineer_features()

        logger.info("Step 3: Model Development")
        exec(open("model_development.py").read())

        logger.info("Step 4: Model Evaluation")
        exec(open("model_evaluation.py").read())

        logger.info("Martian Dust Storm Prediction pipeline completed successfully")
    except Exception as e:
        logger.error(f"An error occurred during the pipeline execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()