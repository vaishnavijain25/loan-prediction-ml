import pandas as pd
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_data():
    try:
        logger.info("Data loading started")

        # ✅ Try both paths
        possible_paths = [
            "loan.csv",
            os.path.join("artifacts", "loan.csv")
        ]

        file_path = None

        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break

        if file_path is None:
            raise FileNotFoundError("loan.csv not found in root or artifacts folder")

        df = pd.read_csv(file_path)

        # ✅ Clean column names
        df.columns = df.columns.str.strip()

        logger.info(f"Data loaded successfully from {file_path}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")

        return df

    except Exception as e:
        logger.error(f"Error in data loading: {e}")
        raise
