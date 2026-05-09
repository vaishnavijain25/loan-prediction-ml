import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def clean_data(df):
    try:
        logger.info("Data cleaning started")

        # ✅ Remove duplicates
        df = df.drop_duplicates()

        # ✅ Clean column names
        df.columns = df.columns.str.strip()

        # ✅ Clean string values safely
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.strip()

        # ✅ Drop unnecessary column safely
        if "loan_id" in df.columns:
            df = df.drop("loan_id", axis=1)

        # ✅ Handle missing values
        if df.isnull().sum().sum() > 0:

            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
            cat_cols = df.select_dtypes(include=['object']).columns

            # Numeric → median
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())

            # Categorical → mode
            df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

        logger.info("Data cleaning completed")

        return df

    except Exception as e:
        logger.error(f"Error in data cleaning: {e}")
        raise