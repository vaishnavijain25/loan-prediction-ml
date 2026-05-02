import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def clean_data(df):
    try:
        logger.info("Data cleaning started")

        # ✅ remove duplicates
        df = df.drop_duplicates()

        # ✅ strip column names (extra spaces remove)
        df.columns = df.columns.str.strip()

        # ✅ strip string values ALSO (VERY IMPORTANT)
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()

        # ✅ drop unnecessary column safely
        if "loan_id" in df.columns:
            df = df.drop("loan_id", axis=1)

        # ✅ handle missing values only if present
        if df.isnull().sum().sum() > 0:

            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
            cat_cols = df.select_dtypes(include=['object']).columns

            # numeric → median
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())

            # categorical → mode
            df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

        logger.info("Data cleaning completed")

        return df

    except Exception as e:
        logger.error(f"Error in data cleaning: {e}")
        raise
