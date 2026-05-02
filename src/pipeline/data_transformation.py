import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)

def transform_data(df):
    try:
        logger.info("Data transformation started")

        # ✅ clean string values (extra spaces remove)
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()

        # ✅ separate target
        y = df["loan_status"]
        X = df.drop("loan_status", axis=1)

        # ✅ safe drop (no error if not present)
        X = X.drop(columns=["loan_id"], errors="ignore")

        # ✅ encode target
        y = y.map({"Approved": 1, "Rejected": 0})

        # ⚠️ IMPORTANT: convert object → string before encoding
        cat_cols = X.select_dtypes(include='object').columns

        encoders = {}

        for col in cat_cols:
            le = LabelEncoder()

            # 🔥 FIX: convert to string first
            X[col] = X[col].astype(str)

            X[col] = le.fit_transform(X[col])
            encoders[col] = le

        # ✅ feature order save (VERY IMPORTANT for API)
        feature_columns = X.columns.tolist()

        # ✅ SAVE artifacts
        os.makedirs("artifacts", exist_ok=True)

        with open("artifacts/encoders.pkl", "wb") as f:
            pickle.dump(encoders, f)

        with open("artifacts/feature_columns.pkl", "wb") as f:
            pickle.dump(feature_columns, f)

        logger.info("Encoders and feature columns saved")

        # ✅ train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info("Data transformation completed")

        return X_train, X_test, y_train, y_test, encoders, feature_columns

    except Exception as e:
        logger.error(f"Error in transformation: {e}")
        raise
