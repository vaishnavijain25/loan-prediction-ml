from src.components.data_ingestion import load_data
from src.pipeline.data_cleaning import clean_data
from src.pipeline.data_transformation import transform_data
from src.pipeline.model_trainer import train_model
from src.pipeline.model_tuner import tune_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    logger.info("Pipeline started")

    # 1. Load Data
    df = load_data()

    # 2. Clean Data
    df = clean_data(df)

    # 3. Transform Data
    X_train, X_test, y_train, y_test, encoders, feature_columns = transform_data(df)

    # 4. Train Models
    best_model, best_f1, scaler, results_df = train_model(
        X_train, X_test, y_train, y_test, encoders, feature_columns
    )

    logger.info("Model comparison completed")

    # 5. Tune Model (NO manual scaling here ❌)
    final_model = tune_model(
        X_train,
        X_test,
        y_train,
        y_test,
        best_model,
        best_f1,
        scaler
    )

    logger.info("Pipeline completed successfully")

except Exception as e:
    logger.error(f"Error in pipeline: {e}")
    raise
