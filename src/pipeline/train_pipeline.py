from src.components.data_ingestion import load_data
from src.pipeline.data_cleaning import clean_data
from src.pipeline.data_transformation import transform_data
from src.pipeline.model_trainer import train_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:

    logger.info("Pipeline started")

    # =========================
    # LOAD DATA
    # =========================

    df = load_data()

    # =========================
    # CLEAN DATA
    # =========================

    df = clean_data(df)

    # =========================
    # TRANSFORM DATA
    # =========================

    X_train, X_test, y_train, y_test, encoders, feature_columns = transform_data(df)

    # =========================
    # TRAIN MODEL
    # =========================

    best_model, best_f1, scaler, results_df = train_model(
        X_train,
        X_test,
        y_train,
        y_test,
        encoders,
        feature_columns
    )

    logger.info("Pipeline completed successfully")

except Exception as e:

    logger.error(f"Error in pipeline: {e}")

    raise