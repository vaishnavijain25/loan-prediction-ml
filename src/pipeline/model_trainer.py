import pandas as pd
import os
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_model(
    X_train,
    X_test,
    y_train,
    y_test,
    encoders,
    feature_columns
):

    try:

        logger.info("Model training started")

        # =========================
        # SCALING
        # =========================

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)

        X_test_scaled = scaler.transform(X_test)

        # =========================
        # MODELS
        # =========================

        models = {

            "LogisticRegression":

                LogisticRegression(
                    max_iter=1000
                ),

            "RandomForest":

                RandomForestClassifier(
                    random_state=42
                ),

            "DecisionTree":

                DecisionTreeClassifier(
                    random_state=42
                )
        }

        results = []

        best_model = None

        best_f1 = 0

        best_model_name = ""

        # =========================
        # TRAINING LOOP
        # =========================

        for name, model in models.items():

            logger.info(f"Training {name}")

            # Logistic Regression → scaled
            if name == "LogisticRegression":

                model.fit(
                    X_train_scaled,
                    y_train
                )

                y_pred = model.predict(
                    X_test_scaled
                )

            # Tree models → non-scaled
            else:

                model.fit(
                    X_train,
                    y_train
                )

                y_pred = model.predict(
                    X_test
                )

            # =========================
            # METRICS
            # =========================

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                zero_division=0
            )

            results.append({

                "Model": name,

                "Accuracy": accuracy,

                "Precision": precision,

                "Recall": recall,

                "F1": f1
            })

            print(
                f"{name} → Accuracy: {accuracy:.4f}, F1: {f1:.4f}"
            )

            # =========================
            # BEST MODEL
            # =========================

            if f1 > best_f1:

                best_f1 = f1

                best_model = model

                best_model_name = name

        # =========================
        # RESULTS DATAFRAME
        # =========================

        results_df = pd.DataFrame(results)

        logger.info(
            f"Best model: {best_model_name} "
            f"with F1: {best_f1:.4f}"
        )

        # =========================
        # SAVE ARTIFACTS
        # =========================

        os.makedirs(
            "artifacts",
            exist_ok=True
        )

        with open(
            "artifacts/model.pkl",
            "wb"
        ) as f:

            pickle.dump(
                best_model,
                f
            )

        with open(
            "artifacts/scaler.pkl",
            "wb"
        ) as f:

            pickle.dump(
                scaler,
                f
            )

        with open(
            "artifacts/encoders.pkl",
            "wb"
        ) as f:

            pickle.dump(
                encoders,
                f
            )

        with open(
            "artifacts/feature_columns.pkl",
            "wb"
        ) as f:

            pickle.dump(
                feature_columns,
                f
            )

        logger.info(
            "Artifacts saved successfully"
        )

        return (
            best_model,
            best_f1,
            scaler,
            results_df
        )

    except Exception as e:

        logger.error(
            f"Error in model training: {e}"
        )

        raise