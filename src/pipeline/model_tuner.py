import os
import pickle

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

from src.utils.logger import get_logger

logger = get_logger(__name__)

def tune_model(X_train, X_test, y_train, y_test, best_model, best_f1, scaler):
    try:
        logger.info("Started XGBoost tuning")

        # ✅ Scale data (IMPORTANT: same scaler from training)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ✅ Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1]
        }

        # ✅ GridSearch
        grid = GridSearchCV(
            XGBClassifier(random_state=42, eval_metric='logloss'),
            param_grid=param_grid,
            cv=3,
            scoring='f1',
            verbose=1
        )

        grid.fit(X_train_scaled, y_train)

        tuned_xgb = grid.best_estimator_

        # ✅ Evaluate tuned model
        y_pred = tuned_xgb.predict(X_test_scaled)
        tuned_f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Tuned XGBoost F1: {tuned_f1:.4f}")

        # ✅ Compare with baseline
        if tuned_f1 > best_f1:
            final_model = tuned_xgb
            print("✅ Tuned XGBoost selected as Final Model")
            logger.info("Tuned XGBoost selected as final model")

        else:
            final_model = best_model
            print("✅ Baseline model selected as Final Model")
            logger.info("Baseline model selected as final model")

        # ✅ SAVE FINAL MODEL
        os.makedirs("artifacts", exist_ok=True)

        with open("artifacts/model.pkl", "wb") as f:
            pickle.dump(final_model, f)

        logger.info("Final model saved successfully")

        return final_model

    except Exception as e:
        logger.error(f"Error in tuning: {e}")
        raise
