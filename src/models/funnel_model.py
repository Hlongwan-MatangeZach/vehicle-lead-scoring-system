"""
Funnel prediction model for the Vehicle Lead Scoring System.

Trains separate binary classifiers for each funnel stage and
produces ranked lead lists for queue prioritisation.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import lightgbm as lgb
import joblib
import logging

logger = logging.getLogger(__name__)

from src.config import (
    FUNNEL_STAGES,
    TARGET_COLUMNS,
    LEAKAGE_COLUMNS,
    ID_COLUMNS,
    RANDOM_STATE,
    MODELS_DIR,
)


class FunnelModel:
    """
    Multi-target funnel prediction model.

    Trains one binary classifier per funnel stage (Quote, Prospect,
    Inspection, Accepted) and provides methods for prediction
    and ranking.
    """

    def __init__(self, model_type: str = "lightgbm"):
        """
        Parameters
        ----------
        model_type : str
            Type of model to use. Options: 'lightgbm', 'xgboost', 'logistic'.
        """
        self.model_type = model_type
        self.models: Dict[str, object] = {}
        self.stage_features: Dict[str, List[str]] = {}
        self.is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.DataFrame] = None,
    ) -> "FunnelModel":
        """
        Train a separate model for each funnel stage.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.DataFrame
            Training targets with columns for each funnel stage.
        X_val : pd.DataFrame, optional
            Validation feature matrix for early stopping.
        y_val : pd.DataFrame, optional
            Validation targets.

        Returns
        -------
        self
        """
        self.feature_columns = list(X_train.columns)

        for stage in FUNNEL_STAGES:
            if stage not in y_train.columns:
                logger.warning(f"target '{stage}' not found, skipping.")
                continue

            logger.info(f"Training model for stage: {stage}")
            model = self._create_model()

            if X_val is not None and y_val is not None:
                callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)]
                model.fit(
                    X_train[self.feature_columns],
                    y_train[stage],
                    eval_set=[(X_val[self.feature_columns], y_val[stage])],
                    callbacks=callbacks
                )
            else:
                model.fit(X_train[self.feature_columns], y_train[stage])

            self.models[stage] = model

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict probability of reaching each funnel stage.

        Returns
        -------
        pd.DataFrame
            Columns: one per funnel stage, values: predicted probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        predictions = {}
        for stage, model in self.models.items():
            predictions[stage] = model.predict_proba(X[self.feature_columns])[:, 1]

        return pd.DataFrame(predictions, index=X.index)

    def rank_leads(
        self,
        X: pd.DataFrame,
        ranking_stage: str = "accepted",
    ) -> pd.DataFrame:
        """
        Rank enquiries by predicted probability of reaching the target stage.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for new enquiries.
        ranking_stage : str
            Funnel stage to rank by (default: 'accepted').

        Returns
        -------
        pd.DataFrame
            Sorted by predicted probability (descending) with rank column.
        """
        probas = self.predict_proba(X)
        probas["rank"] = probas[ranking_stage].rank(ascending=False, method="first")
        return probas.sort_values("rank")

    def _create_model(self):
        """Create a model instance based on the configured model_type."""
        if self.model_type == "lightgbm":
            return lgb.LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=7,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=-1
            )
        else:
            raise NotImplementedError(f"Model type '{self.model_type}' not implemented.")

    def save(self, path: Optional[Path] = None):
        """Serialize the trained model to disk."""
        if path is None:
            path = MODELS_DIR / "funnel_model.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None):
        """Load a previously trained model from disk."""
        if path is None:
            path = MODELS_DIR / "funnel_model.joblib"
        logger.info(f"Loading model from {path}")
        return joblib.load(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("FunnelModel scaffold ready.")
    logger.info(f"Funnel stages: {FUNNEL_STAGES}")
