"""
Main pipeline entry point for the Vehicle Lead Scoring System.

Orchestrates the full pipeline: data loading → feature engineering →
model training → prediction → submission.
"""
import pandas as pd
from src.data.data_loader import load_train_data, load_test_data, load_deal_calls
from src.features.feature_engineer import engineer_features
from src.models.funnel_model import FunnelModel
from src.utils.helpers import (
    create_time_based_split,
    print_class_balance,
    plot_funnel_distribution,
)
from src.config import (
    TARGET_COLUMNS,
    LEAKAGE_COLUMNS,
    ID_COLUMNS,
    FIGURES_DIR,
    REPORTS_DIR,
    FUNNEL_STAGES,
)

import logging

logger = logging.getLogger(__name__)

def main():
    """Run the full lead scoring pipeline."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("=" * 60)
    logger.info("  VEHICLE LEAD SCORING SYSTEM")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    logger.info("\n[1/5] Loading data...")
    train_df = load_train_data()
    test_df = load_test_data()
    calls_df = load_deal_calls()
    logger.info(f"  Train: {train_df.shape}, Test: {test_df.shape}, Calls: {calls_df.shape}")

    # ------------------------------------------------------------------
    # Step 2: Feature engineering
    # ------------------------------------------------------------------
    logger.info("\n[2/5] Engineering features...")
    train_featured = engineer_features(train_df, calls_df)
    test_featured = engineer_features(test_df, calls_df)

    # ------------------------------------------------------------------
    # Step 3: Prepare train/validation split
    # ------------------------------------------------------------------
    logger.info("\n[3/5] Preparing train/validation split...")
    train_split, val_split = create_time_based_split(train_featured)
    logger.info(f"  Train split: {train_split.shape}, Val split: {val_split.shape}")

    # Class balance
    print_class_balance(train_split)

    # Separate features and targets
    exclude_cols = set(ID_COLUMNS) | set(LEAKAGE_COLUMNS) | {"date_created"}
    feature_cols = [c for c in train_split.columns if c not in exclude_cols]

    X_train = train_split[feature_cols]
    y_train = train_split[TARGET_COLUMNS]
    X_val = val_split[feature_cols]
    y_val = val_split[TARGET_COLUMNS]

    # ------------------------------------------------------------------
    # Step 4: Train model
    # ------------------------------------------------------------------
    logger.info("\n[4/5] Training models...")
    model = FunnelModel(model_type="lightgbm")
    model.fit(X_train, y_train, X_val, y_val)

    # Generate funnel plot
    plot_funnel_distribution(train_df, save_path=FIGURES_DIR / "funnel_distribution.png")

    # ------------------------------------------------------------------
    # Step 5: Test Predictions & Submission
    # ------------------------------------------------------------------
    logger.info("\n[5/5] Generating Test Predictions...")
    X_test = test_featured[feature_cols]
    test_probas = model.predict_proba(X_test)
    
    # Save submission
    submission = pd.DataFrame({'deal_ref': test_df['deal_ref']})
    for stage in FUNNEL_STAGES:
        submission[f"p_{stage}"] = test_probas[stage] if stage in test_probas.columns else 0.0
        
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = REPORTS_DIR / "prediction.csv"
    submission.to_csv(submission_path, index=False)
    logger.info(f"  Saved submission to: {submission_path}")
    
    # Save model
    model.save()

    logger.info("\n* Pipeline complete.")


if __name__ == "__main__":
    main()
