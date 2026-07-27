"""
Data loading and cleaning for the Vehicle Lead Scoring System.
Handles reading raw CSV files, type casting, and basic data cleaning.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

from src.config import (
    TRAIN_FILE,
    TEST_FILE,
    DEAL_CALL_FILE,
    TARGET_COLUMNS,
    TARGET_DATE_COLUMNS,
    LEAKAGE_COLUMNS,
    ID_COLUMNS,
)


def load_train_data(filepath: Path = TRAIN_FILE) -> pd.DataFrame:
    """Load and clean the training dataset."""
    df = pd.read_csv(filepath)
    df = _clean_common_columns(df)
    
    # Parse target-related dates for train data
    for col in TARGET_DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            
    return df


def load_test_data(filepath: Path = TEST_FILE) -> pd.DataFrame:
    """Load and clean the test dataset."""
    df = pd.read_csv(filepath)
    df = _clean_common_columns(df)
    return df


def load_deal_calls(filepath: Path = DEAL_CALL_FILE) -> pd.DataFrame:
    """Load and clean the deal call log dataset."""
    df = pd.read_csv(filepath)
    df["date_created"] = pd.to_datetime(df["date_created"], errors="coerce")
    df["call_back_date"] = pd.to_datetime(df["call_back_date"], errors="coerce")
    
    # Clean strings
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()
        
    return df


def _clean_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning steps common to both train and test datasets."""
    # Parse datetime
    df["date_created"] = pd.to_datetime(df["date_created"], errors="coerce")

    # Ensure numeric columns
    for col in ["year", "mileage", "list price", "count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Standardise string columns to lowercase + stripped
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()

    return df


def get_feature_columns(df: pd.DataFrame, is_train: bool = True) -> list:
    """Return the list of usable feature columns (excluding IDs, targets, leakage)."""
    exclude = set(ID_COLUMNS)
    if is_train:
        exclude.update(LEAKAGE_COLUMNS)
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)

    # ==================================================================
    # 1. TRAIN DATA
    # ==================================================================
    logger.info("=" * 70)
    logger.info("  LOADING TRAINING DATA")
    logger.info("=" * 70)
    train_df = load_train_data()
    logger.info(f"\n  Shape : {train_df.shape[0]:,} rows x {train_df.shape[1]} columns")
    logger.info(f"  Date range : {train_df['date_created'].min()} -> {train_df['date_created'].max()}")

    logger.info("\n  --- Target distribution (conversion rates) ---")
    for col in TARGET_COLUMNS:
        if col in train_df.columns:
            pos = train_df[col].sum()
            rate = train_df[col].mean() * 100
            logger.info(f"    {col:12s}: {pos:6,} / {len(train_df):,}  ({rate:.2f}%)")

    # ==================================================================
    # 2. TEST DATA
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  LOADING TEST DATA")
    logger.info("=" * 70)
    test_df = load_test_data()
    logger.info(f"\n  Shape : {test_df.shape[0]:,} rows x {test_df.shape[1]} columns")

    train_only = set(train_df.columns) - set(test_df.columns)
    logger.info(f"  In train but NOT test (targets/leakage): {sorted(train_only)}")

    # ==================================================================
    # 3. DEAL CALL DATA
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  LOADING DEAL CALL DATA")
    logger.info("=" * 70)
    calls_df = load_deal_calls()
    logger.info(f"\n  Shape : {calls_df.shape[0]:,} rows x {calls_df.shape[1]} columns")
    logger.info(f"  Unique deal_refs: {calls_df['deal_ref'].nunique():,}")

    overlap = set(train_df["deal_ref"]) & set(calls_df["deal_ref"])
    logger.info(f"  deal_refs overlapping with train: {len(overlap):,} / {train_df['deal_ref'].nunique():,}")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  DATA LOADING COMPLETE [OK]")
    logger.info("=" * 70)
