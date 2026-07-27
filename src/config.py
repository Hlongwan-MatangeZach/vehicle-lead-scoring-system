"""
Project configuration and constants for the Vehicle Lead Scoring System.
"""
import os
from pathlib import Path

# ============================================================================
# Directory Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output directories
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ============================================================================
# Data Files
# ============================================================================
TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"
DEAL_CALL_FILE = RAW_DATA_DIR / "deal_call.csv"

# ============================================================================
# Funnel Configuration
# ============================================================================
# Ordered funnel stages (each is a binary target column in train.csv)
FUNNEL_STAGES = ["quote", "prospect", "inspection", "accepted"]

# Target columns in train.csv
TARGET_COLUMNS = FUNNEL_STAGES

# Date columns associated with targets (potential leakage — exclude from features)
TARGET_DATE_COLUMNS = [
    "quote_date",
    "prospect_date",
    "inspection_date",
    "accepted_date",
]

# Columns that encode the outcome and must be excluded from features
LEAKAGE_COLUMNS = ["deal_status"] + TARGET_COLUMNS + TARGET_DATE_COLUMNS

# ============================================================================
# Feature Groups
# ============================================================================
# Identifier columns (not features)
ID_COLUMNS = ["deal_ref", "client_id"]

# Vehicle-related features
VEHICLE_FEATURES = ["year", "vehicle", "mileage", "colour", "condition", "service history"]

# Enquiry context features
CONTEXT_FEATURES = ["deal type", "location", "urgency", "list price", "count"]

# Marketing / attribution features
MARKETING_FEATURES = [
    "referer",
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_content",
    "utm_term",
    "ga name final",
    "ga medium final",
    "ga channel final",
]

# Temporal features (raw — will be engineered)
TEMPORAL_FEATURES = ["date_created", "date", "week", "month"]

# ============================================================================
# Model Configuration
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2  # For train/validation split
N_FOLDS = 5  # For cross-validation


