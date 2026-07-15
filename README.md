# Vehicle Lead Scoring System

A machine learning pipeline that predicts the propensity of a consumer to sell their vehicle to **The Buyer**, a South African vehicle-buying platform.

## Problem

The platform generates tens of thousands of vehicle-acquisition enquiries per month, but only a small fraction convert. A limited acquisition team cannot follow up on every enquiry with equal effort. This project builds a model to:

1. **Predict Progression** — probability of reaching each funnel stage (Quote → Prospect → Inspection → Accepted)
2. **Prioritise Queue** — rank new enquiries to maximise accepted deals within the first K calls
3. **Identify Drivers** — explain which features predict conversion and propose actionable recommendations

## Project Structure

```
vehicle-lead-scoring-system/
├── data/
│   ├── raw/                    # Original, immutable data files
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── deal_call.csv
│   └── processed/              # Cleaned and feature-engineered data
├── notebooks/                  # Jupyter notebooks for EDA and experimentation
│   └── starter_notebook.ipynb
├── src/                        # Source code for the pipeline
│   ├── __init__.py
│   ├── config.py               # Project configuration and constants
│   ├── data/                   # Data loading and cleaning
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── features/               # Feature engineering
│   │   ├── __init__.py
│   │   └── feature_engineer.py
│   ├── models/                 # Model training and prediction
│   │   ├── __init__.py
│   │   └── funnel_model.py
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       └── helpers.py
├── models/                     # Serialised trained model artefacts
├── reports/                    # Generated analysis and reports
│   └── figures/                # Generated plots and visualisations
├── requirements.txt            # Python dependencies
└── README.md
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the full pipeline
python -m src.main

# Or step-by-step
python -m src.data.data_loader      # Load and clean data
python -m src.features.feature_engineer  # Engineer features
python -m src.models.funnel_model   # Train and evaluate models
```

## Data

- **train.csv** — ~33,888 labelled enquiries with funnel stage flags
- **test.csv** — ~8,263 unlabelled enquiries for prediction
- **deal_call.csv** — ~39,008 call/follow-up records linked by `deal_ref`
