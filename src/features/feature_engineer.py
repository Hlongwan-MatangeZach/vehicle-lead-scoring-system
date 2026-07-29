"""
Feature engineering for the Vehicle Lead Scoring System.
"""
import pandas as pd
import numpy as np
from typing import Optional, List
from urllib.parse import urlparse
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

class VehicleFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "vehicle" in X.columns:
            X["vehicle_make"] = X["vehicle"].str.split().str[0].str.upper()
            X["engine_size"] = X["vehicle"].str.extract(r"(\d+\.\d+)").astype(float)
            X["is_turbo"] = X["vehicle"].str.contains(r"T[SD]I|TURBO|TSI|THP|TFSI", case=False, na=False).astype(int)
            X["is_automatic"] = X["vehicle"].str.contains(
                r"A/T|AUTO|EDC|DSG|CVT|STRONIC|TIPTRONIC|GEARTRONIC|SPEEDSHIFT", case=False, na=False
            ).astype(int)
            X["is_4wd"] = X["vehicle"].str.contains(r"4WD|AWD|QUATTRO|4MATIC|XDRIVE|4X4", case=False, na=False).astype(int)
        return X

class LocationFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "location" in X.columns:
            location_split = X["location"].str.rsplit(" - ", n=1, expand=True)
            X["city"] = location_split[0] if 0 in location_split.columns else np.nan
            X["province"] = location_split[1] if 1 in location_split.columns else np.nan
        return X

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "date_created" in X.columns:
            dt = pd.to_datetime(X["date_created"], errors="coerce")
            X["enquiry_hour"] = dt.dt.hour
            X["enquiry_day_of_week"] = dt.dt.dayofweek
            X["enquiry_month"] = dt.dt.month
            X["enquiry_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
            
        if "date" in X.columns and "date_created" in X.columns:
            listing_date = pd.to_datetime(X["date"], format="mixed", dayfirst=False, errors="coerce")
            created_date = pd.to_datetime(X["date_created"], errors="coerce")
            X["days_since_listing"] = (created_date - listing_date).dt.days
        return X

class ValueFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "year" in X.columns and "date_created" in X.columns:
            dt = pd.to_datetime(X["date_created"], errors="coerce")
            X["vehicle_age"] = dt.dt.year - X["year"]

        if "list price" in X.columns and "mileage" in X.columns:
            X["price_per_km"] = X["list price"] / X["mileage"].replace(0, np.nan)

        if "list price" in X.columns:
            X["log_list_price"] = np.log1p(X["list price"])
            
        if "vehicle_age" in X.columns and "log_list_price" in X.columns:
            X["price_age_interaction"] = X["log_list_price"] * X["vehicle_age"]
        return X

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Colour
        if "colour" in X.columns:
            popular_colours = {"white", "silver", "grey", "gray"}
            X["is_popular_colour"] = X["colour"].str.lower().str.strip().isin(popular_colours).astype(int)

        # Urgency
        if "urgency" in X.columns:
            urgency_map = {"low": 1, "medium": 2, "trade in": 3, "extreme": 4}
            X["urgency_ordinal"] = X["urgency"].str.lower().str.strip().map(urgency_map)

        # Deal Type
        if "deal type" in X.columns:
            dt = X["deal type"].str.lower().str.strip()
            X["is_cash_deal"] = (dt == "cash").astype(int)
            X["is_trade_in_deal"] = dt.str.contains("trade", na=False).astype(int)
            X["is_cover_deal"] = dt.str.contains("cover", na=False).astype(int)
            X["is_getmore_deal"] = (dt == "getmore").astype(int)

        # Ordinals
        if "condition" in X.columns:
            condition_map = {"poor": 1, "fair": 2, "good": 3, "very good": 4, "excellent": 5}
            X["condition_ordinal"] = X["condition"].str.lower().map(condition_map)

        if "service history" in X.columns:
            service_map = {
                "no service history": 0, "partial service history": 1,
                "full service history": 2, "full service history with agent": 3
            }
            X["service_history_ordinal"] = X["service history"].str.lower().map(service_map)

        return X

class MarketingFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # Referer domain
        if "referer" in X.columns:
            def _parse_domain(url):
                if pd.isna(url) or str(url).strip().lower() in ("nan", ""): return "unknown"
                url = str(url).strip()
                if url.lower() == "direct": return "direct"
                try:
                    domain = urlparse(url).netloc.lower()
                    if not domain: domain = url.split("/")[0].lower()
                    if "google" in domain: return "google"
                    if "facebook" in domain or "fb.com" in domain: return "facebook"
                    if "instagram" in domain: return "instagram"
                    if "bing" in domain: return "bing"
                    if "yahoo" in domain: return "yahoo"
                    if "tiktok" in domain: return "tiktok"
                    return "other"
                except:
                    return "unknown"
            X["referer_domain"] = X["referer"].apply(_parse_domain)
            
        # Marketing channels
        if "ga channel final" in X.columns:
            X["channel_clean"] = X["ga channel final"].fillna("Unknown")
            
        # UTM fields
        utm_cols = ["utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term"]
        existing = [c for c in utm_cols if c in X.columns]
        if existing:
            has_any = pd.DataFrame()
            for c in existing:
                col_str = X[c].astype(str).str.strip().str.lower()
                has_any[c] = ~(col_str.isin(["nan", "", "none"]) | X[c].isna())
            X["has_utm"] = has_any.any(axis=1).astype(int)

        if "utm_source" in X.columns:
            source = X["utm_source"].astype(str).str.lower().str.strip()
            source_map = {"google": "google", "facebook": "facebook", "bing": "bing", "organic": "organic"}
            X["utm_source_clean"] = source.map(source_map).fillna("other")
            is_missing = X["utm_source"].isna() | (X["utm_source"].astype(str).str.strip().str.lower() == "nan")
            X.loc[is_missing, "utm_source_clean"] = "unknown"

        if "utm_medium" in X.columns:
            medium = X["utm_medium"].astype(str).str.lower().str.strip()
            X["is_paid_traffic"] = medium.isin(["cpc", "ppc", "paid", "cpm"]).astype(int)
            
        return X

class CustomerHistoryExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "client_id" in X.columns and "deal_ref" in X.columns:
            client_deal_counts = X.groupby("client_id")["deal_ref"].nunique()
            X["client_deal_count"] = X["client_id"].map(client_deal_counts)
            X["is_repeat_customer"] = (X["client_deal_count"] > 1).astype(int)
        return X

class FinalCategoricalEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cat_cols = ["vehicle_make", "city", "province", "channel_clean", "referer_domain", "utm_source_clean"]
        for col in cat_cols:
            if col in X.columns:
                X[col] = X[col].astype("category")
                
        drop_cols = [
            "vehicle", "location", "colour", "condition", "service history",
            "urgency", "deal type", "referer", "utm_source", "utm_medium",
            "utm_campaign", "utm_content", "utm_term", "ga name final",
            "ga medium final", "ga channel final", "date", "week", "month"
        ]
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")
        return X

def engineer_features(df: pd.DataFrame, deal_calls: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Legacy entry point, maintained for compatibility but rewritten using pipeline."""
    # Build pipeline
    pipeline = Pipeline([
        ('vehicle', VehicleFeatureExtractor()),
        ('location', LocationFeatureExtractor()),
        ('temporal', TemporalFeatureExtractor()),
        ('value', ValueFeatureExtractor()),
        ('categorical', CategoricalEncoder()),
        ('marketing', MarketingFeatureExtractor()),
        ('customer_history', CustomerHistoryExtractor()),
    ])
    
    logger.info("Applying core feature engineering pipeline...")
    df_transformed = pipeline.fit_transform(df)
    
    if deal_calls is not None:
        logger.info("Aggregating deal calls...")
        df_transformed = _aggregate_deal_calls(df_transformed, deal_calls)
        
    final_encoder = FinalCategoricalEncoder()
    df_transformed = final_encoder.fit_transform(df_transformed)
    
    return df_transformed

def _aggregate_deal_calls(df: pd.DataFrame, deal_calls: pd.DataFrame) -> pd.DataFrame:
    if deal_calls is None or deal_calls.empty:
        return df

    calls = deal_calls.copy()
    calls["date_created"] = pd.to_datetime(calls["date_created"], errors="coerce")
    calls["call_back_date"] = pd.to_datetime(calls["call_back_date"], errors="coerce")

    deal_dates = df.groupby("deal_ref")["date_created"].min().reset_index().rename(columns={"date_created": "deal_created"})
    calls = calls.merge(deal_dates, on="deal_ref", how="inner")
    calls = calls[calls["date_created"] <= calls["deal_created"]]

    if calls.empty:
        for col in ["n_calls", "n_callbacks", "has_comments", "n_unique_curators", "first_response_hours"]:
            df[col] = 0
        return df

    call_agg = (
        calls.groupby("deal_ref")
        .agg(
            n_calls=("deal_ref", "count"),
            n_callbacks=("call_back_type", "sum"),
            has_comments=("comments", lambda x: x.notna().sum()),
            n_unique_curators=("curator_id", "nunique"),
        )
        .reset_index()
    )

    valid_callbacks = calls[calls["call_back_date"].notna()]
    if not valid_callbacks.empty:
        first_callback = valid_callbacks.groupby("deal_ref")["call_back_date"].min().reset_index().rename(columns={"call_back_date": "first_callback_date"})
        call_agg = call_agg.merge(first_callback, on="deal_ref", how="left")
        call_agg = call_agg.merge(deal_dates, on="deal_ref", how="left")

        call_agg["first_response_hours"] = (call_agg["first_callback_date"] - call_agg["deal_created"]).dt.total_seconds() / 3600.0
        call_agg = call_agg.drop(columns=["first_callback_date", "deal_created"], errors="ignore")
    else:
        call_agg["first_response_hours"] = np.nan

    df = df.merge(call_agg, on="deal_ref", how="left")
    
    if "first_response_hours" in df.columns:
        df["has_callback"] = df["first_response_hours"].notna().astype(int)
    else:
        df["has_callback"] = 0
    
    fill_cols = ["n_calls", "n_callbacks", "has_comments", "n_unique_curators", "first_response_hours"]
    for col in fill_cols:
        if col in df.columns:
            if col == "first_response_hours":
                df[col] = df[col].fillna(-1.0)
            else:
                df[col] = df[col].fillna(0).astype(int)

    return df
