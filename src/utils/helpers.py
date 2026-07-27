"""
Utility and helper functions for the Vehicle Lead Scoring System.

Common operations used across data, feature, and model modules.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Tuple
import logging

from src.config import FIGURES_DIR, FUNNEL_STAGES

logger = logging.getLogger(__name__)


def plot_funnel_distribution(
    df: pd.DataFrame,
    stages: List[str] = FUNNEL_STAGES,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot the distribution of enquiries across funnel stages.

    Parameters
    ----------
    df : pd.DataFrame
        Training data with binary target columns.
    stages : list of str
        Funnel stage column names.
    save_path : Path, optional
        If provided, save the figure to this path.
    """
    counts = [df[stage].sum() for stage in stages if stage in df.columns]
    labels = [s.title() for s in stages if s in df.columns]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, counts, color=sns.color_palette("viridis", len(labels)))
    ax.set_title("Acquisition Funnel Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Enquiries")
    ax.set_xlabel("Funnel Stage")

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")
    plt.close(fig)


def plot_conversion_rates(
    df: pd.DataFrame,
    group_col: str,
    target_col: str = "accepted",
    top_n: int = 15,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot conversion rate by a categorical grouping column.

    Parameters
    ----------
    df : pd.DataFrame
        Data with target and grouping columns.
    group_col : str
        Column to group by.
    target_col : str
        Binary target column.
    top_n : int
        Show only the top N groups by volume.
    save_path : Path, optional
        If provided, save the figure.
    """
    if group_col not in df.columns or target_col not in df.columns:
        return

    grouped = df.groupby(group_col)[target_col].agg(["mean", "count"])
    grouped = grouped.sort_values("count", ascending=False).head(top_n)
    grouped = grouped.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(grouped.index.astype(str), grouped["mean"], color="steelblue")
    ax.set_xlabel(f"Conversion Rate ({target_col.title()})")
    ax.set_title(f"Conversion by {group_col.title()}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_time_based_split(
    df: pd.DataFrame,
    date_col: str = "date_created",
    test_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically to avoid temporal leakage.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset, must contain `date_col`.
    date_col : str
        Column to sort by.
    test_fraction : float
        Fraction of data to use as validation (latest records).

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (train_split, val_split)
    """
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_fraction))
    return df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]


def print_class_balance(df: pd.DataFrame, targets: List[str] = FUNNEL_STAGES) -> None:
    """Print the class balance for each target column."""
    logger.info("\n" + "=" * 50)
    logger.info("CLASS BALANCE REPORT")
    logger.info("=" * 50)
    for target in targets:
        if target in df.columns:
            pos = df[target].sum()
            total = len(df)
            rate = pos / total * 100
            logger.info(f"  {target:12s}: {pos:6,} / {total:,} ({rate:.2f}%)")
    logger.info("=" * 50)
