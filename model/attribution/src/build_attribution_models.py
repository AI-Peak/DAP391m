#!/usr/bin/env python3
"""
Clean Basic Attribution Models
==============================

Purpose
-------
Build the 3 basic rule-based attribution models for a marketing attribution project:

1. First-touch attribution
2. Last-touch attribution
3. Linear attribution

This script intentionally DOES NOT include:
- Logistic regression
- Markov chain
- Budget simulation
- Model evaluation visualizations

Those parts should live in separate folders/scripts. This script is only for
credit allocation from converted customer journeys.

Expected input files
--------------------
1. data_touchpoints.csv
   Required columns:
   - User ID
   - Channel
   - User_Converted or Is_Conversion / Conversion
   Optional columns:
   - Timestamp
   - Campaign
   - Linear_Weight

2. data_journeys.csv
   Required/recommended columns:
   - User ID
   - Converted
   - First_Touch_Channel
   - Last_Touch_Channel
   Optional columns:
   - First_Touch_Campaign
   - Last_Touch_Campaign
   - N_Touchpoints
   - Channel_Sequence

Main outputs
------------
- attribution_channel_summary.csv
- attribution_model_long.csv
- attribution_validation_checks.csv
- attribution_campaign_summary.csv, if campaign columns are available

Example
-------
python model/attribution/src/build_attribution_models.py \
  --touchpoints data_preparation/processed/data_touchpoints.csv \
  --journeys data_preparation/processed/data_journeys.csv \
  --output-dir model/attribution/outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def normalize_bool(series: pd.Series) -> pd.Series:
    """Convert common boolean-like values to True/False."""
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
                "y": True,
                "n": False,
                "converted": True,
                "not converted": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    """Return the first matching column name from candidates."""
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise KeyError(
            "Missing required column. Expected one of: "
            + ", ".join(candidates)
            + f". Available columns: {list(df.columns)}"
        )

    return None


def ensure_output_dir(output_dir: Path) -> None:
    """Create output directory if it does not exist."""
    output_dir.mkdir(parents=True, exist_ok=True)


def percentage(numerator: pd.Series, denominator: float) -> pd.Series:
    """Convert credit counts to percentage share."""
    if denominator == 0:
        return numerator * 0
    return numerator / denominator * 100


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

def load_input_data(touchpoints_path: Path, journeys_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load touchpoint-level and journey-level data."""
    if not touchpoints_path.exists():
        raise FileNotFoundError(f"Touchpoints file not found: {touchpoints_path}")

    if not journeys_path.exists():
        raise FileNotFoundError(f"Journeys file not found: {journeys_path}")

    touchpoints = pd.read_csv(touchpoints_path)
    journeys = pd.read_csv(journeys_path)

    return touchpoints, journeys


def prepare_journey_data(
    touchpoints: pd.DataFrame,
    journeys: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str, str, str]:
    """
    Standardize key columns and create first/last touch columns if needed.

    Returns
    -------
    touchpoints, journeys, user_col, channel_col, converted_col, user_converted_col
    """
    user_col = find_column(touchpoints, ["User ID", "User_ID", "user_id", "Customer ID", "Customer_ID"])
    journey_user_col = find_column(journeys, ["User ID", "User_ID", "user_id", "Customer ID", "Customer_ID"])
    channel_col = find_column(touchpoints, ["Channel", "channel", "Marketing_Channel"])

    if journey_user_col != user_col:
        journeys = journeys.rename(columns={journey_user_col: user_col})

    # Standardize journey-level converted target.
    converted_col = find_column(journeys, ["Converted", "User_Converted", "Conversion"], required=False)

    if converted_col is None:
        # Fall back to touchpoint-level information.
        user_converted_col_tmp = find_column(
            touchpoints,
            ["User_Converted", "Converted", "Is_Conversion", "Conversion"],
        )
        user_converted = (
            touchpoints.assign(_converted=normalize_bool(touchpoints[user_converted_col_tmp]))
            .groupby(user_col)["_converted"]
            .max()
            .reset_index()
            .rename(columns={"_converted": "Converted"})
        )
        journeys = journeys.merge(user_converted, on=user_col, how="left")
        converted_col = "Converted"
    else:
        journeys[converted_col] = normalize_bool(journeys[converted_col])

    # Standardize touchpoint-level user conversion.
    user_converted_col = find_column(
        touchpoints,
        ["User_Converted", "Converted", "Is_Conversion", "Conversion"],
        required=False,
    )

    if user_converted_col is None:
        touchpoints = touchpoints.merge(journeys[[user_col, converted_col]], on=user_col, how="left")
        user_converted_col = converted_col
    else:
        touchpoints[user_converted_col] = normalize_bool(touchpoints[user_converted_col])

    # Create first/last touch columns if not already present in journeys.
    if "First_Touch_Channel" not in journeys.columns or "Last_Touch_Channel" not in journeys.columns:
        sort_cols = [user_col]
        timestamp_col = find_column(touchpoints, ["Timestamp", "timestamp", "Date", "Datetime"], required=False)

        tp_sorted = touchpoints.copy()
        if timestamp_col is not None:
            tp_sorted[timestamp_col] = pd.to_datetime(tp_sorted[timestamp_col], errors="coerce")
            sort_cols.append(timestamp_col)
        elif "Touchpoint_Rank" in touchpoints.columns:
            sort_cols.append("Touchpoint_Rank")

        tp_sorted = tp_sorted.sort_values(sort_cols)

        first_last = (
            tp_sorted.groupby(user_col)
            .agg(
                First_Touch_Channel=(channel_col, "first"),
                Last_Touch_Channel=(channel_col, "last"),
            )
            .reset_index()
        )

        drop_cols = [c for c in ["First_Touch_Channel", "Last_Touch_Channel"] if c in journeys.columns]
        journeys = journeys.drop(columns=drop_cols).merge(first_last, on=user_col, how="left")

    return touchpoints, journeys, user_col, channel_col, converted_col, user_converted_col


# ---------------------------------------------------------------------------
# Attribution logic
# ---------------------------------------------------------------------------

def compute_first_touch(journeys: pd.DataFrame, converted_col: str) -> pd.DataFrame:
    """Compute first-touch attribution credit by channel."""
    converted_journeys = journeys[journeys[converted_col]].copy()

    result = (
        converted_journeys["First_Touch_Channel"]
        .value_counts(dropna=False)
        .rename_axis("Channel")
        .reset_index(name="First_Touch_Credit")
    )

    return result


def compute_last_touch(journeys: pd.DataFrame, converted_col: str) -> pd.DataFrame:
    """Compute last-touch attribution credit by channel."""
    converted_journeys = journeys[journeys[converted_col]].copy()

    result = (
        converted_journeys["Last_Touch_Channel"]
        .value_counts(dropna=False)
        .rename_axis("Channel")
        .reset_index(name="Last_Touch_Credit")
    )

    return result


def compute_linear_touchpoint_credit(
    touchpoints: pd.DataFrame,
    user_col: str,
    channel_col: str,
    user_converted_col: str,
) -> pd.DataFrame:
    """
    Compute linear attribution credit by channel.

    If Linear_Weight already exists, use it.
    Otherwise, compute 1 / number_of_touchpoints for each converted user's touchpoint.

    Note
    ----
    This is touchpoint-level linear attribution. If a user sees the same channel
    multiple times, each exposure gets its own share of credit. This is standard
    for touchpoint-based linear attribution.
    """
    converted_touchpoints = touchpoints[touchpoints[user_converted_col]].copy()

    if converted_touchpoints.empty:
        return pd.DataFrame(columns=["Channel", "Linear_Credit"])

    if "Linear_Weight" not in converted_touchpoints.columns:
        converted_touchpoints["_n_touchpoints"] = converted_touchpoints.groupby(user_col)[channel_col].transform("count")
        converted_touchpoints["Linear_Weight"] = 1 / converted_touchpoints["_n_touchpoints"]

    result = (
        converted_touchpoints.groupby(channel_col, dropna=False)["Linear_Weight"]
        .sum()
        .reset_index()
        .rename(columns={channel_col: "Channel", "Linear_Weight": "Linear_Credit"})
    )

    return result


def build_channel_summary(
    first_touch: pd.DataFrame,
    last_touch: pd.DataFrame,
    linear: pd.DataFrame,
    total_converted_users: int,
) -> pd.DataFrame:
    """Combine first-touch, last-touch and linear results into one summary table."""
    summary = (
        first_touch.merge(last_touch, on="Channel", how="outer")
        .merge(linear, on="Channel", how="outer")
        .fillna(0)
    )

    credit_cols = ["First_Touch_Credit", "Last_Touch_Credit", "Linear_Credit"]

    for col in credit_cols:
        summary[col] = summary[col].astype(float)

    summary["First_Touch_Share"] = percentage(summary["First_Touch_Credit"], total_converted_users)
    summary["Last_Touch_Share"] = percentage(summary["Last_Touch_Credit"], total_converted_users)
    summary["Linear_Share"] = percentage(summary["Linear_Credit"], total_converted_users)

    ordered_cols = [
        "Channel",
        "First_Touch_Credit",
        "First_Touch_Share",
        "Last_Touch_Credit",
        "Last_Touch_Share",
        "Linear_Credit",
        "Linear_Share",
    ]

    summary = summary[ordered_cols].sort_values("Linear_Share", ascending=False).reset_index(drop=True)

    return summary


def build_long_format(channel_summary: pd.DataFrame) -> pd.DataFrame:
    """Create tidy long-format output for plotting and comparison."""
    records = []

    for _, row in channel_summary.iterrows():
        channel = row["Channel"]

        records.extend(
            [
                {
                    "Model": "First-touch",
                    "Channel": channel,
                    "Credit": row["First_Touch_Credit"],
                    "Share": row["First_Touch_Share"],
                },
                {
                    "Model": "Last-touch",
                    "Channel": channel,
                    "Credit": row["Last_Touch_Credit"],
                    "Share": row["Last_Touch_Share"],
                },
                {
                    "Model": "Linear",
                    "Channel": channel,
                    "Credit": row["Linear_Credit"],
                    "Share": row["Linear_Share"],
                },
            ]
        )

    return pd.DataFrame(records)


def build_validation_checks(
    channel_summary: pd.DataFrame,
    total_converted_users: int,
) -> pd.DataFrame:
    """Validate that each attribution model distributes exactly one credit per converted user."""
    checks = [
        {
            "Check": "Converted users",
            "Expected": total_converted_users,
            "Actual": total_converted_users,
            "Difference": 0,
            "Passed": True,
        }
    ]

    model_credit_map = {
        "First-touch total credit": "First_Touch_Credit",
        "Last-touch total credit": "Last_Touch_Credit",
        "Linear total credit": "Linear_Credit",
    }

    for check_name, credit_col in model_credit_map.items():
        actual = float(channel_summary[credit_col].sum())
        diff = actual - float(total_converted_users)

        checks.append(
            {
                "Check": check_name,
                "Expected": float(total_converted_users),
                "Actual": actual,
                "Difference": diff,
                "Passed": abs(diff) < 1e-6,
            }
        )

    return pd.DataFrame(checks)


def compute_campaign_summary(
    touchpoints: pd.DataFrame,
    journeys: pd.DataFrame,
    user_col: str,
    converted_col: str,
    user_converted_col: str,
) -> pd.DataFrame | None:
    """
    Optional campaign-level attribution summary.

    This is kept because campaign attribution is still part of attribution logic.
    If campaign columns do not exist, the script simply skips this output.
    """
    if "Campaign" not in touchpoints.columns:
        return None

    first_campaign_col = "First_Touch_Campaign" if "First_Touch_Campaign" in journeys.columns else None
    last_campaign_col = "Last_Touch_Campaign" if "Last_Touch_Campaign" in journeys.columns else None

    converted_journeys = journeys[journeys[converted_col]].copy()
    converted_touchpoints = touchpoints[touchpoints[user_converted_col]].copy()

    pieces = []

    if first_campaign_col:
        first = (
            converted_journeys[first_campaign_col]
            .value_counts(dropna=False)
            .rename_axis("Campaign")
            .reset_index(name="First_Touch_Credit")
        )
        pieces.append(first)

    if last_campaign_col:
        last = (
            converted_journeys[last_campaign_col]
            .value_counts(dropna=False)
            .rename_axis("Campaign")
            .reset_index(name="Last_Touch_Credit")
        )
        pieces.append(last)

    if "Linear_Weight" not in converted_touchpoints.columns:
        converted_touchpoints["_n_touchpoints"] = converted_touchpoints.groupby(user_col)["Campaign"].transform("count")
        converted_touchpoints["Linear_Weight"] = 1 / converted_touchpoints["_n_touchpoints"]

    linear = (
        converted_touchpoints.groupby("Campaign", dropna=False)["Linear_Weight"]
        .sum()
        .reset_index()
        .rename(columns={"Linear_Weight": "Linear_Credit"})
    )
    pieces.append(linear)

    if not pieces:
        return None

    campaign_summary = pieces[0]
    for piece in pieces[1:]:
        campaign_summary = campaign_summary.merge(piece, on="Campaign", how="outer")

    campaign_summary = campaign_summary.fillna(0)
    total_converted_users = int(converted_journeys.shape[0])

    for col in ["First_Touch_Credit", "Last_Touch_Credit", "Linear_Credit"]:
        if col in campaign_summary.columns:
            campaign_summary[col.replace("_Credit", "_Share")] = percentage(
                campaign_summary[col].astype(float),
                total_converted_users,
            )

    return campaign_summary.sort_values(
        by="Linear_Credit" if "Linear_Credit" in campaign_summary.columns else campaign_summary.columns[1],
        ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_attribution_pipeline(
    touchpoints_path: Path,
    journeys_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Run the clean attribution pipeline and write outputs."""
    ensure_output_dir(output_dir)

    touchpoints, journeys = load_input_data(touchpoints_path, journeys_path)

    (
        touchpoints,
        journeys,
        user_col,
        channel_col,
        converted_col,
        user_converted_col,
    ) = prepare_journey_data(touchpoints, journeys)

    total_converted_users = int(journeys[converted_col].sum())

    first_touch = compute_first_touch(journeys, converted_col)
    last_touch = compute_last_touch(journeys, converted_col)
    linear = compute_linear_touchpoint_credit(
        touchpoints=touchpoints,
        user_col=user_col,
        channel_col=channel_col,
        user_converted_col=user_converted_col,
    )

    channel_summary = build_channel_summary(
        first_touch=first_touch,
        last_touch=last_touch,
        linear=linear,
        total_converted_users=total_converted_users,
    )

    long_format = build_long_format(channel_summary)
    validation_checks = build_validation_checks(channel_summary, total_converted_users)
    campaign_summary = compute_campaign_summary(
        touchpoints=touchpoints,
        journeys=journeys,
        user_col=user_col,
        converted_col=converted_col,
        user_converted_col=user_converted_col,
    )

    outputs = {
        "channel_summary": output_dir / "attribution_channel_summary.csv",
        "model_long": output_dir / "attribution_model_long.csv",
        "validation_checks": output_dir / "attribution_validation_checks.csv",
    }

    channel_summary.to_csv(outputs["channel_summary"], index=False)
    long_format.to_csv(outputs["model_long"], index=False)
    validation_checks.to_csv(outputs["validation_checks"], index=False)

    if campaign_summary is not None:
        outputs["campaign_summary"] = output_dir / "attribution_campaign_summary.csv"
        campaign_summary.to_csv(outputs["campaign_summary"], index=False)

    summary_txt = output_dir / "attribution_run_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("Clean Basic Attribution Models\n")
        f.write("=" * 32 + "\n\n")
        f.write(f"Touchpoints file: {touchpoints_path}\n")
        f.write(f"Journeys file: {journeys_path}\n")
        f.write(f"Total users: {len(journeys):,}\n")
        f.write(f"Converted users: {total_converted_users:,}\n\n")
        f.write("Validation checks:\n")
        f.write(validation_checks.to_string(index=False))
        f.write("\n\nTop channel summary by Linear_Share:\n")
        f.write(channel_summary.to_string(index=False))

    outputs["run_summary"] = summary_txt

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build clean First-touch, Last-touch and Linear attribution outputs."
    )

    parser.add_argument(
        "--touchpoints",
        type=Path,
        default=Path("data_preparation/processed/data_touchpoints.csv"),
        help="Path to touchpoint-level CSV file.",
    )

    parser.add_argument(
        "--journeys",
        type=Path,
        default=Path("data_preparation/processed/data_journeys.csv"),
        help="Path to journey-level CSV file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model/attribution/outputs"),
        help="Directory where attribution outputs will be saved.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outputs = run_attribution_pipeline(
        touchpoints_path=args.touchpoints,
        journeys_path=args.journeys,
        output_dir=args.output_dir,
    )

    print("\nClean attribution pipeline completed successfully.")
    print("Generated outputs:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
