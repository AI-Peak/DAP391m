import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import rel_entr
from scipy.stats import chi2_contingency, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Constants and Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Define paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data_preparation" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "attribution model"

REQUIRED_FILES = ["data_touchpoints.csv", "data_journeys.csv", "data_encoded.csv"]

def validate_environment():
    """Ensure all required data files exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in REQUIRED_FILES:
        if not (DATA_DIR / f).exists():
            print(f"Error: Missing required file {DATA_DIR / f}")
            sys.exit(1)

def load_data():
    """Load the processed datasets."""
    df_tp = pd.read_csv(DATA_DIR / "data_touchpoints.csv")
    df_jr = pd.read_csv(DATA_DIR / "data_journeys.csv")
    df_enc = pd.read_csv(DATA_DIR / "data_encoded.csv")
    return df_tp, df_jr, df_enc

# --- RQ1: Basic Attribution ---
def run_rq1_basic(df_tp, df_jr):
    print("Running RQ1: Basic Attribution...")
    converted_journeys = df_jr[df_jr["Converted"]]
    converted_touchpoints = df_tp[df_tp["User_Converted"]]
    converted_users = int(df_jr["Converted"].sum())

    first_touch_share = (converted_journeys["First_Touch_Channel"].value_counts() / converted_users * 100).round(2)
    last_touch_share = (converted_journeys["Last_Touch_Channel"].value_counts() / converted_users * 100).round(2)
    linear_credit = converted_touchpoints.groupby("Channel")["Linear_Weight"].sum()
    linear_share = (linear_credit / linear_credit.sum() * 100).round(2)

    attribution = pd.DataFrame({
        "first_touch_pct": first_touch_share,
        "last_touch_pct": last_touch_share,
        "linear_pct": linear_share,
    }).fillna(0).sort_values("linear_pct", ascending=False)

    channel_conversion = df_tp.groupby("Channel").agg(
        touchpoints=("User ID", "count"),
        conversion_touchpoints=("Is_Conversion", "sum"),
    )
    channel_conversion["conversion_rate"] = channel_conversion["conversion_touchpoints"] / channel_conversion["touchpoints"]
    channel_conversion["conversion_rate_pct"] = channel_conversion["conversion_rate"] * 100
    
    # Save outputs
    attribution.to_csv(OUTPUT_DIR / "01_attribution_share.csv")
    channel_conversion.to_csv(OUTPUT_DIR / "01_channel_conversion_rate.csv")
    
    return attribution, channel_conversion

# --- RQ2: Logistic Regression Benchmark ---
def fit_logit(model_name, X, y):
    X_model = X.astype(float)
    y_model = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y_model, test_size=0.30, random_state=RANDOM_SEED, stratify=y_model
    )
    fitted = sm.Logit(y_train, sm.add_constant(X_train, has_constant="add")).fit(disp=0, maxiter=200)
    y_score = fitted.predict(sm.add_constant(X_test, has_constant="add"))
    return fitted, {
        "model": model_name,
        "pseudo_r2_mcfadden": float(fitted.prsquared),
        "auc_test": float(roc_auc_score(y_test, y_score)),
    }

def run_rq2_logit(df_enc, df_jr, attribution_base):
    print("Running RQ2: Logistic Regression...")
    channel_cols = [col for col in df_enc.columns if col.startswith("Channel_")]
    
    # Row-level model (row_channel)
    row_y = (df_enc["Conversion"] == "Yes")
    # Drop first column to avoid dummy variable trap
    fitted_row, metrics_row = fit_logit("row_channel", df_enc[channel_cols[1:]], row_y)
    
    df_user = (
        df_enc.groupby("User ID")[channel_cols]
        .max()
        .reset_index()
        .merge(df_jr[["User ID", "Converted", "N_Touchpoints"]], on="User ID", how="inner")
    )

    # Main Benchmark Model: channel + length
    model_obj, metrics_user = fit_logit("channel_plus_length", df_user[channel_cols + ["N_Touchpoints"]], df_user["Converted"])
    
    # Consolidated Metrics
    model_metrics = pd.DataFrame([metrics_row, metrics_user]).round(4)
    model_metrics.to_csv(OUTPUT_DIR / "02_model_metrics.csv", index=False)

    # Adjusted Coefficients for User-level model
    conf = model_obj.conf_int()
    conf.columns = ["ci_low", "ci_high"]
    adjusted_coefficients = pd.DataFrame({
        "coef": model_obj.params,
        "odds_ratio": np.exp(model_obj.params),
        "or_ci_low": np.exp(conf["ci_low"]),
        "or_ci_high": np.exp(conf["ci_high"]),
        "p_value": model_obj.pvalues,
    }).round(4)

    channel_beta = model_obj.params[channel_cols].copy()
    channel_beta.index = channel_beta.index.str.replace("Channel_", "", regex=False)
    channel_score = np.exp(channel_beta - channel_beta.max())
    logit_adjusted_share = (channel_score / channel_score.sum() * 100).round(2)
    
    # Save outputs
    adjusted_coefficients.to_csv(OUTPUT_DIR / "02_adjusted_coefficients.csv")
    logit_adjusted_share.to_frame("logit_adjusted_share_pct").to_csv(OUTPUT_DIR / "02_adjusted_channel_share.csv")
    
    return logit_adjusted_share

# --- RQ2 Extension: Markov Chain ---
def conversion_absorption_probability(P):
    absorbing_states = ["CONVERSION", "NULL"]
    transient_states = [state for state in P.index if state not in absorbing_states]
    Q = P.loc[transient_states, transient_states].to_numpy(dtype=float)
    R = P.loc[transient_states, ["CONVERSION"]].to_numpy(dtype=float)
    N = np.linalg.solve(np.eye(len(transient_states)) - Q, R)
    return float(N[transient_states.index("START"), 0])

def remove_channel_from_transition_matrix(P, channel):
    P_removed = P.copy()
    outgoing = P_removed.loc[channel].copy()
    for state in list(P_removed.index):
        if state == channel: continue
        inbound = P_removed.loc[state, channel]
        if inbound != 0:
            P_removed.loc[state, :] = P_removed.loc[state, :] + inbound * outgoing
            P_removed.loc[state, channel] = 0
    P_removed = P_removed.drop(index=channel, columns=channel)
    P_removed = P_removed.div(P_removed.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return P_removed

def run_rq2_markov(df_jr):
    print("Running RQ2 Extension: Markov Chain...")
    def journey_to_path(row):
        seq = str(row["Channel_Sequence"]).split(" -> ")
        term = "CONVERSION" if bool(row["Converted"]) else "NULL"
        return ["START"] + seq + [term]

    paths = [journey_to_path(row) for _, row in df_jr.iterrows()]
    channels = sorted(set(" -> ".join(df_jr["Channel_Sequence"]).split(" -> ")))
    states = ["START"] + channels + ["CONVERSION", "NULL"]
    
    counts = pd.DataFrame(0.0, index=states, columns=states)
    for path in paths:
        for src, tgt in zip(path[:-1], path[1:]):
            counts.loc[src, tgt] += 1
    counts.loc["CONVERSION", "CONVERSION"] = 1
    counts.loc["NULL", "NULL"] = 1
    transition_matrix = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    baseline_prob = conversion_absorption_probability(transition_matrix)
    removal_effects = []
    for c in channels:
        removed_matrix = remove_channel_from_transition_matrix(transition_matrix, c)
        prob_without = conversion_absorption_probability(removed_matrix)
        removal_effects.append({"channel": c, "removal_effect": baseline_prob - prob_without})
    
    markov_df = pd.DataFrame(removal_effects)
    markov_df["positive_removal"] = markov_df["removal_effect"].clip(lower=0)
    markov_df["markov_share_pct"] = (markov_df["positive_removal"] / markov_df["positive_removal"].sum() * 100).fillna(0)
    
    markov_df.to_csv(OUTPUT_DIR / "03_markov_attribution.csv", index=False)
    return markov_df

# --- RQ3: Simulation ---
def run_rq3_simulation(channel_conversion, attribution_base):
    print("Running RQ3: Simulation...")
    TOTAL_BUDGET = 100_000.0
    TOTAL_TOUCHPOINTS = 10_000
    COST_PER_TOUCH = TOTAL_BUDGET / TOTAL_TOUCHPOINTS
    
    channels = sorted(attribution_base.index.tolist())
    conv_rates = channel_conversion["conversion_rate"].reindex(channels)
    linear_weights = (attribution_base["linear_pct"] / attribution_base["linear_pct"].sum()).reindex(channels)
    
    def simulate(weights):
        spend = TOTAL_BUDGET * weights
        tps = spend / COST_PER_TOUCH
        convs = tps * conv_rates
        return float(convs.sum())

    w_equal = pd.Series(1/len(channels), index=channels)
    w_conv = conv_rates / conv_rates.sum()
    w_linear = linear_weights
    
    results = pd.DataFrame([
        {"scenario": "Equal Split", "conversions": simulate(w_equal)},
        {"scenario": "Conv Rate Weighted", "conversions": simulate(w_conv)},
        {"scenario": "Linear Attribution Weighted", "conversions": simulate(w_linear)},
    ])
    
    results.to_csv(OUTPUT_DIR / "04_simulation_results.csv", index=False)
    return results

def main():
    validate_environment()
    df_tp, df_jr, df_enc = load_data()
    
    attr_base, chan_conv = run_rq1_basic(df_tp, df_jr)
    logit_share = run_rq2_logit(df_enc, df_jr, attr_base)
    markov_res = run_rq2_markov(df_jr)
    sim_res = run_rq3_simulation(chan_conv, attr_base)
    
    print("\nProcessing complete. All outputs saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
