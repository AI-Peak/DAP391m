import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Define paths
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = MODEL_ROOT.parent.parent
DATA_DIR = PROJECT_ROOT / "data_preparation" / "processed"
OUTPUT_DIR = MODEL_ROOT / "outputs"

def validate_environment():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in ["data_encoded.csv", "data_journeys.csv"]:
        if not (DATA_DIR / f).exists():
            print(f"Error: Missing required file {DATA_DIR / f}")
            sys.exit(1)

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

def main():
    validate_environment()
    df_enc = pd.read_csv(DATA_DIR / "data_encoded.csv")
    df_jr = pd.read_csv(DATA_DIR / "data_journeys.csv")
    
    channel_cols = [col for col in df_enc.columns if col.startswith("Channel_")]
    
    print("Preparing user-level features (channels + journey length)...")
    df_user = (
        df_enc.groupby("User ID")[channel_cols]
        .max()
        .reset_index()
        .merge(df_jr[["User ID", "Converted", "N_Touchpoints"]], on="User ID", how="inner")
    )
    
    print("Fitting Channel-Plus-Length Logistic Regression...")
    # Using both Channels and N_Touchpoints as predictors
    predictors = channel_cols + ["N_Touchpoints"]
    fitted_full, metrics_full = fit_logit("channel_plus_length", df_user[predictors], df_user["Converted"])
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics_full]).round(4)
    metrics_df.to_csv(OUTPUT_DIR / "channel_plus_length_metrics.csv", index=False)
    
    # Save coefficients
    conf = fitted_full.conf_int()
    conf.columns = ["ci_low", "ci_high"]
    coefficients = pd.DataFrame({
        "coef": fitted_full.params,
        "odds_ratio": np.exp(fitted_full.params),
        "p_value": fitted_full.pvalues,
    }).round(4)
    coefficients.to_csv(OUTPUT_DIR / "channel_plus_length_coefficients.csv")
    
    print(f"Done. Outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
