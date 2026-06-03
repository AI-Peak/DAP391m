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
    if not (DATA_DIR / "data_encoded.csv").exists():
        print(f"Error: Missing required file {DATA_DIR / 'data_encoded.csv'}")
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
    
    channel_cols = [col for col in df_enc.columns if col.startswith("Channel_")]
    row_y = (df_enc["Conversion"] == "Yes")
    
    print("Fitting Row-Channel Logistic Regression...")
    # Drop first column to avoid dummy variable trap
    fitted_row, metrics_row = fit_logit("row_channel", df_enc[channel_cols[1:]], row_y)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics_row]).round(4)
    metrics_df.to_csv(OUTPUT_DIR / "row_channel_metrics.csv", index=False)
    
    # Save coefficients
    conf = fitted_row.conf_int()
    conf.columns = ["ci_low", "ci_high"]
    coefficients = pd.DataFrame({
        "coef": fitted_row.params,
        "odds_ratio": np.exp(fitted_row.params),
        "p_value": fitted_row.pvalues,
    }).round(4)
    coefficients.to_csv(OUTPUT_DIR / "row_channel_coefficients.csv")
    
    print(f"Done. Outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
