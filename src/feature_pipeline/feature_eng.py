from pathlib import Path
import pandas as pd
from category_encoders import TargetEncoder
from joblib import dump

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# FEATURE FUNCTIONS
def add_data_features(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month

    # place after date for readability (optional)
    df.insert(1, "year", df.pop("year"))
    df.insert(2, "quarter", df.pop("quarter"))
    df.insert(3, "month", df.pop("month"))
    return df

def frequency_encode(train: pd.DataFrame, eval: pd.DataFrame, col: str):
    # avoid shadowing builtins: use explicit names inside function
    freq_map = train[col].value_counts()
    train[f"{col}_freq"] = train[col].map(freq_map)
    eval[col + "_freq"] = eval[col].map(freq_map).fillna(0)
    return train, eval, freq_map

def target_encode(train: pd.DataFrame, eval: pd.DataFrame, col: str, target: str):
    te = TargetEncoder(cols=[col])
    encoded_col = f"{col}_encoded" if col != "city_full" else "city_full_encoded"
    # fit_transform expects X (DataFrame) and y (Series)
    train_enc = te.fit_transform(train[[col]], train[target])
    train[encoded_col] = train_enc[col]
    eval_enc = te.transform(eval[[col]])
    eval[encoded_col] = eval_enc[col]
    return train, eval, te


def drop_unused_columns(train_df: pd.DataFrame, eval_df: pd.DataFrame):
    # Safe, idempotent removal of raw and leakage-prone columns before model input.
    cols_to_drop = [
        c for c in [
            "date",
            "city_full",
            "zipcode",
            "median_sale_price",
            "address",
            "raw_address",
            "id",
            "latitude",
            "longitude",
        ]
        if c in train_df.columns
    ]
    return train_df.drop(columns=cols_to_drop, errors="ignore"), eval_df.drop(columns=cols_to_drop, errors="ignore")

def run_feature_engineering(
        in_train_path: Path | str | None = None,
        in_eval_path: Path | str | None = None,
        in_holdout_path: Path | str | None = None,
        output_dir: Path | str = PROCESSED_DIR,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Input Defaults
    if in_train_path is None:
        in_train_path = PROCESSED_DIR / "cleaning_train.csv"
    if in_eval_path is None:
        in_eval_path = PROCESSED_DIR / "cleaning_eval.csv"
    if in_holdout_path is None:
        in_holdout_path = PROCESSED_DIR / "cleaning_holdout.csv"
    
    train_df = pd.read_csv(in_train_path)
    eval_df = pd.read_csv(in_eval_path)
    holdout_df = pd.read_csv(in_holdout_path)

    print("Train date range:", train_df["date"].min(), "to", train_df["date"].max())
    print("Eval date range:", eval_df["date"].min(), "to", eval_df["date"].max())
    print("Holdout date range:", holdout_df["date"].min(), "to", holdout_df["date"].max())

    # Data features
    train_df = add_data_features(train_df)
    eval_df = add_data_features(eval_df)
    holdout_df = add_data_features(holdout_df)

    # Frequency encode zipcode (fit on train only)
    freq_map = None
    if "zipcode" in train_df.columns:
        train_df, eval_df, freq_map = frequency_encode(train_df, eval_df, "zipcode")
        holdout_df["zipcode_freq"] = holdout_df["zipcode"].map(freq_map).fillna(0)
        dump(freq_map, MODELS_DIR / "freq_encoder.pkl")     # mapping saved

    # Target encode city_full (fit on train only)
    target_encoder = None
    if "city_full" in train_df.columns:
        train_df, eval_df, target_encoder = target_encode(train_df, eval_df, "city_full", "price")
        # transform holdout using the trained encoder
        holdout_enc = target_encoder.transform(holdout_df[["city_full"]])
        holdout_df["city_full_encoded"] = holdout_enc["city_full"]
        dump(target_encoder, MODELS_DIR / "target_encoder.pkl")     # encoder saved
    
    # Drop leakage / raw categoricals
    train_df, eval_df = drop_unused_columns(train_df, eval_df)
    holdout_df, _ = drop_unused_columns(holdout_df.copy(), holdout_df.copy())

    # Save engineered Data
    # filenames aligned with notebooks that expect `feature_eng_*.csv`
    out_train_path = output_dir / "feature_eng_train.csv"
    out_eval_path = output_dir / "feature_eng_eval.csv"
    out_holdout_path = output_dir / "feature_eng_holdout.csv"
    train_df.to_csv(out_train_path, index=False)
    eval_df.to_csv(out_eval_path, index=False)
    holdout_df.to_csv(out_holdout_path, index=False)

    print("✅ Feature engineering complete.")
    print("   Train shape:", train_df.shape)
    print("   Eval  shape:", eval_df.shape)
    print("   Holdout shape:", holdout_df.shape)
    print("   Encoders saved to models/")

    return train_df, eval_df, holdout_df, freq_map, target_encoder


if __name__ == "__main__":
    run_feature_engineering()