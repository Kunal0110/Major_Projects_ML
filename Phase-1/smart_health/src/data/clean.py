from pathlib import Path
import numpy as np
import pandas as pd

ROOT      = Path(__file__).resolve().parents[2]
RAW_CSV   = ROOT / "data" / "diabetic_data.csv"
PROC_CSV  = ROOT / "data" / "cleaned_diabetic_data.csv"

def _load() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV, na_values=["?"])  # treat '?' as NA


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # keep only first encounters + within-30-day readmits flag
    df = df[df["readmitted"].isin(["<30", "NO"])]
    df["readmitted_in_30_days"] = (df["readmitted"] == "<30").astype("int8")
    df.drop(columns=["readmitted", "encounter_id"], inplace=True)

    # simple comorbidity count example
    df["n_comorbidities"] = (
        df[["diag_1", "diag_2", "diag_3"]].notna().sum(axis=1).astype("int8")
    )

    # numeric down-casting
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].apply(pd.to_numeric, downcast="integer")

    return df.reset_index(drop=True)


def clean_and_save() -> pd.DataFrame:
    if PROC_CSV.exists():
        return pd.read_csv(PROC_CSV)

    if not RAW_CSV.exists():
        raise FileNotFoundError(f"{RAW_CSV} not found")

    df = _clean(_load())
    PROC_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC_CSV, index=False)
    print(f"shape = {df.shape}")
    return df


if __name__ == "__main__":
    clean_and_save()