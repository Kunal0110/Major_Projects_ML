from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "diabetic_data.csv"
PROC_CSV = ROOT / "data" / "cleaned_diabetic_data.csv"

def _load() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV, na_values="?") # tread ? as NA

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    #Filtering readmitted patients within 30 days
    df["readmittted_in_30_days"] = (df["readmitted"] == "<30").astype("int8")
    df.drop(columns="readmitted", inplace=True)

    #id cols
    df.drop(columns="encounter_id", inplace=True, errors="ignore")

    #Drop duplicates and only keep first encounters
    df.drop_duplicates(subset=["patient_nbr"], keep="first", inplace=True)

    #drop columns which have missing cols > 40%
    high_na = df.columns[df.isna().mean() > 0.4]
    df.drop(columns=high_na, inplace=True)

    #Removing unknown gender
    if "gender" in df.columns:
        df = df[df["gender"] != "Unknown/Invalid"]
    
    #Simple derived numberic feature
    df["n_comorbities"] = (df[["diag_1", "diag_2", "diag_3"]].notna().sum(axis=1).astype("int8"))

    #Tighten numeric dtypes
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].apply(pd.to_numeric, downcast = "integer")

    return df.reset_index(drop=True)

def clean_and_save() -> pd.DataFrame:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"{RAW_CSV} not found")
    df = _clean(_load())
    PROC_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC_CSV, index=False)
    print("shape= :{df.shape}")
    return df

if __name__ == "__main__":
    clean_and_save()