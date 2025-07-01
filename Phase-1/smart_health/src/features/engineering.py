"""
Feature engineering for the UCI Diabetes Readmission data set

• Groups ICD-9 diagnosis codes into 6 broad body-system buckets
• Adds service-utilisation counts
• Flags medication changes / number of meds
• Adds simple demographic flags
"""

from typing import Iterable
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Return a *new* DataFrame with engineered cols (keeps originals)."""
    df = df.copy()

    # Diagnosis buckets (ICD-9 codes)
    diag_cols = ["diag_1", "diag_2", "diag_3"]
    for col in diag_cols:
        codes = pd.to_numeric(df[col].str.slice(0, 3), errors="coerce").fillna(0).astype(int)

        df[f"{col}_group"] = "Other"
        df.loc[codes.between(390, 459), f"{col}_group"] = "Circulatory"
        df.loc[codes.between(250, 250), f"{col}_group"] = "Diabetes"
        df.loc[codes.between(460, 519), f"{col}_group"] = "Respiratory"
        df.loc[codes.between(800, 999), f"{col}_group"] = "Injury"
        df.loc[codes.between(710, 739), f"{col}_group"] = "Musculoskeletal"
        df.loc[df[col] == "V", f"{col}_group"] = "ExternalCauses"

    # Service utilisation
    df["service_utilisation"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )

    # Medication change / # unique meds
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone",
        "metformin-rosiglitazone", "metformin-pioglitazone",
    ]
    med_cols = [c for c in med_cols if c in df.columns]

    df["med_change_count"] = df[med_cols].isin({"Up", "Down"}).sum(axis=1)
    df["num_meds"]        = df[med_cols].ne("No").sum(axis=1)

    # Demographic flag: gender unknown
    if "gender" in df.columns:
        df["gender_unknown"] = (df["gender"] == "Unknown/Invalid").astype("int8")

    return df


class EngineerFeatures(BaseEstimator, TransformerMixin):
    """Scikit-learn wrapper so we can drop this step into a Pipeline."""

    def __init__(self, passthrough_cols: Iterable[str] | None = None):
        self.passthrough_cols = passthrough_cols

    # no-op fit – required by sklearn
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = _engineer(pd.DataFrame(X).copy())
        keep = (list(self.passthrough_cols) if self.passthrough_cols else []) + [c for c in df.columns if c not in (self.passthrough_cols or [])]
        return df[keep]
