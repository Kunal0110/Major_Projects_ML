"""
Imbalanced-learn Pipeline that:
  1. Engineers features
  2. Preprocesses numerics / categoricals
  3. Balances positives with SMOTE
  4. Fits the classifier
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.features.engineering import EngineerFeatures


def _infer_column_groups(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    if "readmitted_in_30_days" in num_cols:
        num_cols.remove("readmitted_in_30_days")
    return num_cols, cat_cols


def _build_preprocessor(df: pd.DataFrame):
    num_cols, cat_cols = _infer_column_groups(df)

    num_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def make_pipeline(model, df_sample: pd.DataFrame):
    """Return a full imblearn Pipeline ready for Grid/Random search."""
    preprocess = _build_preprocessor(df_sample)

    return ImbPipeline(
        [
            ("engineer", EngineerFeatures()),
            ("pre", preprocess),
            ("smote", SMOTE(k_neighbors=5, random_state=42)),
            ("clf", model),
        ]
    )