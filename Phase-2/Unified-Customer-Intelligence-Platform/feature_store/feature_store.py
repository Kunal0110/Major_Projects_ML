import duckdb
import pandas as pd
from pathlib import Path
import yaml

from feature_store.demographic_features import build_demographic_features
from feature_store.billing_features import build_billing_features
from feature_store.usage_features import build_usage_features
from feature_store.revenue_features import build_revenue_features
from feature_store.marketing_features import build_marketing_features

Data_Silver = Path("data/silver")
Feature_DIR = Path("data/features")
Feature_DIR.mkdir(parents=True, exist_ok= True)

Catalog_Path = Path("feature_store/feature_catalog.yaml")

class FeatureStore:
    def __init__(self):
        self.conn = duckdb.connect("feature_store.db")
        self._load_catalog()

    def _load_catalog(self):
        if Catalog_Path.exists():
            with open(Catalog_Path, "r") as f:
                self.catalog = yaml.safe_load(f)
        else:
            self.catalog = {}
    
    def _save_catalog(self):
        with open(Catalog_Path, "w") as f:
            yaml.dump(self.catalog, f)
    
    # Build Feature Groups

    def build_all(self):
        print("Building demographic features")
        dem = build_demographic_features(Data_Silver)
        self._register("demographic", dem)

        print("Building billing features")
        bill = build_billing_features(Data_Silver)
        self._register("billing", bill)

        print("Building usage features")
        use = build_usage_features(Data_Silver)
        self._register("usage", use)

        print("Building revenue features")
        rev = build_revenue_features(Data_Silver)
        self._register("revenue", rev)

        print("Building marketing features")
        mkt = build_marketing_features(Data_Silver)
        self._register("marketing", mkt)

        print("All feature groups are built successfully")

    # Register and Save

    def _register(self, feature_group: str, df:pd.DataFrame):
        out_path = Feature_DIR / f"{feature_group}.parquet"
        df.to_parquet(out_path, index=False)

        #DuckDB table
        self.conn.execute(f"DROP TABLE IF EXISTS {feature_group}")
        self.conn.register(feature_group, df)

        #Updating catalog
        self.catalog[feature_group] = {
            "columns" : list(df.columns),
            "path": str(out_path)
        }
        self._save_catalog()

    # Retrieve Feature Group

    def get(self, feature_group:str):
        if feature_group not in self.catalog:
            raise ValueError(f"{feature_group} not found in catalog")
        
        path = self.catalog[feature_group]["path"]
        return pd.read_parquet(path)
    
    # Merge Feature Groups into one final gold dataset

    def build_gold_feature_table(self):
        df = self.get("demographic")

        for group in ["billing", "revenue", "usage", "marketing"]:
            fg = self.get(group)
            df =  df.merge(fg, on="customer_id", how="left")

        return df
    
if __name__ == "__main__":
    fs = FeatureStore()
    fs.build_all()
    gold = fs.build_gold_feature_table()
    gold.to_parquet("data/gold/customer_gold_master.parquet", index=False)
    print("GOLD table saved")