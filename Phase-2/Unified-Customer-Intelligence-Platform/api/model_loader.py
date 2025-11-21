import joblib
from pathlib import Path


class ModelSlot:
    """Holds one model and its metadata."""
    def __init__(self):
        self.model = None
        self.metadata = None
        self.path = None


class ModelRegistry:
    def __init__(self):
        # Initialize model namespaces
        self.churn = ModelSlot()
        self.segmentation = ModelSlot()
        self.clv = ModelSlot()

        # Base directory
        self.base_dir = Path("models")

    def load_all(self):
        """Loads all saved models into registry."""

        # ---- CHURN MODEL ----
        churn_path = self.base_dir / "churn" / "stacking_model.pkl"
        if churn_path.exists():
            self.churn.model = joblib.load(churn_path)
            self.churn.path = churn_path
        else:
            print("[WARN] No churn model found")

        # ---- SEGMENTATION MODEL ----
        seg_path = self.base_dir / "segmentation" / "kmeans_model.pkl"
        if seg_path.exists():
            self.segmentation.model = joblib.load(seg_path)
            self.segmentation.path = seg_path
        else:
            print("[WARN] No segmentation model found")

        # ---- CLV MODEL ----
        clv_path = self.base_dir / "clv" / "clv_model.pkl"
        if clv_path.exists():
            self.clv.model = joblib.load(clv_path)
            self.clv.path = clv_path
        else:
            print("[WARN] No clv model found")

        return self


registry = ModelRegistry().load_all()