import joblib
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelSlot:
    """Holds one model and its metadata."""
    def __init__(self):
        self.model = None
        self.metadata = {}
        self.path = None
        self.loaded_at = None
        self.version = "1.0.0"


class ModelRegistry:
    def __init__(self):
        # Initialize model namespaces
        self.churn = ModelSlot()
        self.segmentation = ModelSlot()
        self.clv = ModelSlot()

        # Base directory
        self.base_dir = Path("models")

    def load_model(self, slot, model_path, model_name):
        """Load a single model with error handling"""
        try:
            if model_path.exists():
                slot.model = joblib.load(model_path)
                slot.path = model_path
                slot.loaded_at = datetime.now()
                slot.metadata = {
                    "name": model_name,
                    "size_mb": model_path.stat().st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(model_path.stat().st_mtime)
                }
                logger.info(f"✅ Loaded {model_name} model from {model_path}")
                return True
            else:
                logger.warning(f"⚠️ {model_name} model not found at {model_path}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name} model: {e}")
            return False

    def load_all(self):
        """Loads all saved models into registry with enhanced error handling."""
        logger.info("🚀 Loading ML models...")
        
        models_loaded = 0
        total_models = 3
        
        # Load models - prioritize enhanced models
        enhanced_churn = self.base_dir / "churn" / "xgb_best.pkl"
        fallback_churn = self.base_dir / "churn" / "stacking_model.pkl"
        
        if enhanced_churn.exists():
            if self.load_model(self.churn, enhanced_churn, "Enhanced Churn (XGBoost)"):
                models_loaded += 1
        elif self.load_model(self.churn, fallback_churn, "Churn (Stacking)"):
            models_loaded += 1
            
        if self.load_model(self.segmentation, self.base_dir / "segmentation" / "kmeans_model.pkl", "Segmentation"):
            models_loaded += 1
            
        if self.load_model(self.clv, self.base_dir / "clv" / "clv_model.pkl", "CLV"):
            models_loaded += 1
        
        logger.info(f"📊 Model loading complete: {models_loaded}/{total_models} models loaded")
        return self
    
    def get_status(self):
        """Get registry status"""
        return {
            "churn": {
                "loaded": self.churn.model is not None,
                "path": str(self.churn.path) if self.churn.path else None,
                "loaded_at": self.churn.loaded_at.isoformat() if self.churn.loaded_at else None,
                "metadata": self.churn.metadata
            },
            "segmentation": {
                "loaded": self.segmentation.model is not None,
                "path": str(self.segmentation.path) if self.segmentation.path else None,
                "loaded_at": self.segmentation.loaded_at.isoformat() if self.segmentation.loaded_at else None,
                "metadata": self.segmentation.metadata
            },
            "clv": {
                "loaded": self.clv.model is not None,
                "path": str(self.clv.path) if self.clv.path else None,
                "loaded_at": self.clv.loaded_at.isoformat() if self.clv.loaded_at else None,
                "metadata": self.clv.metadata
            }
        }


registry = ModelRegistry().load_all()