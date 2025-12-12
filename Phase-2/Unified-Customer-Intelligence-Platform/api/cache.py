import redis
import json
import hashlib
from typing import Any, Optional
import os

class CacheManager:
    def __init__(self):
        self.redis_client = None
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
        except:
            print("Redis not available, using in-memory cache")
            self.cache = {}
    
    def _get_key(self, data: dict) -> str:
        """Generate cache key from input data"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                return json.loads(value) if value else None
            else:
                return self.cache.get(key)
        except:
            return None
    
    def set(self, key: str, value: Any, expire: int = 3600):
        """Set value in cache with expiration"""
        try:
            if self.redis_client:
                self.redis_client.setex(key, expire, json.dumps(value))
            else:
                self.cache[key] = value
        except:
            pass
    
    def get_prediction(self, model_type: str, data: dict) -> Optional[Any]:
        """Get cached prediction"""
        key = f"{model_type}:{self._get_key(data)}"
        return self.get(key)
    
    def cache_prediction(self, model_type: str, data: dict, result: Any):
        """Cache prediction result"""
        key = f"{model_type}:{self._get_key(data)}"
        self.set(key, result, expire=1800)  # 30 minutes

cache_manager = CacheManager()