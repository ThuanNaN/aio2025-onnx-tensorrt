import os
from typing import Optional, Any, Dict
from ultralytics import YOLO


class ModelLoader:
    """
    Manages loading, caching, and unloading of YOLO models in different formats.
    Supports PyTorch, ONNX, and TensorRT model formats.
    """
    
    def __init__(self, model_paths: Optional[Dict[str, str]] = None):
        """
        Initialize ModelLoader with optional custom model paths.
        
        Args:
            model_paths: Dictionary mapping model format names to file paths
        """
        self.model_paths = model_paths or {
            "pytorch": "models/yolo26l.pt",
            "onnx": "models/yolo26l.onnx",
            "tensorrt": "models/yolo26l.engine"
        }
        self._models: Dict[str, Any] = {
            "pytorch": None,
            "onnx": None,
            "tensorrt": None
        }
    
    def load_all_models(self):
        """Load all available models at startup"""
        for model_type, model_path in self.model_paths.items():
            if os.path.exists(model_path):
                try:
                    self.load_model(model_type, model_path)
                    print(f"Loaded {model_type} model from {model_path}")
                except Exception as e:
                    print(f"Failed to load {model_type} model: {e}")
            else:
                print(f"Model file not found: {model_path}")
    
    def unload_all_models(self):
        """Unload all models and free resources"""
        for model_type in list(self._models.keys()):
            if self._models[model_type] is not None:
                try:
                    del self._models[model_type]
                    print(f"Unloaded {model_type} model")
                except Exception as e:
                    print(f"Error unloading {model_type} model: {e}")
                finally:
                    self._models[model_type] = None
    
    def load_model(self, model_type: str, model_path: Optional[str] = None):
        """
        Load a specific model type.
        
        Args:
            model_type: Type of model ('pytorch', 'onnx', or 'tensorrt')
            model_path: Optional custom path to model file
            
        Returns:
            Loaded model instance
        """
        path = model_path if model_path else self.model_paths.get(model_type)
        
        if not path:
            raise ValueError(f"No path configured for model type: {model_type}")
        
        if model_type == "pytorch":
            return self._load_pytorch_model(path)
        elif model_type == "onnx":
            return self._load_onnx_model(path)
        elif model_type == "tensorrt":
            return self._load_tensorrt_model(path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _load_pytorch_model(self, model_path: str):
        """Load PyTorch YOLO model"""
        if self._models["pytorch"] is None:
            self._models["pytorch"] = YOLO(model_path, task='detect')
        return self._models["pytorch"]
    
    def _load_onnx_model(self, model_path: str):
        """Load ONNX model with GPU acceleration"""
        if self._models["onnx"] is None:
            self._models["onnx"] = YOLO(model_path, task='detect')
        return self._models["onnx"]

    
    def _load_tensorrt_model(self, model_path: str):
        """Load TensorRT model"""
        if self._models["tensorrt"] is None:
            self._models["tensorrt"] = YOLO(model_path, task='detect')
        return self._models["tensorrt"]
    
    def get_model(self, model_type: str):
        """
        Get a loaded model instance.
        
        Args:
            model_type: Type of model to retrieve
            
        Returns:
            Model instance or None if not loaded
        """
        return self._models.get(model_type)
    
    def is_model_loaded(self, model_type: str) -> bool:
        """Check if a model is currently loaded"""
        return self._models.get(model_type) is not None
    
    def unload_model(self, model_type: str):
        """Unload a specific model"""
        if model_type in self._models and self._models[model_type] is not None:
            try:
                del self._models[model_type]
                self._models[model_type] = None
                print(f"Unloaded {model_type} model")
            except Exception as e:
                print(f"Error unloading {model_type} model: {e}")
    
    def get_models_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all models"""
        status = {}
        for format_name, model_path in self.model_paths.items():
            status[format_name] = {
                "path": model_path,
                "available": os.path.exists(model_path),
                "loaded": self.is_model_loaded(format_name)
            }
        return status
