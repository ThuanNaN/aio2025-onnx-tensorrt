import os
import time
import io
import numpy as np
from PIL import Image
import onnxruntime as ort
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional, Literal, Any, Dict
from ultralytics import YOLO

router = APIRouter()


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
        """Load ONNX model"""
        if self._models["onnx"] is None:
            # Configure session options for better performance
            session_options = ort.SessionOptions()
            session_options.log_severity_level = 3  # 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self._models["onnx"] = ort.InferenceSession(
                model_path, 
                sess_options=session_options,
                providers=providers
            )
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


# Global model loader instance
model_loader = ModelLoader()


def preprocess_image(image: Image.Image, img_size: int = 640):
    """Preprocess image for YOLO inference"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize maintaining aspect ratio
    original_size = image.size
    image = image.resize((img_size, img_size))
    
    return image, original_size


def postprocess_results(results, format_type: str, conf_threshold: float = 0.25):
    """Convert inference results to standardized format"""
    detections = []
    
    if format_type in ["pytorch", "tensorrt"]:
        # Ultralytics YOLO format
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                detection = {
                    "class_id": int(box.cls),
                    "class_name": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3])
                    }
                }
                if detection["confidence"] >= conf_threshold:
                    detections.append(detection)
    
    elif format_type == "onnx":
        # ONNX format - needs custom post-processing
        # This is a placeholder - actual implementation depends on ONNX output format
        pass
    
    return detections


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_format: Literal["pytorch", "onnx", "tensorrt"] = Form("pytorch"),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    img_size: int = Form(640)
):
    """
    Perform object detection on an uploaded image using yolo26l
    
    Args:
        file: Image file to process
        model_format: Model format to use (pytorch, onnx, tensorrt)
        conf_threshold: Confidence threshold for detections
        iou_threshold: IOU threshold for NMS
        img_size: Input image size
        
    Returns:
        JSON response with detection results
    """
    try:
        # Read and validate image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get model path
        model_path = model_loader.model_paths.get(model_format, "")
        if not model_path:
            raise HTTPException(status_code=400, detail=f"Invalid model format: {model_format}")
        
        # Preprocess image
        processed_image, original_size = preprocess_image(image, img_size)
        
        # Perform inference based on format
        start_time = time.time()
        
        if model_format == "pytorch":
            model = model_loader.load_model("pytorch")
            results = model.predict(
                processed_image,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            detections = postprocess_results(results, "pytorch", conf_threshold)
            
        elif model_format == "onnx":
            model = model_loader.load_model("onnx")
            # Convert image to model input format
            img_array = np.array(processed_image).astype(np.float32) / 255.0
            img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            # Run inference
            input_name = model.get_inputs()[0].name
            outputs = model.run(None, {input_name: img_array})
            
            # Post-process outputs
            detections = postprocess_results(outputs, "onnx", conf_threshold)
            
        elif model_format == "tensorrt":
            model = model_loader.load_model("tensorrt")
            results = model.predict(
                processed_image,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            detections = postprocess_results(results, "tensorrt", conf_threshold)
        
        inference_time = time.time() - start_time
        
        return JSONResponse(content={
            "success": True,
            "model_format": model_format,
            "inference_time": round(inference_time, 4),
            "image_size": {
                "original": {"width": original_size[0], "height": original_size[1]},
                "processed": {"width": img_size, "height": img_size}
            },
            "detections_count": len(detections),
            "detections": detections,
            "parameters": {
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold
            }
        })
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found: {model_path}. Please ensure the model is downloaded."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.get("/models")
async def get_available_models():
    """Get list of available models and their status"""
    models_status = model_loader.get_models_status()
    
    return {
        "models": models_status,
        "supported_formats": ["pytorch", "onnx", "tensorrt"]
    }


@router.get("/health")
async def health_check():
    """Check YOLO endpoint health"""
    return {
        "status": "healthy",
        "endpoint": "yolo",
        "models_loaded": {
            format_name: model_loader.is_model_loaded(format_name)
            for format_name in ["pytorch", "onnx", "tensorrt"]
        }
    }
