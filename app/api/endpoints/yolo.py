import time
import io
from PIL import Image
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Literal
from app.utils.load_model import ModelLoader
from app.utils.process import preprocess_image, postprocess_results

router = APIRouter()
model_loader = ModelLoader()


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
            detections = postprocess_results(results, conf_threshold)
            
        elif model_format == "onnx":
            model = model_loader.load_model("onnx")
            results = model.predict(
                processed_image,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            detections = postprocess_results(results, conf_threshold)
        elif model_format == "tensorrt":
            model = model_loader.load_model("tensorrt")
            results = model.predict(
                processed_image,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            detections = postprocess_results(results, conf_threshold)
        
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
