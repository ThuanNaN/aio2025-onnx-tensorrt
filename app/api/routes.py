from fastapi import APIRouter
from app.api.endpoints.yolo import router as yolo_router

api_router = APIRouter()

# Include YOLO endpoints
api_router.include_router(yolo_router, prefix="/yolo", tags=["yolo26n"])


@api_router.get("/info")
async def get_info():
    """Get API information"""
    return {
        "api_name": "ONNX TensorRT API",
        "description": "Model inference API supporting ONNX and TensorRT formats",
        "endpoints": [
            "/api/v1/info",
            "/api/v1/yolo/predict",
            "/health"
        ]
    }
