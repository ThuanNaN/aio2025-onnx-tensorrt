"""
Script to convert YOLO26 PyTorch model to ONNX format
"""
from pathlib import Path
from ultralytics import YOLO

# Models directory
MODELS_DIR = Path("models")

def convert_to_onnx():
    """Convert YOLO26 PyTorch model to ONNX format"""
    
    print("=" * 60)
    print("Converting YOLO26 to ONNX Format")
    print("=" * 60)
    
    pytorch_path = MODELS_DIR / "yolo26l.pt"
    onnx_path = MODELS_DIR / "yolo26l.onnx"
    
    # Check if PyTorch model exists
    if not pytorch_path.exists():
        print(f"\n✗ PyTorch model not found at: {pytorch_path}")
        print("  Please run download_yolo.py first")
        return False
    
    print("\nConverting to ONNX format...")
    try:
        model = YOLO(str(pytorch_path))
        # Export with static shapes for better GPU performance
        model.export(
            format="onnx", 
            device=0,
            half=False,
            dynamic=False, 
            simplify=True, 
            batch=1, 
            imgsz=640,
            nms=False
        )
        print(f"✓ ONNX model saved to: {onnx_path}")
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Exported with static batch size=1, imgsz=640 for optimal GPU performance")
    except Exception as e:
        print(f"✗ Failed to export to ONNX: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    convert_to_onnx()
