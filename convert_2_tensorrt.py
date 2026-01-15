"""
Script to convert YOLO26 PyTorch model to TensorRT format
"""
from pathlib import Path
from ultralytics import YOLO

# Models directory
MODELS_DIR = Path("models")

def convert_to_tensorrt():
    """Convert YOLO26 PyTorch model to TensorRT format"""
    
    print("=" * 60)
    print("Converting YOLO26 to TensorRT Format")
    print("=" * 60)
    
    pytorch_path = MODELS_DIR / "yolo26l.pt"
    tensorrt_path = MODELS_DIR / "yolo26l.engine"
    
    # Check if PyTorch model exists
    if not pytorch_path.exists():
        print(f"\n✗ PyTorch model not found at: {pytorch_path}")
        print("  Please run download_yolo.py first")
        return False
    
    print("\nConverting to TensorRT format...")
    print("Note: TensorRT conversion requires NVIDIA GPU and TensorRT installed")
    try:
        model = YOLO(str(pytorch_path))
        model.export(format="engine", device=0, half=False)
        print(f"✓ TensorRT model saved to: {tensorrt_path}")
        size_mb = tensorrt_path.stat().st_size / (1024 * 1024)
        print(f"  Size: {size_mb:.2f} MB")
    except Exception as e:
        print(f"✗ Failed to export to TensorRT: {e}")
        print("  This is normal if TensorRT is not installed or GPU is not available")
        return False
    
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    convert_to_tensorrt()
