"""
Script to download YOLO26 PyTorch model
"""
from pathlib import Path
from ultralytics import YOLO

# Create models directory
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def download_yolo26():
    """Download YOLO26 PyTorch model"""
    
    print("=" * 60)
    print("Downloading YOLO26 PyTorch Model")
    print("=" * 60)
    
    pytorch_path = MODELS_DIR / "yolo26l.pt"
    
    print("\nDownloading YOLO26 PyTorch model...")
    try:
        model = YOLO(str(pytorch_path))
        print(f"✓ PyTorch model saved to: {pytorch_path}")
        size_mb = pytorch_path.stat().st_size / (1024 * 1024)
        print(f"  Size: {size_mb:.2f} MB")
    except Exception as e:
        print(f"✗ Failed to download PyTorch model: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    download_yolo26()
