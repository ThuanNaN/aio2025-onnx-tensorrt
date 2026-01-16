from PIL import Image

def preprocess_image(image: Image.Image, img_size: int = 640):
    """Preprocess image for YOLO inference"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize maintaining aspect ratio
    original_size = image.size
    image = image.resize((img_size, img_size))
    
    return image, original_size


def postprocess_results(results, conf_threshold: float = 0.25):
    """Convert inference results to standardized format"""
    detections = []
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
    
    return detections
