#!/usr/bin/env python3
"""
Traffic Light Detection Script
Detects traffic lights using YOLOv8
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2


def detect_traffic_lights(model_path, image_path, output_dir='outputs/traffic_lights', conf_threshold=0.25):
    """
    Detect traffic lights in an image or directory of images
    
    Args:
        model_path: Path to yolov8s.pt model
        image_path: Path to image file or directory
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
    """
    # Relevant COCO classes for traffic lights
    # Class 9 = traffic light
    relevant_classes = [9]  # traffic light
    
    # Load the model
    print(f"Loading YOLO11s model...")
    model = YOLO(model_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    print(f"Running detection on {image_path}...")
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True,
        project=output_path.parent,
        name=output_path.name,
        exist_ok=True
    )
    
    # Print detection summary
    print("\n" + "="*60)
    print("TRAFFIC LIGHT DETECTION RESULTS")
    print("="*60)
    
    for i, result in enumerate(results):
        print(f"\nImage {i+1}: {result.path}")
        if len(result.boxes) == 0:
            print("  No traffic lights detected")
        else:
            print(f"  Detected {len(result.boxes)} traffic light(s):")
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]
                print(f"    - {class_name}: {conf:.2%} confidence")
    
    print("\n" + "="*60)
    print(f"Results saved to: {output_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Traffic Light Detection using YOLOv8')
    parser.add_argument('--model', type=str, default='yolo11s.pt',
                        help='Path to the YOLO11 model (will auto-download if not found)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file or directory')
    parser.add_argument('--output', type=str, default='outputs/traffic_lights',
                        help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (0.0-1.0)')
    
    args = parser.parse_args()
    
    detect_traffic_lights(
        model_path=args.model,
        image_path=args.image,
        output_dir=args.output,
        conf_threshold=args.conf
    )


if __name__ == '__main__':
    main()
