#!/usr/bin/env python3
"""
Traffic Light Color Detection Script
Detects traffic lights and identifies their color (red, yellow, green)
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np


def detect_arrow_direction(roi, color_mask):
    """
    Detect arrow direction in the lit region
    
    Args:
        roi: Region of interest (BGR image)
        color_mask: Binary mask of the detected color
    
    Returns:
        str: Arrow direction ('left' or 'right') or None if no arrow (circular light)
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    if w == 0 or h == 0:
        return None
    
    # Extract just the lit region
    lit_region = color_mask[y:y+h, x:x+w]
    
    if lit_region.size == 0:
        return None
    
    # Calculate aspect ratio
    aspect_ratio = w / h
    
    # If aspect ratio is close to 1 (square/circular), it's not an arrow
    if 0.8 < aspect_ratio < 1.2:
        return None
    
    # Only detect horizontal arrows (left/right)
    if aspect_ratio > 1.2:  # Wider than tall - horizontal arrow
        # Analyze horizontal mass distribution
        width_center = w // 2
        
        left_mass = np.sum(lit_region[:, :width_center]) if width_center > 0 else 0
        right_mass = np.sum(lit_region[:, width_center:]) if width_center < w else 0
        
        total_mass = left_mass + right_mass
        
        if total_mass == 0:
            return None
        
        # Calculate ratio - arrow points where there is MORE mass (the arrowhead)
        left_ratio = left_mass / total_mass if total_mass > 0 else 0.5
        
        # Arrow points to the side with more mass (arrowhead is heavier)
        if left_ratio > 0.55:
            return 'left'  # More mass on left, arrow points left
        elif left_ratio < 0.45:
            return 'right'  # More mass on right, arrow points right
    
    # Not a left/right arrow
    return None


def analyze_traffic_light_color(image, box):
    """
    Analyze the color of a detected traffic light and detect arrow direction
    Uses brightness analysis to identify which light is actually ON
    
    Args:
        image: The full image (BGR format)
        box: Bounding box coordinates [x1, y1, x2, y2]
    
    Returns:
        tuple: (color, arrow_direction) 
               - If circular light: ('green', None) -> display as "GREEN"
               - If arrow light: ('green', 'left') -> display as "GREEN LEFT"
    """
    x1, y1, x2, y2 = map(int, box)
    
    # Extract the traffic light region
    roi = image[y1:y2, x1:x2]
    
    if roi.size == 0:
        return 'unknown', None
    
    # Convert to HSV and grayscale
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Find bright regions (lit lights) - they should be significantly brighter
    _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # If no bright pixels detected, traffic light might be off or too dim
    if np.sum(bright_mask) < 100:
        return 'unknown', None
    
    # Define color ranges in HSV
    # Red (wraps around in HSV, so two ranges)
    red_lower1 = np.array([0, 70, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 70, 50])
    red_upper2 = np.array([180, 255, 255])
    
    # Yellow
    yellow_lower = np.array([15, 80, 80])
    yellow_upper = np.array([35, 255, 255])
    
    # Green
    green_lower = np.array([35, 40, 40])
    green_upper = np.array([95, 255, 255])
    
    # Check for each color only in bright regions
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_red = cv2.bitwise_and(mask_red, bright_mask)  # Only bright red regions
    
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_yellow = cv2.bitwise_and(mask_yellow, bright_mask)  # Only bright yellow regions
    
    mask_green = cv2.inRange(hsv, green_lower, green_upper)
    mask_green = cv2.bitwise_and(mask_green, bright_mask)  # Only bright green regions
    
    # Count pixels for each color (only in bright regions)
    red_pixels = np.sum(mask_red)
    yellow_pixels = np.sum(mask_yellow)
    green_pixels = np.sum(mask_green)
    
    # Determine dominant color with higher threshold
    max_pixels = max(red_pixels, yellow_pixels, green_pixels)
    
    if max_pixels < 200:  # Higher threshold - must be clearly lit
        return 'unknown', None
    
    # Determine which color and its arrow direction
    if red_pixels == max_pixels:
        arrow = detect_arrow_direction(roi, mask_red)
        return 'red', arrow
    elif yellow_pixels == max_pixels:
        arrow = detect_arrow_direction(roi, mask_yellow)
        return 'yellow', arrow
    elif green_pixels == max_pixels:
        arrow = detect_arrow_direction(roi, mask_green)
        return 'green', arrow
    
    return 'unknown', None


def detect_traffic_light_colors(model_path, image_path, output_dir='outputs/traffic_light_colors', conf_threshold=0.25):
    """
    Detect traffic lights and identify their colors
    
    Args:
        model_path: Path to YOLO11 model
        image_path: Path to image file or directory
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
    """
    # Relevant COCO classes for traffic lights
    relevant_classes = [9]  # traffic light
    
    # Load the model
    print(f"Loading YOLO11s model...")
    model = YOLO(model_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of images
    image_path = Path(image_path)
    if image_path.is_file():
        image_files = [image_path]
    else:
        image_files = list(image_path.glob('*.jpg')) + list(image_path.glob('*.jpeg')) + \
                     list(image_path.glob('*.png')) + list(image_path.glob('*.JPG')) + \
                     list(image_path.glob('*.JPEG')) + list(image_path.glob('*.PNG'))
    
    print(f"Running detection on {len(image_files)} image(s)...")
    
    # Print detection summary header
    print("\n" + "="*70)
    print("TRAFFIC LIGHT COLOR DETECTION RESULTS")
    print("="*70)
    
    for idx, img_file in enumerate(image_files):
        # Load image
        image = cv2.imread(str(img_file))
        img_height, img_width = image.shape[:2]
        
        # Run inference
        results = model.predict(
            source=str(img_file),
            conf=conf_threshold,
            classes=relevant_classes,
            verbose=False
        )
        
        print(f"\nImage {idx+1}: {img_file.name}")
        
        if len(results[0].boxes) == 0:
            print("  No traffic lights detected")
        else:
            # Collect all detections with their info
            detections = []
            for i, box in enumerate(results[0].boxes):
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                # Analyze the color and arrow direction
                color, arrow = analyze_traffic_light_color(image, xyxy)
                
                # Skip if unable to identify clearly lit signal
                if color == 'unknown':
                    continue
                
                x1, y1, x2, y2 = map(int, xyxy)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                detections.append({
                    'id': i + 1,
                    'color': color,
                    'arrow': arrow,
                    'conf': conf,
                    'bbox': xyxy,
                    'center_x': center_x,
                    'center_y': center_y
                })
            
            if not detections:
                print("  No clearly lit traffic lights detected")
                continue
            
            print(f"  Detected {len(detections)} traffic light(s):")
            
            # Find the center-most traffic light (main signal for our lane)
            img_center_x = img_width / 2
            center_light = min(detections, key=lambda d: abs(d['center_x'] - img_center_x))
            
            # Separate main signals and arrow signals
            arrow_signals = [d for d in detections if d['arrow'] is not None]
            main_signal = center_light if center_light['arrow'] is None else None
            
            # Print all detections
            for det in detections:
                is_arrow = det['arrow'] is not None
                
                if det['arrow'] is None:
                    print(f"    Light {det['id']}: {det['color'].upper()} - {det['conf']:.2%} confidence [Main Signal]")
                else:
                    print(f"    Light {det['id']}: {det['color'].upper()} {det['arrow'].upper()} - {det['conf']:.2%} confidence [Arrow Signal]")
            
            # Report traffic status (not decision)
            print(f"\n  🚦 TRAFFIC STATUS:")
            
            # Main signal status
            if main_signal:
                if main_signal['color'] == 'red':
                    print(f"     Main: RED - Straight/Left movement NOT allowed")
                elif main_signal['color'] == 'yellow':
                    print(f"     Main: YELLOW - Prepare to stop")
                elif main_signal['color'] == 'green':
                    print(f"     Main: GREEN - Straight/Left movement allowed")
            
            # Arrow signals status
            if arrow_signals:
                for arrow in arrow_signals:
                    direction = arrow['arrow'].upper()
                    if arrow['color'] == 'green':
                        print(f"     Arrow: GREEN {direction} - {direction} turn allowed")
                    elif arrow['color'] == 'red':
                        print(f"     Arrow: RED {direction} - {direction} turn NOT allowed")
                    elif arrow['color'] == 'yellow':
                        print(f"     Arrow: YELLOW {direction} - {direction} turn ending")
            
            # Draw on image
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                
                # Color for bounding box
                box_colors = {
                    'red': (0, 0, 255),
                    'yellow': (0, 255, 255),
                    'green': (0, 255, 0),
                    'unknown': (128, 128, 128)
                }
                box_color = box_colors.get(det['color'], (128, 128, 128))
                
                # Thicker box for center light
                thickness = 3 if det['id'] == center_light['id'] else 2
                cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)
                
                # Add label
                if det['arrow'] is None:
                    label = f"{det['color'].upper()}"
                else:
                    label = f"{det['color'].upper()} {det['arrow'].upper()}"
                
                if det['id'] == center_light['id']:
                    label += " [OUR LANE]"
                    
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0] + 5, y1), box_color, -1)
                cv2.putText(image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save annotated image
        output_file = output_path / img_file.name
        cv2.imwrite(str(output_file), image)
    
    print("\n" + "="*70)
    print(f"Results saved to: {output_path}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Traffic Light Color Detection using YOLO11')
    parser.add_argument('--model', type=str, default='yolo11s.pt',
                        help='Path to the YOLO11 model (will auto-download if not found)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file or directory')
    parser.add_argument('--output', type=str, default='outputs/traffic_light_colors',
                        help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (0.0-1.0)')
    
    args = parser.parse_args()
    
    detect_traffic_light_colors(
        model_path=args.model,
        image_path=args.image,
        output_dir=args.output,
        conf_threshold=args.conf
    )


if __name__ == '__main__':
    main()
