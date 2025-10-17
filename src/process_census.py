import cv2
import numpy as np
from ultralytics import YOLO
import math

class CensusTractProcessor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tile_size = 640  # YOLO input size
        self.overlap = 100    # Overlap between tiles to catch edge cases
    
    def split_image_into_tiles(self, image, tile_size=640, overlap=100):
        """Split large satellite image into overlapping tiles"""
        height, width = image.shape[:2]
        tiles = []
        positions = []  # Store tile positions for coordinate mapping
        
        # Calculate step size (tile_size - overlap)
        step = tile_size - overlap
        
        for y in range(0, height - tile_size + 1, step):
            for x in range(0, width - tile_size + 1, step):
                # Extract tile
                tile = image[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
                positions.append((x, y))  # Top-left corner position
        
        # Handle remaining edges
        # Right edge tiles
        if width % step != 0:
            x = width - tile_size
            for y in range(0, height - tile_size + 1, step):
                tile = image[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
                positions.append((x, y))
        
        # Bottom edge tiles  
        if height % step != 0:
            y = height - tile_size
            for x in range(0, width - tile_size + 1, step):
                tile = image[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
                positions.append((x, y))
        
        # Bottom-right corner
        if width % step != 0 and height % step != 0:
            x = width - tile_size
            y = height - tile_size
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            positions.append((x, y))
        
        return tiles, positions
    
    def process_tiles(self, tiles, confidence_threshold=0.85):
        """Run YOLO detection on each tile"""
        all_detections = []
        
        for i, tile in enumerate(tiles):
            # Run detection on tile
            results = self.model(tile, verbose=False)
            
            if results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    conf = box.conf[0].item()
                    
                    if conf > confidence_threshold:
                        # Get bounding box coordinates (relative to tile)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        detection = {
                            'tile_idx': i,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2],  # Tile coordinates
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                        }
                        all_detections.append(detection)
        
        return all_detections
    
    def map_to_full_image(self, detections, positions):
        """Convert tile coordinates to full image coordinates"""
        mapped_detections = []
        
        for detection in detections:
            tile_idx = detection['tile_idx']
            tile_x, tile_y = positions[tile_idx]
            
            # Convert bbox from tile coords to full image coords
            x1, y1, x2, y2 = detection['bbox']
            full_x1 = x1 + tile_x
            full_y1 = y1 + tile_y
            full_x2 = x2 + tile_x  
            full_y2 = y2 + tile_y
            
            mapped_detection = {
                'confidence': detection['confidence'],
                'bbox': [full_x1, full_y1, full_x2, full_y2],
                'center': [(full_x1 + full_x2) / 2, (full_y1 + full_y2) / 2]
            }
            mapped_detections.append(mapped_detection)
        
        return mapped_detections
    
    def remove_duplicate_detections(self, detections, iou_threshold=0.3):
        """Remove overlapping detections using Non-Maximum Suppression"""
        if len(detections) == 0:
            return []
        
        # Convert to format needed for NMS
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['confidence'] for det in detections])
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            scores.tolist(), 
            score_threshold=0.3, 
            nms_threshold=iou_threshold
        )
        
        # Filter detections
        if len(indices) > 0:
            indices = indices.flatten()
            filtered_detections = [detections[i] for i in indices]
            return filtered_detections
        else:
            return []
    
    def process_census_tract(self, image_path, output_path=None):
        """Main function to process a census tract satellite image"""
        # Load large satellite image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Processing image: {image.shape[1]}x{image.shape[0]} pixels")
        
        # Step 1: Split into tiles
        tiles, positions = self.split_image_into_tiles(image)
        print(f"Created {len(tiles)} tiles for processing")
        
        # Step 2: Run detection on each tile
        detections = self.process_tiles(tiles)
        print(f"Found {len(detections)} raw detections")
        
        # Step 3: Map back to full image coordinates
        mapped_detections = self.map_to_full_image(detections, positions)
        
        # Step 4: Remove duplicates at tile boundaries
        final_detections = self.remove_duplicate_detections(mapped_detections)
        print(f"Final count after removing duplicates: {len(final_detections)}")
        
        # Step 5: Draw results on original image
        result_image = image.copy()
        for detection in final_detections:
            x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]
            conf = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Add confidence label
            label = f"Data Center: {conf:.2f}"
            cv2.putText(result_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save result
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"Results saved to: {output_path}")
        
        return final_detections, result_image

# Example usage after training
def detect_datacenters_in_census_tract(image_path):
    """Simple function to detect data centers in a census tract image"""
    
    # Initialize processor with your trained model
    processor = CensusTractProcessor('runs/detect/train/weights/best.pt')
    
    # Process the large satellite image
    detections, result_image = processor.process_census_tract(
        image_path, 
        output_path='census_tract_detections.jpg'
    )
    
    # Print summary
    print(f"\n=== RESULTS ===")
    print(f"Data centers found: {len(detections)}")
    for i, det in enumerate(detections):
        center_x, center_y = det['center']
        conf = det['confidence']
        print(f"  {i+1}. Center: ({center_x:.0f}, {center_y:.0f}), Confidence: {conf:.3f}")
    
    return detections, result_image