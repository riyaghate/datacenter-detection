import cv2
import numpy as np
import os
import json
import glob
import shutil
from pathlib import Path
from process_census import CensusTractProcessor

class NAIPPatchProcessor(CensusTractProcessor):
    """
    Simple processor that uses your existing process_census_tract method
    but organizes results into before/after folders
    """
    
    def add_patch_grid_to_detection_result(self, result_image):
        """
        Add patch grid lines to show how image was divided, including overlaps
        """
        height, width = result_image.shape[:2]
        
        # Calculate all patch positions (same logic as split_image_into_tiles)
        step = self.tile_size - self.overlap  # 540 pixels
        
        # Draw all patch rectangles to show overlaps
        for y in range(0, height - self.tile_size + 1, step):
            for x in range(0, width - self.tile_size + 1, step):
                # Draw patch rectangle
                cv2.rectangle(result_image, (x, y), (x + self.tile_size, y + self.tile_size), 
                             (255, 0, 0), 2)  # Blue rectangles for patches
        
        # Handle edge cases (patches that don't fit the regular grid)
        if width % step != 0:
            x = width - self.tile_size
            for y in range(0, height - self.tile_size + 1, step):
                cv2.rectangle(result_image, (x, y), (x + self.tile_size, y + self.tile_size), 
                             (255, 0, 0), 2)
        
        if height % step != 0:
            y = height - self.tile_size
            for x in range(0, width - self.tile_size + 1, step):
                cv2.rectangle(result_image, (x, y), (x + self.tile_size, y + self.tile_size), 
                             (255, 0, 0), 2)
        
        if width % step != 0 and height % step != 0:
            x = width - self.tile_size
            y = height - self.tile_size
            cv2.rectangle(result_image, (x, y), (x + self.tile_size, y + self.tile_size), 
                         (255, 0, 0), 2)
        
        return result_image
    
    def process_tiles_to_folders(self, tiles_directory, output_base_dir='processed_tiles', confidence_threshold=0.9):# Red lines
        
        # Draw horizontal lines  
        for y in range(0, height, step):
            cv2.line(result_image, (0, y), (width, y), (0, 0, 255), 2)  # Red lines
        
        # Save the grid visualization
        cv2.imwrite(save_path, result_image)
        print(f"Patch grid saved to: {save_path}")
        
        return result_image
    
    def process_tiles_to_folders(self, tiles_directory, output_base_dir='processed_tiles', confidence_threshold=0.9):
        """
        Process NAIP tiles using your existing method and organize into before/after folders
        """
        # Find all tile files
        tile_files = glob.glob(os.path.join(tiles_directory, '*.png'))
        if not tile_files:
            tile_files = glob.glob(os.path.join(tiles_directory, '*.jpg'))
        
        if not tile_files:
            print(f"No image files found in {tiles_directory}")
            return None
        
        # Create before/after directories
        before_dir = os.path.join(output_base_dir, 'before_processing')
        after_dir = os.path.join(output_base_dir, 'after_processing')
        os.makedirs(before_dir, exist_ok=True)
        os.makedirs(after_dir, exist_ok=True)
        
        print(f"Found {len(tile_files)} tiles to process")
        print(f"Before folder: {before_dir}")
        print(f"After folder: {after_dir}")
        print("=" * 50)
        
        all_detections = []
        processed_tiles = []
        
        for i, tile_path in enumerate(tile_files):
            tile_name = os.path.splitext(os.path.basename(tile_path))[0]
            print(f"\n[{i+1}/{len(tile_files)}] Processing: {tile_name}")
            
            # Copy original to before folder
            before_path = os.path.join(before_dir, f"{tile_name}.jpg")
            shutil.copy2(tile_path, before_path)
            
            try:
                # Run detection using your existing method
                after_path = os.path.join(after_dir, f"{tile_name}_detections.jpg")
                
                detections, result_image = self.process_census_tract(
                    tile_path,
                    after_path
                )
                
                # Add patch grid overlay to show how it was processed
                result_with_grid = self.add_patch_grid_to_detection_result(result_image)
                
                # Save the final result with both detections and grid
                cv2.imwrite(after_path, result_with_grid)
                
                # Track results
                tile_info = {
                    'tile_name': tile_name,
                    'tile_path': tile_path,
                    'before_path': before_path,
                    'after_path': after_path,
                    'detections_count': len(detections),
                    'detections': detections
                }
                
                processed_tiles.append(tile_info)
                all_detections.extend(detections)
                
                print(f"✓ {tile_name}: {len(detections)} detections found")
                print(f"  Before: {before_path}")
                print(f"  After: {after_path}")
                
            except Exception as e:
                print(f"✗ Error processing {tile_name}: {e}")
                continue
        
        # Save summary results
        summary_path = os.path.join(output_base_dir, 'processing_summary.json')
        
        summary = {
            'total_tiles_processed': len(processed_tiles),
            'total_detections': len(all_detections),
            'before_folder': before_dir,
            'after_folder': after_dir,
            'tiles': processed_tiles
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n" + "=" * 60)
        print(f"PROCESSING COMPLETE!")
        print(f"Tiles processed: {len(processed_tiles)}")
        print(f"Total detections: {len(all_detections)}")
        print(f"Before images: {before_dir}")
        print(f"After images: {after_dir}")
        print(f"Summary: {summary_path}")
        
        return summary

# Example usage
if __name__ == "__main__":
    # Initialize with your trained model
    processor = NAIPPatchProcessor('runs/detect/train/weights/best.pt')
    
    print("NAIP Patch Processor for Data Center Detection")
    print("=" * 60)
    
    # Check what we have
    tiles_dir = 'data/naip_tiles'
    single_image = 'data/loudoun_county_naip.png'
    
    if os.path.exists(tiles_dir):
        tile_files = glob.glob(os.path.join(tiles_dir, '*.png'))
        if tile_files:
            print(f"Found {len(tile_files)} tiles in {tiles_dir}")
            print("Processing all tiles...")
            
            # Process all tiles using your existing method
            summary = processor.process_tiles_to_folders(tiles_dir)
            
            if summary:
                print(f"\nFinal Summary:")
                print(f"Total tiles processed: {summary['total_tiles_processed']}")
                print(f"Total detections found: {summary['total_detections']}")
        else:
            print(f"No tile files found in {tiles_dir}")
    
    elif os.path.exists(single_image):
        print(f"Processing single county image: {single_image}")
        
        # Process single image using your existing method
        detections, result_image = processor.process_census_tract(
            single_image,
            'processed_tiles/loudoun_county_detections.jpg'
        )
        
        if detections:
            print(f"\nFinal Results:")
            print(f"Data centers detected: {len(detections)}")
    
    else:
        print("No NAIP data found!")
        print("Please run naip_downloader.py first")