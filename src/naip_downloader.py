import requests
import os
from urllib.parse import urlencode
import json
import time
from pathlib import Path

class NAIPDownloader:
    def __init__(self):
        # Loudoun County, VA bounding box (verified coordinates)
        # These coordinates are based on county boundary research
        self.loudoun_bbox = {
            'west': -77.85,   # Western boundary (near WV border)
            'south': 38.95,   # Southern boundary (near Prince William County)
            'east': -77.35,   # Eastern boundary (near Fairfax County)
            'north': 39.30    # Northern boundary (Potomac River)
        }
        
        # Known landmarks for verification
        self.verification_landmarks = {
            'leesburg': {'lat': 39.1156, 'lon': -77.5636},     # County seat
            'dulles_airport': {'lat': 39.0458, 'lon': -77.4558},  # Major landmark
            'sterling': {'lat': 39.0067, 'lon': -77.4286},     # Major town
            'ashburn': {'lat': 39.0437, 'lon': -77.4874},      # Data center hub
            'purcellville': {'lat': 39.1368, 'lon': -77.7144}  # Western town
        }
        
        self.base_url = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
        
    def verify_bbox_coverage(self):
        """
        Verify that our bounding box actually covers Loudoun County landmarks
        """
        print("Verifying Loudoun County Coverage:")
        print("=" * 40)
        
        all_landmarks_covered = True
        
        for landmark, coords in self.verification_landmarks.items():
            lat, lon = coords['lat'], coords['lon']
            
            # Check if landmark is within our bounding box
            within_bounds = (
                self.loudoun_bbox['west'] <= lon <= self.loudoun_bbox['east'] and
                self.loudoun_bbox['south'] <= lat <= self.loudoun_bbox['north']
            )
            
            status = "✓ COVERED" if within_bounds else "✗ MISSING"
            print(f"{landmark.replace('_', ' ').title()}: {lat:.4f}, {lon:.4f} - {status}")
            
            if not within_bounds:
                all_landmarks_covered = False
        
        print(f"\nBounding Box: {self.loudoun_bbox}")
        print(f"All landmarks covered: {'Yes' if all_landmarks_covered else 'No'}")
        
        if not all_landmarks_covered:
            print("WARNING: Some landmarks are outside the bounding box!")
            print("Consider adjusting the coordinates.")
        
        return all_landmarks_covered
    
    def get_metadata_info(self):
        """
        Get information about what the NAIP service will return
        """
        metadata_url = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer"
        
        try:
            response = requests.get(f"{metadata_url}?f=json")
            if response.status_code == 200:
                metadata = response.json()
                
                print("NAIP Service Information:")
                print("=" * 30)
                print(f"Service Name: {metadata.get('name', 'Unknown')}")
                print(f"Description: {metadata.get('serviceDescription', 'N/A')[:100]}...")
                
                if 'extent' in metadata:
                    extent = metadata['extent']
                    print(f"Service Coverage:")
                    print(f"  X: {extent.get('xmin', 'N/A')} to {extent.get('xmax', 'N/A')}")
                    print(f"  Y: {extent.get('ymin', 'N/A')} to {extent.get('ymax', 'N/A')}")
                
                if 'pixelSizeX' in metadata:
                    pixel_size = metadata['pixelSizeX']
                    print(f"Pixel Size: {pixel_size} meters")
                
                return metadata
            else:
                print(f"Could not fetch metadata: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error fetching metadata: {e}")
            return None
        
    def get_loudoun_naip_url(self, image_format='png', size_width=4000, size_height=3000):
        """
        Generate URL to download NAIP imagery for Loudoun County
        
        Args:
            image_format: 'png', 'jpg', or 'tiff'
            size_width: Width of output image in pixels
            size_height: Height of output image in pixels
        """
        # Convert lat/lon to Web Mercator (EPSG:3857) for the API
        # Approximate conversion for the bounding box
        bbox_webmercator = self.latlon_to_webmercator_bbox(self.loudoun_bbox)
        
        params = {
            'bbox': f"{bbox_webmercator['west']},{bbox_webmercator['south']},{bbox_webmercator['east']},{bbox_webmercator['north']}",
            'bboxSR': '3857',  # Web Mercator
            'size': f"{size_width},{size_height}",
            'imageSR': '3857',
            'format': image_format,
            'pixelType': 'U8',
            'noData': '',
            'noDataInterpretation': 'esriNoDataMatchAny',
            'interpolation': 'RSP_BilinearInterpolation',
            'compression': '',
            'compressionQuality': '',
            'bandIds': '',
            'mosaicRule': '',
            'renderingRule': '',
            'f': 'image'
        }
        
        return f"{self.base_url}?{urlencode(params)}"
    
    def latlon_to_webmercator_bbox(self, bbox):
        """Convert lat/lon bounding box to Web Mercator projection"""
        import math
        
        def lat_to_y(lat):
            return math.log(math.tan(math.pi/4 + math.radians(lat)/2)) * 6378137
        
        def lon_to_x(lon):
            return math.radians(lon) * 6378137
        
        return {
            'west': lon_to_x(bbox['west']),
            'south': lat_to_y(bbox['south']),
            'east': lon_to_x(bbox['east']),
            'north': lat_to_y(bbox['north'])
        }
    
    def download_loudoun_naip(self, output_path='data/loudoun_county_naip.png', 
                             image_format='png', size=(4000, 3000)):
        """
        Download NAIP imagery for Loudoun County
        
        Args:
            output_path: Where to save the downloaded image
            image_format: Image format ('png', 'jpg', 'tiff')
            size: Tuple of (width, height) in pixels
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Check if file already exists
        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024*1024)
            print(f"File already exists ({file_size_mb:.1f} MB), skipping download: {output_path}")
            return output_path, self.loudoun_bbox
        
        # Generate download URL
        url = self.get_loudoun_naip_url(image_format, size[0], size[1])
        
        print(f"Downloading NAIP imagery for Loudoun County...")
        print(f"Bounding box: {self.loudoun_bbox}")
        print(f"Output size: {size[0]} x {size[1]} pixels")
        print(f"URL: {url}")
        
        try:
            # Download the image
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Save the image
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Successfully downloaded NAIP imagery to: {output_path}")
            print(f"File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
            
            return output_path, self.loudoun_bbox
            
        except requests.RequestException as e:
            print(f"Error downloading NAIP imagery: {e}")
            return None, None
    
    def download_naip_tiles(self, output_dir='data/naip_tiles', tile_size=(2000, 2000), overlap=0.1):
        """
        Download NAIP imagery as multiple smaller tiles for processing
        
        Args:
            output_dir: Directory to save tiles
            tile_size: Size of each tile in pixels
            overlap: Overlap between tiles (0.1 = 10% overlap)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate how to divide the county into tiles
        bbox = self.loudoun_bbox
        
        # Calculate approximate degrees per tile based on tile size and resolution
        # At 0.6m resolution, each pixel = 0.6m
        # tile_size pixels = tile_size * 0.6 meters
        meters_per_tile = tile_size[0] * 0.6
        
        # Rough conversion: 1 degree ≈ 111,000 meters
        degrees_per_tile_lat = meters_per_tile / 111000
        degrees_per_tile_lon = meters_per_tile / (111000 * abs(math.cos(math.radians(bbox['north']))))
        
        # Calculate overlap in degrees
        overlap_lat = degrees_per_tile_lat * overlap
        overlap_lon = degrees_per_tile_lon * overlap
        
        # Calculate tile grid
        step_lat = degrees_per_tile_lat - overlap_lat
        step_lon = degrees_per_tile_lon - overlap_lon
        
        tile_count = 0
        tiles_info = []
        
        lat = bbox['south']
        row = 0
        while lat < bbox['north']:
            lon = bbox['west']
            col = 0
            while lon < bbox['east']:
                # Define tile bounds
                tile_bbox = {
                    'west': lon,
                    'south': lat,
                    'east': min(lon + degrees_per_tile_lon, bbox['east']),
                    'north': min(lat + degrees_per_tile_lat, bbox['north'])
                }
                
                # Download tile
                tile_filename = f"loudoun_tile_{row:02d}_{col:02d}.png"
                tile_path = os.path.join(output_dir, tile_filename)
                
                # Skip if tile already exists
                if os.path.exists(tile_path):
                    print(f"Tile {tile_count + 1}: {tile_filename} - ALREADY EXISTS, skipping")
                    tiles_info.append({
                        'filename': tile_filename,
                        'path': tile_path,
                        'bbox': tile_bbox,
                        'row': row,
                        'col': col,
                        'status': 'existing'
                    })
                    tile_count += 1
                    lon += step_lon
                    col += 1
                    continue
                
                print(f"Downloading tile {tile_count + 1}: {tile_filename}")
                
                # Temporarily modify bbox for this tile
                original_bbox = self.loudoun_bbox
                self.loudoun_bbox = tile_bbox
                
                tile_url = self.get_loudoun_naip_url('png', tile_size[0], tile_size[1])
                
                try:
                    response = requests.get(tile_url, stream=True, timeout=60)
                    response.raise_for_status()
                    
                    with open(tile_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    tiles_info.append({
                        'filename': tile_filename,
                        'path': tile_path,
                        'bbox': tile_bbox,
                        'row': row,
                        'col': col,
                        'status': 'downloaded'
                    })
                    
                    tile_count += 1
                    print(f"  Saved: {tile_path}")
                    
                    # Small delay to be respectful to the server
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"  Error downloading tile {tile_filename}: {e}")
                    tiles_info.append({
                        'filename': tile_filename,
                        'path': tile_path,
                        'bbox': tile_bbox,
                        'row': row,
                        'col': col,
                        'status': 'failed'
                    })
                
                # Restore original bbox
                self.loudoun_bbox = original_bbox
                
                lon += step_lon
                col += 1
            
            lat += step_lat
            row += 1
        
        print(f"\nDownloaded {tile_count} tiles to {output_dir}")
        
        # Save tiles metadata
        metadata_path = os.path.join(output_dir, 'tiles_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(tiles_info, f, indent=2)
        
        print(f"Tile metadata saved to: {metadata_path}")
        return tiles_info

# Example usage
if __name__ == "__main__":
    import math
    
    downloader = NAIPDownloader()
    
    print("NAIP Downloader for Loudoun County, Virginia")
    print("=" * 50)
    
    # Option 1: Download single large image
    print("\n1. Downloading single county image...")
    image_path, bbox = downloader.download_loudoun_naip(
        output_path='data/loudoun_county_naip.png',
        size=(6000, 4500)  # High resolution
    )
    
    if image_path:
        print(f"County image saved to: {image_path}")
        print(f"Image covers bounding box: {bbox}")
        print(f"At 0.6m resolution, each 640x640 patch = 384m x 384m on ground")
    
    # Option 2: Download as tiles (better for processing)
    print("\n2. Downloading as tiles...")
    tiles = downloader.download_naip_tiles(
        output_dir='data/naip_tiles',
        tile_size=(2000, 2000),  # 2000x2000 pixel tiles
        overlap=0.1  # 10% overlap
    )
    
    print(f"\nDownload complete! Found {len(tiles)} tiles.")
    print("Next steps:")
    print("1. Use naip_patch_processor.py to visualize patches")
    print("2. Run your model on the patches")
    print("3. Map results back to coordinates")