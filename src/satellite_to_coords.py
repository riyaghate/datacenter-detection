from process_census import CensusTractProcessor
import cv2

def dms_to_decimal(degrees, minutes, seconds, direction=None):
    """Convert degrees/minutes/seconds to decimal degrees"""
    decimal = degrees + minutes/60 + seconds/3600
    
    # For Southern latitudes or Western longitudes, make negative
    if direction in ['S', 'W']:
        decimal = -decimal
    
    return decimal

def pixel_to_coordinates(pixel_x, pixel_y, image_bounds, image_size):
    """Convert pixel coordinates to lat/lon"""
    img_width, img_height = image_size
    min_lat, min_lon, max_lat, max_lon = image_bounds
    
    lon = min_lon + (pixel_x / img_width) * (max_lon - min_lon)
    lat = max_lat - (pixel_y / img_height) * (max_lat - min_lat)
    
    return lat, lon

# Load your trained model
processor = CensusTractProcessor('runs/detect/datacenter_v24/weights/best.pt')

# Convert your coordinates from DMS to decimal
# Replace these with your actual corner coordinates from Google Earth Pro
north_lat = dms_to_decimal(39, 2, 2.49)    # Top edge:  39° 2'2.49"N
south_lat = dms_to_decimal(38, 57, 2.43)    # Bottom edge:   38°57'2.43"N
west_lon = dms_to_decimal(77, 32, 29.37, 'W') # Left edge:    77°32'29.37"W
east_lon = dms_to_decimal(77, 23, 2.89, 'W') # Right edge:   77°23'2.89"W

# Define image bounds
image_bounds = (south_lat, west_lon, north_lat, east_lon)

# Get image dimensions
image = cv2.imread('nova_large_area.jpg')
image_size = (image.shape[1], image.shape[0])  # (width, height)

print(f"Image bounds: {image_bounds}")
print(f"Image size: {image_size}")

# Process the image
detections, result_image = processor.process_census_tract(
    'nova_large_area.jpg',
    'nova_detected_output.jpg'
)

print(f"Found {len(detections)} data centers in the NoVA area")

# Convert detections to real coordinates
for i, detection in enumerate(detections):
    pixel_x, pixel_y = detection['center']
    lat, lon = pixel_to_coordinates(pixel_x, pixel_y, image_bounds, image_size)
    print(f"Pixel coordinates: {pixel_x}, {pixel_y}")
    print(f"Image bounds: {image_bounds}")
    print(f"Converted coordinates: {lat}, {lon}")
    confidence = detection['confidence']
    
    print(f"Data center {i+1}:")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Coordinates: {lat:.6f}, {lon:.6f}")
    print(f"  Google Maps: https://maps.google.com/maps?q={lat},{lon}")
    print()