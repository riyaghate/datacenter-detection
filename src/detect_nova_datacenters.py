# In detect_nova_datacenters.py
from process_census import CensusTractProcessor
import subprocess

# Run the detection
processor = CensusTractProcessor('runs/detect/datacenter_v24/weights/best.pt')
detections, result_image = processor.process_census_tract(
    'nova_large_area.jpg',
    'nova_detected_output.jpg'
)

# Save detection results to a file for satellite_to_coords.py to read
with open('detection_results.txt', 'w') as f:
    for detection in detections:
        f.write(f"{detection['center'][0]},{detection['center'][1]},{detection['confidence']}\n")

# Call the coordinate conversion script
subprocess.run(['python', 'satellite_to_coords.py'])