from geopy.distance import geodesic
import json
import os

def extract_laps(file_path, start_lat=59.346392, start_lon=18.072960, lap_distance_threshold=8.9):
    # Load JSON data
    with open(file_path, "r") as file:
        data = json.load(file)

    timestamps = data["timestamps"]
    laps = []
    current_lap = []
    start_point = (start_lat, start_lon)
    recording = False

    for entry in timestamps:
        lat, lon = float(entry["locationData"]["latitude"]), float(entry["locationData"]["longitude"])
        current_distance = geodesic(start_point, (lat, lon)).meters

        if current_distance < lap_distance_threshold:
            if not recording:
                recording = True  # Start recording the lap
                current_lap = [entry]
            else:
                current_lap.append(entry)
        elif recording:
            # If we have moved far enough and were recording, complete the lap
            laps.append(current_lap)
            current_lap = []  # Reset for next lap
            recording = False  # Stop recording until we return to start

    return laps


# Example usage:
file_path = "./Unsegmented/P01/2024-12-10_123330-logfile-subject-P01_B.json"  
output_dir = "./Segmented/P01"
# Ensure the directory exists
os.makedirs(output_dir, exist_ok=True)
laps_data = extract_laps(file_path)

# Save each lap as a separate JSON file
for i, lap in enumerate(laps_data[:5]):  # Limiting to first 5 laps
    file_path = os.path.join(output_dir, f"lap_{i+1}.json")
    with open(file_path, "w") as f:
        json.dump(lap, f, indent=4)


print(f"Extracted {len(laps_data)} laps and saved the first 5 laps as separate files.")