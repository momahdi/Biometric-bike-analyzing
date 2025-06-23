import os
import json
import folium

def plot_lap(file_path, output_file):
    with open(file_path, "r") as file:
        data = json.load(file)

    lap_data = data.get('timestamps', [])
    if not lap_data:
        print(f"No timestamp data in {file_path}")
        return

    latitudes = [float(entry["locationData"]["latitude"]) for entry in lap_data]
    longitudes = [float(entry["locationData"]["longitude"]) for entry in lap_data]

    start_location = (latitudes[0], longitudes[0])
    m = folium.Map(location=start_location, zoom_start=16)

    for entry in lap_data:
        lat, lon = float(entry["locationData"]["latitude"]), float(entry["locationData"]["longitude"])
        folium.CircleMarker(
            location=(lat, lon),
            radius=3,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.6,
        ).add_to(m)

    route = list(zip(latitudes, longitudes))
    folium.PolyLine(route, color="red", weight=2.5, opacity=0.8).add_to(m)

    folium.Marker(start_location, popup="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((latitudes[-1], longitudes[-1]), popup="End", icon=folium.Icon(color="red")).add_to(m)

    m.save(output_file)
    print(f"Map saved: {output_file}")


def process_all_laps(base_dir):
    for day in ['Day1', 'Day2']:
        track_a_path = os.path.join(base_dir, day, 'Track-A')
        if not os.path.exists(track_a_path):
            continue

        for participant in os.listdir(track_a_path):
            participant_path = os.path.join(track_a_path, participant)
            if os.path.isdir(participant_path):
                for file in os.listdir(participant_path):
                    if file.endswith(".json"):
                        file_path = os.path.join(participant_path, file)
                        plot_dir = os.path.join(participant_path, "plot")
                        os.makedirs(plot_dir, exist_ok=True)

                        output_file = os.path.join(plot_dir, f"{os.path.splitext(file)[0]}_map.html")
                        plot_lap(file_path, output_file)


# Replace this with the actual base path of your data
base_directory = "./Unsegmented"
process_all_laps(base_directory)
