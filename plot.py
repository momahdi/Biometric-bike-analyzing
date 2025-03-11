import json
import folium

def plot_lap(file_path, output_file):
    with open(file_path, "r") as file:
        data = json.load(file)
        
    lap_data = data["timestamps"]
    # Extract latitude and longitude
    latitudes = [float(entry["locationData"]["latitude"]) for entry in lap_data]
    longitudes = [float(entry["locationData"]["longitude"]) for entry in lap_data]

    # Create a map centered around the starting point
    start_location = (latitudes[0], longitudes[0])
    m = folium.Map(location=start_location, zoom_start=16)

    # Add markers for each recorded location
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

    # Add the lap route to the map
    route = list(zip(latitudes, longitudes))
    folium.PolyLine(route, color="red", weight=2.5, opacity=0.8).add_to(m)

    # Mark start and end points
    folium.Marker(start_location, popup="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((latitudes[-1], longitudes[-1]), popup="End", icon=folium.Icon(color="red")).add_to(m)

    m.save(output_file)
    print(f"Map saved: {output_file}")


plot_lap("./Unsegmented/P01/2024-12-10_123330-logfile-subject-P01_B.json", "lap_1_map.html")  

