# visualize_cluster_with_route.py
import os
import re
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from shapely import wkt
from shapely.geometry import LineString, MultiLineString, Point
import pickle
from datetime import datetime, timedelta
import tempfile

st.set_page_config(page_title="EV Route Viewer ", layout="wide")
st.title("🚚 EV Van Route Viewer — Show One Van (Points + Stored Route)")

ROUTES_PKL = "routes_dataset.pkl"

# -------------------------
# Helpers for geometry parsing
# -------------------------
def parse_malformed_coord_string(s: str):
    """Parse strings like '-37.6722 144.9207, -37.6721 144.9206' -> [(lat, lon), ...]."""
    pairs = re.findall(r"(-?\d+\.\d+)\s+(-?\d+\.\d+)", s)
    coords = [(float(a), float(b)) for a, b in pairs]
    if not coords:
        return []
    return coords

def normalize_coords_list(coords):
    """
    Accept coords sequence (list of (x,y)) and return list of (lat, lon).
    Handle both (lon,lat) and (lat,lon).
    """
    if not coords:
        return []
    
    # Check first coordinate to determine order
    x0, y0 = coords[0]
    
    # If x is longitude (typically larger absolute values for Melbourne)
    if abs(x0) > 90 or (abs(x0) > 180 and abs(y0) <= 90):
        # (lon, lat) -> convert to (lat, lon)
        return [(y, x) for x, y in coords]
    else:
        # Assume (lat, lon)
        return coords

def geom_to_latlon_seq(g):
    """
    Accept g which can be:
     - shapely LineString / MultiLineString
     - a WKT string
     - a malformed coord string
     - a list of coordinate tuples
    Return list of (lat, lon) sequences (list of lists), one per segment.
    """
    segs = []
    if g is None:
        return segs

    # If it's a list of strings (multiple parts)
    if isinstance(g, list):
        for part in g:
            segs.extend(geom_to_latlon_seq(part))
        return segs

    # If it's a shapely geometry object
    if isinstance(g, (LineString, MultiLineString)):
        if isinstance(g, LineString):
            segs.append(normalize_coords_list(list(g.coords)))
        else:
            for part in g.geoms:
                segs.append(normalize_coords_list(list(part.coords)))
        return segs

    # If it's a WKT string
    if isinstance(g, str):
        s = g.strip()
        # try WKT parse
        try:
            geom_obj = wkt.loads(s)
            return geom_to_latlon_seq(geom_obj)
        except Exception:
            # fallback to parse malformed coord string
            coords = parse_malformed_coord_string(s)
            if coords:
                segs.append(normalize_coords_list(coords))
            return segs

    # If it's an iterable of coordinate pairs
    try:
        coords = list(g)
        # check inner items
        if coords and isinstance(coords[0], (list, tuple)) and len(coords[0]) >= 2:
            segs.append(normalize_coords_list(coords))
            return segs
    except Exception:
        pass

    return segs

def get_route_geometry_from_data(route_data):
    """
    Extract geometry from route data in various formats.
    Handles both old and new data structures.
    """
    # Try different possible geometry fields
    geometry = route_data.get('geometry')
    if geometry is not None:
        return geometry
    
    geom_wkt = route_data.get('geom_wkt')
    if geom_wkt is not None:
        return geom_wkt
    
    geom_parts = route_data.get('geom_parts')
    if geom_parts is not None:
        return geom_parts
    
    return None

def calculate_time_schedule(total_distance_km, num_stops, start_time_str="07:00"):
    """
    Calculate time schedule for the route.
    Assumes:
    - Average speed: 40 km/h (urban driving)
    - Stop time: 15 minutes per delivery
    - Start time: 7:00 AM
    """
    start_time = datetime.strptime(start_time_str, "%H:%M")
    average_speed_kmh = 40
    stop_time_minutes = 15
    
    times = []
    current_time = start_time
    
    # Start at depot
    times.append(current_time.strftime("%H:%M"))
    
    # Calculate segment times and stops
    segment_distance = total_distance_km / (num_stops + 1)  # +1 for return to depot
    
    for i in range(num_stops + 1):  # +1 for return to depot
        # Travel time to next point
        travel_time_hours = segment_distance / average_speed_kmh
        current_time += timedelta(hours=travel_time_hours)
        
        if i < num_stops:  # Delivery stop
            times.append(current_time.strftime("%H:%M"))
            # Add stop time
            current_time += timedelta(minutes=stop_time_minutes)
        else:  # Return to depot
            times.append(current_time.strftime("%H:%M"))
    
    return times

def calculate_energy_consumption_basic(total_distance_km, efficiency_kwh_per_100km=25):
    """Calculate basic energy consumption."""
    return (total_distance_km * efficiency_kwh_per_100km) / 100

def calculate_energy_consumption_dynamic(route_segments, num_stops, initial_efficiency=40, final_efficiency=25):
    """
    Calculate dynamic energy consumption considering unloading.
    Efficiency improves from initial_efficiency to final_efficiency as vehicle unloads.
    """
    if len(route_segments) != num_stops + 1:
        # If we don't have segment distances, estimate evenly
        segment_distance = total_distance_km / (num_stops + 1)
        route_segments = [segment_distance] * (num_stops + 1)
    
    total_energy = 0
    efficiency_decrement = (initial_efficiency - final_efficiency) / num_stops
    
    for i, segment_distance in enumerate(route_segments):
        # Current efficiency based on how many stops completed
        current_efficiency = initial_efficiency - (i * efficiency_decrement)
        current_efficiency = max(final_efficiency, current_efficiency)
        
        segment_energy = (segment_distance * current_efficiency) / 100
        total_energy += segment_energy
    
    return total_energy

# -------------------------
# File Upload Section
# -------------------------
st.sidebar.header("Route Data Source")

# Option to use existing file or upload new
data_source = st.sidebar.radio("Choose data source:", 
                               ["Use existing routes_dataset.pkl", "Upload new route file"])

routes = None

if data_source == "Upload new route file":
    uploaded_file = st.sidebar.file_uploader(
        "Drag and drop your route file here", 
        type=['pkl', 'pickle'],
        help="Upload a .pkl file containing route data"
    )
    
    if uploaded_file is not None:
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Load the routes from uploaded file
            with open(tmp_path, 'rb') as f:
                routes = pickle.load(f)
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            st.sidebar.success(f"✅ Successfully loaded {len(routes)} routes from uploaded file!")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading uploaded file: {e}")
            routes = None

else:
    # Use existing file
    if not os.path.exists(ROUTES_PKL):
        st.sidebar.warning(f"⚠️ {ROUTES_PKL} not found. Please upload a file or generate routes first.")
    else:
        try:
            with open(ROUTES_PKL, 'rb') as f:
                routes = pickle.load(f)
            st.sidebar.success(f"✅ Loaded {len(routes)} routes from {ROUTES_PKL}")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {ROUTES_PKL}: {e}")
            routes = None

# -------------------------
# Check if routes are loaded
# -------------------------
if routes is None:
    st.error("No route data available. Please upload a route file or ensure routes_dataset.pkl exists.")
    st.stop()

if not routes:
    st.error("No routes found in the loaded data.")
    st.stop()

# -------------------------
# Sidebar: select cluster and display info
# -------------------------
st.sidebar.header("Route Selection")
cluster_ids = sorted([r.get("cluster_id", i) for i, r in enumerate(routes)])
selected_cluster = st.sidebar.selectbox("Select Van ID", cluster_ids)

# find the matching route
route = next((r for r in routes if r.get("cluster_id", None) == selected_cluster), None)
if route is None:
    st.error(f"Cluster {selected_cluster} not found.")
    st.stop()

points = route.get("points", []) or route.get("stores", [])
route_nodes = route.get("route_nodes", [])
distance_km = route.get("distance_km", 0)

# Fix missing names in points
for i, point in enumerate(points):
    if not point.get('name') or pd.isna(point.get('name')):
        points[i]['name'] = f"Supermarket {i+1}"

# -------------------------
# Energy Consumption Calculations
# -------------------------
st.sidebar.header("Energy Consumption")

# Basic consumption
basic_efficiency = 25  # kWh/100km
basic_energy = calculate_energy_consumption_basic(distance_km, basic_efficiency)
st.sidebar.metric("Basic Energy Consumption (Unloaded)", f"{basic_energy:.1f} kWh", 
                 f"at {basic_efficiency} kWh/100km")

# Dynamic consumption (SOC calculation)
if st.sidebar.button("🔋 Initially fully loaded?", help="Calculate dynamic energy consumption with unloading"):
    # Estimate segment distances (in real implementation, you'd have actual segment distances)
    num_segments = len(points) + 1  # to/from depot
    segment_distance = distance_km / num_segments
    route_segments = [segment_distance] * num_segments
    
    dynamic_energy = calculate_energy_consumption_dynamic(
        route_segments, 
        len(points),
        initial_efficiency=40,
        final_efficiency=25
    )
    
    energy_saving = basic_energy - dynamic_energy
    saving_percentage = (energy_saving / basic_energy) * 100 if basic_energy > 0 else 0
    
    st.sidebar.success(f"**Load-based Consumption:** {dynamic_energy:.1f} kWh")
    st.sidebar.info(f"**Energy Loss:** {energy_saving:.1f} kWh ({saving_percentage:.1f}%)")

# Route information
st.sidebar.header("Route Information")
st.sidebar.metric("Total Distance", f"{distance_km:.2f} km")
st.sidebar.metric("Number of Stops", len(points))
st.sidebar.metric("Route Nodes", len(route_nodes))

if route_nodes:
    st.sidebar.subheader("Route Sequence")
    st.sidebar.write(f"Depot → {len(route_nodes)-2} stops → Depot")

# -------------------------
# Main display
# -------------------------
st.markdown(f"### Van {selected_cluster} — {len(points)} delivery points")
st.caption(f"Route distance: {distance_km:.2f} km | Total route nodes: {len(route_nodes)}")

# Calculate time schedule
time_schedule = calculate_time_schedule(distance_km, len(points))

# Show points table with timestamps
if points:
    st.subheader("Delivery Schedule")
    
    # Create schedule dataframe
    schedule_data = []
    for i, point in enumerate(points):
        schedule_data.append({
            'Name': point.get('name', 'Supermarket'),
            'Latitude': f"{point.get('lat', 0):.6f}",
            'Longitude': f"{point.get('lon', 0):.6f}",
            'Arrival Time': time_schedule[i + 1] if i + 1 < len(time_schedule) else "N/A",
        })
    
    # Add depot entries
    schedule_df = pd.DataFrame(schedule_data)
    
    # Display the schedule
    st.dataframe(schedule_df, use_container_width=True)

# -------------------------
# Create base map
# -------------------------
depot_lat, depot_lon = -37.8065669, 144.9109551

# Use cluster center or depot as map center
if points:
    lat_avg = sum(p.get("lat", depot_lat) for p in points) / len(points)
    lon_avg = sum(p.get("lon", depot_lon) for p in points) / len(points)
    map_center = [lat_avg, lon_avg]
else:
    map_center = [depot_lat, depot_lon]

m = folium.Map(location=map_center, zoom_start=13, tiles="cartodbpositron")

# Add depot marker
folium.Marker(
    [depot_lat, depot_lon],
    tooltip="Depot (Linfox Warehouse)",
    popup="Depot",
    icon=folium.Icon(color="blue", icon="home", prefix="fa")
).add_to(m)

# Plot delivery points with numbers
for i, p in enumerate(points):
    folium.CircleMarker(
        [p.get("lat", 0), p.get("lon", 0)],
        radius=8,
        color="red",
        fill=True,
        fill_color="white",
        fill_opacity=1,
        weight=2,
        popup=f"{i+1}. {p.get('name', 'Supermarket')}",
        tooltip=f"{i+1}. {p.get('name', 'Supermarket')}",
    ).add_to(m)
    
    # Add number label
    folium.Marker(
        [p.get("lat", 0), p.get("lon", 0)],
        icon=folium.DivIcon(
            html=f'<div style="font-size: 12pt; color: black; font-weight: bold;">{i+1}</div>'
        ),
        tooltip=f"{i+1}. {p.get('name', 'Supermarket')}",
    ).add_to(m)

# Plot route geometry
route_geometry = get_route_geometry_from_data(route)
if route_geometry:
    segs = geom_to_latlon_seq(route_geometry)
    for i, seq in enumerate(segs):
        if len(seq) >= 2:
            folium.PolyLine(
                seq, 
                color="blue", 
                weight=5, 
                opacity=0.7,
                tooltip=f"Route Segment {i+1}"
            ).add_to(m)
    st.success(f"✓ Displaying {len(segs)} route segments")
else:
    st.warning("No route geometry found in data")

# Add cluster centroid marker
if points:
    folium.Marker(
        [lat_avg, lon_avg],
        tooltip=f"Van {selected_cluster} centroid",
        popup=f"Centroid of {len(points)} points",
        icon=folium.Icon(color="orange", icon="star", prefix="fa")
    ).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# -------------------------
# Display map
# -------------------------
st.subheader("Route Map")
st_folium(m, width=1200, height=700)

# -------------------------
# Debug information (collapsible)
# -------------------------
with st.expander("Debug Information"):
    st.write("Route data keys:", list(route.keys()))
    st.write("Points sample:", points[:2] if points else "No points")
    st.write("Route nodes sample:", route_nodes[:5] if route_nodes else "No route nodes")
    if route_geometry:
        st.write("Geometry type:", type(route_geometry))
        if isinstance(route_geometry, (LineString, MultiLineString)):
            st.write("Geometry bounds:", route_geometry.bounds)
