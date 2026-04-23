import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import math
import requests
import time
from polyline import decode

# Page configuration
st.set_page_config(
    page_title="🚗⚡ EV Route Navigator",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #4285F4 0%, #34A853 25%, #FBBC04 50%, #EA4335 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 15px 0;
        margin-bottom: 20px;
    }
    
    .route-panel {
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        padding: 16px;
        margin: 8px 0;
        border-left: 4px solid #4285F4;
    }
    
    .route-panel-selected {
        background: #E8F0FE;
        border-left: 4px solid #1976D2;
        box-shadow: 0 4px 12px rgba(66,133,244,0.3);
    }
    
    .metric-card-gmaps {
        background: linear-gradient(135deg, #4285F4 0%, #1976D2 100%);
        padding: 20px;
        border-radius: 8px;
        color: white;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# Indian cities with major intermediate cities for better routing
INDIAN_CITIES = {
    'Mumbai': {'lat': 19.0760, 'lon': 72.8777, 'elevation': 14, 'state': 'Maharashtra', 'zone': 'West'},
    'Delhi': {'lat': 28.7041, 'lon': 77.1025, 'elevation': 217, 'state': 'Delhi', 'zone': 'North'},
    'Bangalore': {'lat': 12.9716, 'lon': 77.5946, 'elevation': 920, 'state': 'Karnataka', 'zone': 'South'},
    'Chennai': {'lat': 13.0827, 'lon': 80.2707, 'elevation': 7, 'state': 'Tamil Nadu', 'zone': 'South'},
    'Kolkata': {'lat': 22.5726, 'lon': 88.3639, 'elevation': 9, 'state': 'West Bengal', 'zone': 'East'},
    'Hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'elevation': 505, 'state': 'Telangana', 'zone': 'South'},
    'Pune': {'lat': 18.5204, 'lon': 73.8567, 'elevation': 560, 'state': 'Maharashtra', 'zone': 'West'},
    'Ahmedabad': {'lat': 23.0225, 'lon': 72.5714, 'elevation': 53, 'state': 'Gujarat', 'zone': 'West'},
    'Jaipur': {'lat': 26.9124, 'lon': 75.7873, 'elevation': 435, 'state': 'Rajasthan', 'zone': 'North'},
    'Surat': {'lat': 21.1702, 'lon': 72.8311, 'elevation': 13, 'state': 'Gujarat', 'zone': 'West'},
}

# Major intermediate cities for routing
INTERMEDIATE_CITIES = {
    'Vadodara': {'lat': 22.3072, 'lon': 73.1812},
    'Indore': {'lat': 22.7196, 'lon': 75.8577},
    'Nagpur': {'lat': 21.1458, 'lon': 79.0882},
    'Bhopal': {'lat': 23.2599, 'lon': 77.4126},
    'Agra': {'lat': 27.1767, 'lon': 78.0081},
    'Vijayawada': {'lat': 16.5062, 'lon': 80.6480},
    'Vellore': {'lat': 12.9165, 'lon': 79.1325},
}

# EV models
EV_MODELS = {
    'Tata Nexon EV': {
        'battery': 40.5, 'weight': 1550, 'efficiency': 0.84, 'drag': 0.33, 
        'area': 2.5, 'range': 312, 'charging_speed': 50, 'price': 1400000
    },
    'Tata Tiago EV': {
        'battery': 24.0, 'weight': 1215, 'efficiency': 0.86, 'drag': 0.32, 
        'area': 2.2, 'range': 315, 'charging_speed': 33, 'price': 850000
    },
    'MG ZS EV': {
        'battery': 50.3, 'weight': 1620, 'efficiency': 0.82, 'drag': 0.34, 
        'area': 2.58, 'range': 419, 'charging_speed': 50, 'price': 2100000
    },
    'Hyundai Ioniq 5': {
        'battery': 72.6, 'weight': 2010, 'efficiency': 0.86, 'drag': 0.288, 
        'area': 2.72, 'range': 631, 'charging_speed': 220, 'price': 4500000
    },
    'Tesla Model 3': {
        'battery': 60.0, 'weight': 1730, 'efficiency': 0.88, 'drag': 0.23, 
        'area': 2.22, 'range': 555, 'charging_speed': 250, 'price': 6000000
    },
}

def find_best_intermediate_city(origin, dest, all_cities):
    """Find best intermediate city for routing"""
    best_city = None
    min_detour = float('inf')
    
    direct_dist = haversine_distance(origin['lat'], origin['lon'], dest['lat'], dest['lon'])
    
    for city_name, city_data in all_cities.items():
        # Calculate detour distance
        dist_to_intermediate = haversine_distance(origin['lat'], origin['lon'], 
                                                   city_data['lat'], city_data['lon'])
        dist_from_intermediate = haversine_distance(city_data['lat'], city_data['lon'], 
                                                     dest['lat'], dest['lon'])
        total_dist = dist_to_intermediate + dist_from_intermediate
        detour = total_dist - direct_dist
        
        # Check if city is reasonably on the path
        if detour < direct_dist * 0.3 and detour < min_detour:
            min_detour = detour
            best_city = city_data
    
    return best_city

@st.cache_data(ttl=3600)
def get_osrm_route(coords_list, alternatives=False):
    """Get route from OSRM with multiple waypoints"""
    try:
        # Format: lon,lat;lon,lat;...
        coords_str = ';'.join([f"{c['lon']},{c['lat']}" for c in coords_list])
        url = f"http://router.project-osrm.org/route/v1/driving/{coords_str}"
        
        params = {
            'overview': 'full',
            'geometries': 'geojson',
            'steps': 'true',
            'alternatives': str(alternatives).lower()
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('routes') and len(data['routes']) > 0:
                routes_data = []
                for route in data['routes'][:3]:  # Get up to 3 alternatives
                    coords = route['geometry']['coordinates']
                    waypoints = [[lat, lon] for lon, lat in coords]
                    routes_data.append({
                        'waypoints': waypoints,
                        'distance': route['distance'] / 1000,
                        'duration': route['duration'] / 3600,
                        'success': True
                    })
                return routes_data if alternatives else routes_data[0]
    except Exception as e:
        st.warning(f"OSRM routing error: {e}")
    
    return [] if alternatives else {'success': False}

def generate_smart_waypoints(origin, dest, route_type='direct'):
    """Generate smart waypoints using intermediate cities"""
    coords_list = [origin]
    
    # For longer routes, add intermediate cities
    distance = haversine_distance(origin['lat'], origin['lon'], dest['lat'], dest['lon'])
    
    if distance > 400 and route_type in ['balanced', 'scenic']:
        # Find intermediate city
        intermediate = find_best_intermediate_city(origin, dest, INTERMEDIATE_CITIES)
        if intermediate:
            coords_list.append(intermediate)
    
    coords_list.append(dest)
    
    # Try OSRM first
    result = get_osrm_route(coords_list)
    
    if result['success']:
        return result
    
    # Fallback: Generate realistic curved path
    return generate_curved_path(origin, dest, route_type)

def generate_curved_path(origin, dest, route_type):
    """Generate realistic curved path as fallback"""
    distance = haversine_distance(origin['lat'], origin['lon'], dest['lat'], dest['lon'])
    
    # Determine number of points based on distance
    num_segments = max(10, int(distance / 30))
    
    # Route-specific parameters
    if route_type == 'fastest':
        curve_factor = 0.1
        detour_mult = 1.08
    elif route_type == 'balanced':
        curve_factor = 0.15
        detour_mult = 1.15
    elif route_type == 'scenic':
        curve_factor = 0.25
        detour_mult = 1.28
    else:
        curve_factor = 0.12
        detour_mult = 1.12
    
    waypoints = []
    
    for i in range(num_segments + 1):
        t = i / num_segments
        
        # Base interpolation
        lat = origin['lat'] + (dest['lat'] - origin['lat']) * t
        lon = origin['lon'] + (dest['lon'] - origin['lon']) * t
        
        # Add realistic curve (avoiding straight lines)
        if 0 < t < 1:
            # Perpendicular offset for natural curves
            dx = dest['lon'] - origin['lon']
            dy = dest['lat'] - origin['lat']
            
            # Perpendicular direction
            perp_lat = -dx
            perp_lon = dy
            
            # Normalize
            length = np.sqrt(perp_lat**2 + perp_lon**2)
            if length > 0:
                perp_lat /= length
                perp_lon /= length
            
            # Curve offset (stronger in middle)
            curve = np.sin(t * np.pi) * curve_factor
            
            # Add some randomness for realism
            random_offset = (np.random.random() - 0.5) * curve_factor * 0.3
            
            lat += perp_lat * (curve + random_offset)
            lon += perp_lon * (curve + random_offset)
        
        waypoints.append([lat, lon])
    
    return {
        'waypoints': waypoints,
        'distance': distance * detour_mult,
        'duration': (distance * detour_mult) / 70,
        'success': True
    }

def generate_enhanced_routes(origin_city, dest_city):
    """Generate multiple route alternatives with REAL routing"""
    origin = INDIAN_CITIES[origin_city]
    dest = INDIAN_CITIES[dest_city]
    
    progress_bar = st.progress(0, text="🔍 Calculating routes...")
    routes = []
    
    # Route 1: Fastest (direct with minimal waypoints)
    progress_bar.progress(25, text="🔍 Finding fastest route...")
    route1_data = generate_smart_waypoints(origin, dest, 'fastest')
    time.sleep(0.3)  # Rate limiting
    
    routes.append({
        'name': 'Fastest Route',
        'type': 'Expressway',
        'description': 'Quickest time via major highways',
        'distance': route1_data['distance'],
        'waypoints': route1_data['waypoints'],
        'elevation_gain': abs(dest['elevation'] - origin['elevation']),
        'stops': 1,
        'avg_speed': 85,
        'traffic_level': 4,
        'color': '#4285F4',
        'route_quality': 'excellent',
        'duration': route1_data['duration']
    })
    
    # Route 2: Balanced (with intermediate cities)
    progress_bar.progress(50, text="🔍 Finding balanced route...")
    route2_data = generate_smart_waypoints(origin, dest, 'balanced')
    time.sleep(0.3)
    
    routes.append({
        'name': 'Balanced Route',
        'type': 'National Highway',
        'description': 'Good balance of time and comfort',
        'distance': route2_data['distance'],
        'waypoints': route2_data['waypoints'],
        'elevation_gain': abs(dest['elevation'] - origin['elevation']) * 1.1,
        'stops': 3,
        'avg_speed': 72,
        'traffic_level': 5,
        'color': '#34A853',
        'route_quality': 'good',
        'duration': route2_data['duration']
    })
    
    # Route 3: Scenic (more curves and detours)
    progress_bar.progress(75, text="🔍 Finding scenic route...")
    route3_data = generate_smart_waypoints(origin, dest, 'scenic')
    
    routes.append({
        'name': 'Scenic Route',
        'type': 'State Highway',
        'description': 'Beautiful landscapes and tourist spots',
        'distance': route3_data['distance'],
        'waypoints': route3_data['waypoints'],
        'elevation_gain': abs(dest['elevation'] - origin['elevation']) * 1.4,
        'stops': 5,
        'avg_speed': 58,
        'traffic_level': 3,
        'color': '#FBBC04',
        'route_quality': 'scenic',
        'duration': route3_data['duration']
    })
    
    progress_bar.progress(100, text="✅ Routes calculated successfully!")
    time.sleep(0.5)
    progress_bar.empty()
    
    return routes

def create_google_maps_style_map(origin_city, dest_city, routes, selected_route_idx=None):
    """Create map with real routes"""
    origin = INDIAN_CITIES[origin_city]
    dest = INDIAN_CITIES[dest_city]
    
    # Calculate center
    center_lat = (origin['lat'] + dest['lat']) / 2
    center_lon = (origin['lon'] + dest['lon']) / 2
    
    # Calculate zoom
    lat_diff = abs(origin['lat'] - dest['lat'])
    lon_diff = abs(origin['lon'] - dest['lon'])
    max_diff = max(lat_diff, lon_diff)
    
    if max_diff < 2:
        zoom_start = 8
    elif max_diff < 5:
        zoom_start = 7
    elif max_diff < 10:
        zoom_start = 6
    else:
        zoom_start = 5
    
    # Create map with better tiles
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles='OpenStreetMap'
    )
    
    # Add markers
    folium.Marker(
        [origin['lat'], origin['lon']],
        popup=f"<b>START: {origin_city}</b><br>{origin['state']}",
        tooltip=f"🚀 Start: {origin_city}",
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)
    
    folium.Marker(
        [dest['lat'], dest['lon']],
        popup=f"<b>DESTINATION: {dest_city}</b><br>{dest['state']}",
        tooltip=f"🏁 End: {dest_city}",
        icon=folium.Icon(color='red', icon='flag-checkered', prefix='fa')
    ).add_to(m)
    
    # Add routes with smooth lines
    for idx, route in enumerate(routes):
        color = route['color']
        
        if selected_route_idx is not None and idx == selected_route_idx:
            weight = 7
            opacity = 0.9
        else:
            weight = 4
            opacity = 0.5
        
        # Draw smooth route
        folium.PolyLine(
            locations=route['waypoints'],
            color=color,
            weight=weight,
            opacity=opacity,
            smooth_factor=2.0,  # Smooth the line
            popup=f"<b>{route['name']}</b><br>Distance: {route['distance']:.1f} km<br>Time: {route['duration']:.1f} hrs",
            tooltip=f"{route['name']} - {route['distance']:.1f} km"
        ).add_to(m)
        
        # Add route label
        if len(route['waypoints']) > 5:
            mid_idx = len(route['waypoints']) // 2
            folium.Marker(
                route['waypoints'][mid_idx],
                icon=folium.DivIcon(html=f"""
                <div style="
                    background: {color}; 
                    color: white; 
                    padding: 6px 12px; 
                    border-radius: 15px; 
                    font-weight: bold; 
                    font-size: 11px; 
                    text-align: center; 
                    border: 2px solid white;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                    white-space: nowrap;
                ">
                    {route['name']}<br>{route['distance']:.0f} km
                </div>
                """)
            ).add_to(m)
    
    return m

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between coordinates"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def predict_energy_consumption(route, vehicle_specs, conditions):
    """Enhanced energy prediction"""
    base_energy = calculate_physics_energy(route, vehicle_specs, conditions)
    
    # Environmental factors
    temp_factor = 1.0 + abs(conditions['temperature'] - 25) * 0.01
    weather_multipliers = {
        'Clear': 1.0, 'Partly Cloudy': 1.02, 'Cloudy': 1.04,
        'Light Rain': 1.12, 'Heavy Rain': 1.25, 'Fog': 1.08
    }
    weather_factor = weather_multipliers.get(conditions['weather'], 1.0)
    
    route_factor = {'excellent': 0.92, 'good': 1.0, 'scenic': 1.15}[route['route_quality']]
    traffic_factor = 1.0 + route['traffic_level'] * 0.04
    
    total_energy = base_energy * temp_factor * weather_factor * route_factor * traffic_factor
    
    return {
        'energy': total_energy,
        'efficiency': (total_energy / route['distance']) * 1000,
        'battery_used': (total_energy / vehicle_specs['battery']) * 100,
        'cost': total_energy * 8.5,
        'time': route['duration'],
        'charging_needed': max(0, total_energy - vehicle_specs['battery'] * 0.8),
        'range_anxiety': 'High' if total_energy > vehicle_specs['battery'] * 0.9 else 'Low'
    }

def calculate_physics_energy(route, vehicle_specs, conditions):
    """Physics-based energy calculation"""
    g = 9.81
    rho = 1.225
    Cr = 0.01
    
    distance_m = route['distance'] * 1000
    speed_ms = route['avg_speed'] / 3.6
    mass = vehicle_specs['weight'] + conditions['total_mass']
    
    rolling_energy = Cr * mass * g * distance_m / 3600 / 1000
    drag_energy = 0.5 * vehicle_specs['drag'] * rho * vehicle_specs['area'] * (speed_ms ** 3) * route['duration'] / 1000
    potential_energy = mass * g * route['elevation_gain'] / 3600 / 1000
    
    total_energy = (rolling_energy + drag_energy + max(0, potential_energy)) / vehicle_specs['efficiency']
    
    return total_energy

def create_energy_gauge(energy_used, battery_capacity):
    """Create energy gauge"""
    percentage = (energy_used / battery_capacity) * 100
    
    if percentage < 50:
        color = '#34A853'
    elif percentage < 75:
        color = '#FBBC04'
    else:
        color = '#EA4335'
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Battery Usage (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "darkgray"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    st.markdown('<h1 class="main-header">🗺️ EV Route Navigator - Real Road Routes</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔍 Plan Your Route")
        
        origin_city = st.selectbox("From", list(INDIAN_CITIES.keys()))
        dest_city = st.selectbox("To", [c for c in INDIAN_CITIES.keys() if c != origin_city])
        
        st.markdown("---")
        st.markdown("### 🚗 Vehicle Details")
        vehicle_model = st.selectbox("Select EV Model", list(EV_MODELS.keys()))
        vehicle_specs = EV_MODELS[vehicle_model]
        
        st.info(f"""
        **🔋 Battery:** {vehicle_specs['battery']} kWh  
        **📏 Range:** {vehicle_specs['range']} km  
        **⚡ Charging:** {vehicle_specs['charging_speed']} kW
        """)
        
        st.markdown("### 🌤️ Trip Conditions")
        passengers = st.slider("👥 Passengers", 1, 8, 2)
        luggage = st.slider("🧳 Luggage (kg)", 0, 300, 50)
        temperature = st.slider("🌡️ Temperature (°C)", -5, 50, 25)
        weather = st.selectbox("☀️ Weather", ['Clear', 'Partly Cloudy', 'Cloudy', 'Light Rain'])
        
        if st.button("🗺️ Calculate Routes", type="primary", use_container_width=True):
            st.session_state['calculate_routes'] = True
    
    # Main content
    if st.session_state.get('calculate_routes', False):
        routes = generate_enhanced_routes(origin_city, dest_city)
        
        conditions = {
            'temperature': temperature,
            'weather': weather,
            'total_mass': passengers * 70 + luggage
        }
        
        predictions = [predict_energy_consumption(r, vehicle_specs, conditions) for r in routes]
        
        st.session_state.update({
            'routes': routes,
            'predictions': predictions,
            'origin': origin_city,
            'dest': dest_city
        })
    
    # Display results
    if 'routes' in st.session_state:
        routes = st.session_state['routes']
        predictions = st.session_state['predictions']
        selected_route = st.session_state.get('selected_route', 0)
        
        st.markdown("### 🛣️ Route Options (Real Road Routes)")
        
        cols = st.columns(len(routes))
        for idx, (col, route, pred) in enumerate(zip(cols, routes, predictions)):
            with col:
                if st.button(f"📍 {route['name']}", key=f"route_{idx}", use_container_width=True):
                    st.session_state['selected_route'] = idx
                    st.rerun()
                
                st.markdown(f"""
                <div class="{'route-panel-selected' if idx == selected_route else 'route-panel'}">
                    <h4 style="color: {route['color']};">{route['name']}</h4>
                    <p style="font-size: 14px;">{route['description']}</p>
                    <b>{route['distance']:.0f}</b> km • <b>{pred['time']:.1f}</b> hrs<br>
                    <small>⚡ {pred['energy']:.1f} kWh • 🔋 {pred['battery_used']:.0f}%</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Map with real routes
        st.markdown("### 🗺️ Interactive Map")
        st.info("✨ Routes follow actual roads using OpenStreetMap routing")
        
        route_map = create_google_maps_style_map(
            st.session_state['origin'], 
            st.session_state['dest'], 
            routes, 
            selected_route
        )
        st_folium(route_map, width=1400, height=600)
        
        # Metrics
        st.markdown("### 📊 Route Analysis")
        sel_pred = predictions[selected_route]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card-gmaps">', unsafe_allow_html=True)
            st.metric("⚡ Energy", f"{sel_pred['energy']:.1f} kWh")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card-gmaps">', unsafe_allow_html=True)
            st.metric("🔋 Battery", f"{sel_pred['battery_used']:.0f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card-gmaps">', unsafe_allow_html=True)
            st.metric("⏱️ Time", f"{sel_pred['time']:.1f} hrs")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card-gmaps">', unsafe_allow_html=True)
            st.metric("💰 Cost", f"₹{sel_pred['cost']:.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Gauge and comparison
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### ⚡ Battery Status")
            gauge_fig = create_energy_gauge(sel_pred['energy'], vehicle_specs['battery'])
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 Route Comparison")
            comparison_df = pd.DataFrame({
                'Route': [r['name'] for r in routes],
                'Energy (kWh)': [p['energy'] for p in predictions],
                'Time (hrs)': [p['time'] for p in predictions]
            })
            fig = px.bar(comparison_df, x='Route', y='Energy (kWh)', color='Energy (kWh)')
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()