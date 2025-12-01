import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import folium_static
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import os
from pathlib import Path
import io
import json
import zipfile
import shutil
from datetime import datetime
import tempfile
from utils.ui_components import render_sidebar
import fiona


def load_css():
    with open("assets/custom.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Configuration and utility functions - defined before any execution
st.set_page_config(page_title="Store Location GIS", page_icon="🏪", layout="wide")

def _get_population_color(population, max_pop):
    """Get color based on population density"""
    ratio = population / max_pop
    if ratio > 0.7:
        return 'darkred'
    elif ratio > 0.5:
        return 'orange'
    elif ratio > 0.3:
        return 'yellow'
    else:
        return 'lightblue'

def load_gis_data():
    """Load and preprocess GIS datasets"""
    try:
        # Load store data
        reliance_df = pd.read_csv('reliance fresh dataset.csv')
        competitor_df = pd.read_excel('KPN fresh dataset.xlsx')

        # Clean and standardize column names
        reliance_df = reliance_df.rename(columns={
            'latitude': 'latitude',
            'longitude': 'longitude',
            'Name': 'name',
            'Address': 'address'
        })

        # Standardize competitor columns (KPN Fresh)
        competitor_df.columns = ['index', 'name', 'address', 'latitude', 'longitude']
        
        # Ensure coordinates are numeric
        competitor_df['latitude'] = pd.to_numeric(competitor_df['latitude'], errors='coerce')
        competitor_df['longitude'] = pd.to_numeric(competitor_df['longitude'], errors='coerce')
        
        # Drop rows with invalid coordinates
        competitor_df = competitor_df.dropna(subset=['latitude', 'longitude'])

        # Add brand identifiers
        reliance_df['brand'] = 'Reliance Fresh'
        competitor_df['brand'] = 'KPN Fresh'

        # Create GeoDataFrames for stores
        reliance_gdf = gpd.GeoDataFrame(
            reliance_df,
            geometry=gpd.points_from_xy(reliance_df.longitude, reliance_df.latitude),
            crs="EPSG:4326"
        )

        competitor_gdf = gpd.GeoDataFrame(
            competitor_df,
            geometry=gpd.points_from_xy(competitor_df.longitude, competitor_df.latitude),
            crs="EPSG:4326"
        )

        # Load Ward Boundaries (KML) - STRICTLY KML ONLY
        import fiona
        import xml.etree.ElementTree as ET

        fiona.drvsupport.supported_drivers['KML'] = 'rw'

        # 1. Get geometries from KML
        wards_gdf = gpd.read_file('bbmp_final_new_wards.kml', driver='KML')

        # 2. Parse attributes from ExtendedData
        tree = ET.parse('bbmp_final_new_wards.kml')
        root = tree.getroot()
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}

        ward_data = []
        for placemark in root.findall('.//kml:Placemark', ns):
            data = {
                'Ward Name': None,
                'Population': 0,
                'Assembly constituency': None,
            }

            extended_data = placemark.find('kml:ExtendedData', ns)
            if extended_data is not None:
                schema_data = extended_data.find('kml:SchemaData', ns)
                if schema_data is not None:
                    for simple_data in schema_data.findall('kml:SimpleData', ns):
                        name = simple_data.get('name')
                        value = simple_data.text

                        # Check for various field name possibilities
                        if name in ['name_en', 'WARD_NAME', 'Ward Name']:
                            data['Ward Name'] = value
                        elif name in ['population', 'POP_TOTAL', 'Population']:
                            try:
                                data['Population'] = int(float(value))
                            except:
                                pass
                        elif name in ['assembly_constituency_name_en', 'ASS_CONST1', 'Assembly Constituency']:
                            data['Assembly constituency'] = value

            ward_data.append(data)

        # 3. Attach attributes to GeoDataFrame
        if len(ward_data) == len(wards_gdf):
            ward_df_attrs = pd.DataFrame(ward_data)
            ward_df_attrs.reset_index(drop=True, inplace=True)
            wards_gdf = wards_gdf.reset_index(drop=True)
            wards_gdf = pd.concat([wards_gdf, ward_df_attrs], axis=1)
        else:
            st.error(
                f"Critical Error: KML geometry count ({len(wards_gdf)}) "
                f"!= attribute count ({len(ward_data)})"
            )
            return None, None, None

        # 4. Compute centroid-based coordinates for each ward
        #    (since KML doesn't store LAT/LON directly)
        wards_gdf = wards_gdf.set_geometry('geometry')
        wards_gdf = wards_gdf.to_crs(epsg=4326)  # ensure lat/lon CRS

        wards_gdf['latitude'] = wards_gdf.geometry.centroid.y
        wards_gdf['longitude'] = wards_gdf.geometry.centroid.x

        # 5. Load CSV for accurate statistics (Source of Truth for 8.4M population)
        wards_stats_df = pd.read_csv('bbmp_wards_full_merged.csv')
        # Ensure Population is numeric
        if wards_stats_df['Population'].dtype == object:
            wards_stats_df['Population'] = wards_stats_df['Population'].astype(str).str.replace(',', '')
        wards_stats_df['Population'] = pd.to_numeric(wards_stats_df['Population'], errors='coerce').fillna(0)
        
        # Clean data: Remove rows with no Ward Name or 0 Population
        wards_stats_df = wards_stats_df.dropna(subset=['Ward Name'])
        wards_stats_df = wards_stats_df[wards_stats_df['Population'] > 0]

        # Sync CSV with KML: Only keep wards that exist in the KML map (198 wards)
        # Normalize names for comparison (strip whitespace, title case)
        kml_wards = set(wards_gdf['Ward Name'].str.strip().str.title())
        wards_stats_df['Ward Name Clean'] = wards_stats_df['Ward Name'].str.strip().str.title()
        
        # Filter
        wards_stats_df = wards_stats_df[wards_stats_df['Ward Name Clean'].isin(kml_wards)]
        wards_stats_df = wards_stats_df.drop(columns=['Ward Name Clean'])
        
        # If filtering removed too many (mismatched names), fallback or warn
        if len(wards_stats_df) < 150:
            # Fallback: If names don't match well, just use the KML attributes as the source of truth
            st.warning(f"⚠️ Name mismatch detected between KML ({len(wards_gdf)}) and CSV. Using KML data for statistics.")
            wards_stats_df = pd.DataFrame(ward_data)
            wards_stats_df = wards_stats_df.dropna(subset=['Ward Name'])
            wards_stats_df = wards_stats_df[wards_stats_df['Population'] > 0]
        else:
            st.success(f"✅ Successfully synced {len(wards_stats_df)} wards from CSV with Map.")

        # CRITICAL FIX: Merge Lat/Lon from KML (wards_gdf) into Statistics (wards_stats_df)
        # The recommendation engine needs 'latitude' and 'longitude' in the stats dataframe
        
        # 1. Prepare KML coordinates with normalized names
        wards_gdf['Ward Name Clean'] = wards_gdf['Ward Name'].str.strip().str.title()
        coords_lookup = wards_gdf[['Ward Name Clean', 'latitude', 'longitude']].drop_duplicates(subset=['Ward Name Clean'])
        
        # 2. Prepare Stats dataframe with normalized names
        wards_stats_df['Ward Name Clean'] = wards_stats_df['Ward Name'].str.strip().str.title()
        
        # 3. Merge
        wards_stats_df = pd.merge(wards_stats_df, coords_lookup, on='Ward Name Clean', how='left')
        
        # 4. Cleanup
        wards_stats_df = wards_stats_df.drop(columns=['Ward Name Clean'])
        wards_stats_df = wards_stats_df.dropna(subset=['latitude', 'longitude'])

        return reliance_gdf, competitor_gdf, wards_gdf, wards_stats_df

    except Exception as e:
        st.error(f"Error loading GIS data: {str(e)}")
        return None, None, None

def create_qgis_project(reliance_stores, competitor_stores, population_data,
                       recommendations=None, include_demographics=True, project_name="KPN_Fresh_GIS"):
    """Create QGIS project file (.qgz) with all layers"""

    try:
        # Create temporary directory for project files
        temp_dir = tempfile.mkdtemp()
        project_dir = Path(temp_dir) / project_name.replace(' ', '_')
        project_dir.mkdir(exist_ok=True)

        # 1. Export Reliance Stores to GeoJSON
        reliance_geojson = reliance_stores.to_json()

        reliance_path = project_dir / "01_reliance_stores.geojson"
        with open(reliance_path, 'w') as f:
            f.write(reliance_geojson)

        # 2. Export Competitor Stores to GeoJSON
        competitor_geojson = competitor_stores.to_json()

        competitor_path = project_dir / "02_competitor_stores.geojson"
        with open(competitor_path, 'w') as f:
            f.write(competitor_geojson)

        # 3. Export Demographics if requested
        demo_paths = []
        if include_demographics and population_data is not None:
            # population_data is now a GeoDataFrame with real boundaries
            demo_geojson = population_data.to_json()

            demo_path = project_dir / "03_demographics.geojson"
            with open(demo_path, 'w') as f:
                f.write(demo_geojson)
            demo_paths.append(demo_path)

        # 4. Export Recommendations if provided
        rec_paths = []
        if recommendations is not None and len(recommendations) > 0:
            rec_data = []
            for rec in recommendations:
                rec_data.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [rec['lng'], rec['lat']]
                    },
                    'properties': {
                        'ward_name': rec['ward_name'],
                        'assembly': rec['assembly'],
                        'population': rec['population'],
                        'total_score': rec['total_score'],
                        'distance_km': rec['nearest_competitor_km']
                    }
                })

            rec_geojson = {
                'type': 'FeatureCollection',
                'features': rec_data
            }

            rec_path = project_dir / "04_recommendations.geojson"
            with open(rec_path, 'w') as f:
                json.dump(rec_geojson, f, indent=2)
            rec_paths.append(rec_path)

        # 5. Create ZIP file for download
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all GeoJSON files
            for geojson_file in [reliance_path, competitor_path] + demo_paths + rec_paths:
                zipf.write(geojson_file, geojson_file.name)

            # Add README
            readme_content = f"""
# {project_name} QGIS Project

## Layers Included:
1. **01_reliance_stores.geojson** - Reliance Fresh store locations
2. **02_competitor_stores.geojson** - KPN Fresh competitor locations
3. **03_demographics.geojson** - Bangalore ward boundaries and population
4. **04_recommendations.geojson** - AI-generated location recommendations

## How to Use:
1. Extract this ZIP file (You'll find .geojson files)
2. Open QGIS Desktop Application
3. Click 'Layer' → 'Add Layer' → 'Add Vector Layer'
4. Select each .geojson file individually, OR drag and drop the whole folder
5. Each layer will load with proper coordinates and attributes

## Professional Analysis Available in QGIS:
- Advanced spatial analysis tools
- Network analysis and accessibility modeling
- Custom demographic overlays
- Professional cartography and reporting
- Integration with external GIS data sources

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
For: KPN Fresh Strategic Expansion Analysis
            """

            readme_path = project_dir / "README.txt"
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            zipf.write(readme_path, "README.txt")

        zip_data = zip_buffer.getvalue()

        # Cleanup temporary files
        shutil.rmtree(temp_dir)

        return zip_data, True

    except Exception as e:
        print(f"Error creating QGIS project: {str(e)}")
        return None, False

def auto_launch_qgis(qgis_project_path):
    """Attempt to auto-launch QGIS with the project"""
    try:
        import subprocess
        import platform

        system = platform.system()

        if system == "Windows":
            # Try common QGIS installation paths on Windows
            qgis_paths = [
                "C:\\Program Files\\QGIS 3.34\\bin\\qgis.exe",
                "C:\\Program Files\\QGIS 3.28\\bin\\qgis.exe",
                "C:\\Program Files\\QGIS 3.22\\bin\\qgis.exe",
                "C:\\Program Files\\QGIS 3.16\\bin\\qgis.exe"
            ]

            for qgis_path in qgis_paths:
                if os.path.exists(qgis_path):
                    subprocess.Popen([qgis_path, str(qgis_project_path)])
                    return True

        elif system == "Darwin":  # macOS
            subprocess.Popen(["open", "-a", "QGIS", str(qgis_project_path)])
            return True

        elif system == "Linux":
            subprocess.Popen(["qgis", str(qgis_project_path)])
            return True

        return False

    except Exception as e:
        print(f"Could not auto-launch QGIS: {str(e)}")
        return False



# Check authentication
if not st.session_state.get('authenticated', False):
    st.error("Please login first!")
    st.stop()

# Sidebar
render_sidebar()

# Page title and description
st.title("🏪 Store Location GIS Analytics")
st.markdown("**Strategic Location Intelligence for KPN Fresh Expansion**")
st.markdown("Analyze Bangalore population data, competitive landscape (vs Reliance Fresh), and identify optimal new store locations")

# Load data on app startup (if not already loaded or if data needs refresh)
# Check for file updates
current_file_time = os.path.getmtime('reliance fresh dataset.csv') if os.path.exists('reliance fresh dataset.csv') else 0
last_file_time = st.session_state.get('data_timestamp', 0)

should_reload = ('reliance_stores' not in st.session_state) or (current_file_time > last_file_time)

if not should_reload and 'population_data' in st.session_state:
    # Check if we have the new data structure (GeoDataFrame)
    if not isinstance(st.session_state.population_data, gpd.GeoDataFrame):
        should_reload = True

# Self-healing: Force reload if ward_stats is missing required columns (lat/lon) for recommendations
if not should_reload and 'ward_stats' in st.session_state:
    if 'latitude' not in st.session_state.ward_stats.columns:
        should_reload = True

if should_reload:
    reliance_gdf, competitor_gdf, wards_gdf, wards_stats_df = load_gis_data()
    if all([reliance_gdf is not None, competitor_gdf is not None, wards_gdf is not None, wards_stats_df is not None]):
        st.session_state.reliance_stores = reliance_gdf
        st.session_state.competitor_stores = competitor_gdf
        st.session_state.population_data = wards_gdf
        st.session_state.ward_stats = wards_stats_df
        st.session_state.data_timestamp = current_file_time
        # Clear recommendations if data changed
        if 'gis_recommendations' in st.session_state:
            del st.session_state['gis_recommendations']



# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Live Map View",
    "📊 Population Intelligence",
    "🎯 Location Recommendations",
    "📈 Analysis Dashboard"
])

with tab1:
    # Main content area
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### 🔧 Live Filters & Controls")

        # Advanced filtering controls
        if 'population_data' in st.session_state:
            # Population density filter
            pop_min = int(st.session_state.population_data['Population'].min())
            pop_max = int(st.session_state.population_data['Population'].max())
            pop_range = st.slider("Population Range", pop_min, pop_max, (20000, 60000),
                                 help="Filter wards by population density")

            # Distance buffer around competitors
            buffer_distance = st.slider("Distance from Competitors (km)",
                                      0.5, 5.0, 1.0, 0.1,
                                      help="Minimum distance from competitor stores")

            # Ward type filter (simplified classification)
            ward_filter = st.multiselect(
                "Ward Types",
                ["High Population", "Medium Population", "Low Population"],
                default=["High Population", "Medium Population", "Low Population"],
                help="Filter by population density categories"
            )

        st.markdown("---")

        # Statistics
        if 'reliance_stores' in st.session_state and 'competitor_stores' in st.session_state:
            st.markdown("### 📊 Network Statistics")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Reliance Fresh (Competitor)", len(st.session_state.reliance_stores))
            with col_b:
                st.metric("KPN Fresh (Target)", len(st.session_state.competitor_stores))

            if 'ward_stats' in st.session_state:
                st.metric("Total Population", f"{st.session_state.ward_stats['Population'].sum():,}")
                st.metric("Average Ward Population",
                         f"{st.session_state.ward_stats['Population'].mean():,.0f}")

    with col2:
        st.markdown("### 🗺️ Interactive Bangalore Map")

        # Map controls
        # Map controls
        col_map1, col_map2 = st.columns([2, 1])
        with col_map1:
            map_style = st.selectbox("Map Style", ["OpenStreetMap", "Satellite"])
        with col_map2:
            st.write("") # Spacer

        if 'reliance_stores' in st.session_state:
            try:
                # Initialize map with satellite option
                if map_style == "OpenStreetMap":
                    tiles = 'OpenStreetMap'
                elif map_style == "Satellite":
                    tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
                    attr = 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'

                m = folium.Map(
                    location=[12.9716, 77.5946],
                    zoom_start=10,  # Closer zoom to show all stores
                    tiles=tiles,
                    attr=attr if map_style == "Satellite" else None
                )

                # Add Ward Polygons Layer
                if 'population_data' in st.session_state:
                    wards_gdf = st.session_state.population_data.copy()
                    
                    import branca.colormap as cm
                    # Ensure population is numeric
                    wards_gdf['Population'] = pd.to_numeric(wards_gdf['Population'], errors='coerce').fillna(0)
                    
                    # --- APPLY FILTERS ---
                    
                    # 1. Population Range Filter
                    wards_gdf = wards_gdf[
                        (wards_gdf['Population'] >= pop_range[0]) & 
                        (wards_gdf['Population'] <= pop_range[1])
                    ]
                    
                    # 2. Ward Type Filter
                    # Calculate density categories (same logic as Tab 2)
                    min_pop_val = st.session_state.population_data['Population'].min()
                    max_pop_val = st.session_state.population_data['Population'].max()
                    range_pop_val = max_pop_val - min_pop_val
                    
                    def get_density_cat(pop):
                        ratio = (pop - min_pop_val) / range_pop_val
                        if ratio > 0.7: return "High Population" # "Very High" -> High
                        elif ratio > 0.5: return "High Population"
                        elif ratio > 0.3: return "Medium Population"
                        else: return "Low Population"
                        
                    wards_gdf['Density_Category'] = wards_gdf['Population'].apply(get_density_cat)
                    
                    # Filter by selected types
                    if ward_filter:
                        wards_gdf = wards_gdf[wards_gdf['Density_Category'].isin(ward_filter)]
                    
                    # ---------------------
                    
                    pop_min = st.session_state.population_data['Population'].min() # Use global min/max for consistent colors
                    pop_max = st.session_state.population_data['Population'].max()
                    colormap = cm.linear.YlOrRd_09.scale(pop_min, pop_max)
                    colormap.caption = "Population by Ward"
                    
                    def ward_style(feature):
                        pop = feature['properties']['Population']
                        return {
                            'fillColor': colormap(pop),
                            'color': 'black',
                            'weight': 0.7,
                            'fillOpacity': 0.5,
                        }
                    
                    if not wards_gdf.empty:
                        folium.GeoJson(
                            wards_gdf,
                            style_function=ward_style,
                            tooltip=folium.GeoJsonTooltip(
                                fields=['Ward Name', 'Population', 'Assembly constituency'],
                                aliases=['Ward:', 'Population:', 'Assembly:']
                            ),
                            name="BBMP Wards"
                        ).add_to(m)
                    else:
                        st.warning("No wards match the selected filters.")
                    
                    colormap.add_to(m)

                # Add stores with permanent labels (no clustering for visibility)
                if 'reliance_stores' in st.session_state:
                    for idx, store in st.session_state.reliance_stores.iterrows():
                        popup_content = f"""
                        <b>{store['name']}</b><br>
                        <b>Brand:</b> {store['brand']}<br>
                        <b>Address:</b> {store.get('address', 'N/A')[:60]}...
                        """

                        # Add marker with permanent label
                        folium.Marker(
                            location=[store.geometry.y, store.geometry.x],
                            popup=popup_content,
                            icon=folium.Icon(color='blue', icon='shopping-cart', prefix='fa'),
                            tooltip=f"Reliance: {store['name']}"
                        ).add_to(m)

                if 'competitor_stores' in st.session_state:
                    for idx, store in st.session_state.competitor_stores.iterrows():
                        popup_content = f"""
                        <b>{store['name']}</b><br>
                        <b>Brand:</b> {store['brand']}<br>
                        <b>Address:</b> {store.get('address', 'N/A')[:60]}...
                        """

                        # Add marker with permanent label
                        folium.Marker(
                            location=[store.geometry.y, store.geometry.x],
                            popup=popup_content,
                            icon=folium.Icon(color='red', icon='store', prefix='fa'),
                            tooltip=f"KPN Fresh: {store['name']}"
                        ).add_to(m)
                        
                        # Add Buffer Circle
                        folium.Circle(
                            location=[store.geometry.y, store.geometry.x],
                            radius=buffer_distance * 1000, # Convert km to meters
                            color='red',
                            fill=True,
                            fill_color='red',
                            fill_opacity=0.1,
                            weight=1,
                            tooltip=f"Buffer: {buffer_distance}km"
                        ).add_to(m)

                # Add layer control
                folium.LayerControl().add_to(m)

                # Display larger map
                folium_static(m, width=1200, height=800)

            except Exception as e:
                st.error(f"Error creating map: {str(e)}")
                if "_get_population_color" in str(e):
                    st.error("Please refresh the page - function definition issue.")
        else:
            st.info("👆 Data loading automatically on first visit...")

with tab2:
    st.markdown("### 📊 Population Intelligence Dashboard")

    if 'ward_stats' in st.session_state:
        pop_df = st.session_state.ward_stats

        # Population summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Bangalore Population",
                     f"{pop_df['Population'].sum():,}")
        with col2:
            st.metric("Average Ward Population",
                     f"{pop_df['Population'].mean():,.0f}")
        with col3:
            st.metric("Highest Ward Population",
                     f"{pop_df.loc[pop_df['Population'].idxmax(), 'Ward Name']}")
        with col4:
            st.metric("Lowest Ward Population",
                     f"{pop_df.loc[pop_df['Population'].idxmin(), 'Ward Name']}")

        st.markdown("---")

        # Population distribution charts
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.markdown("#### Population Density Distribution")
            import plotly.figure_factory as ff
            import plotly.express as px

            pop_values = pop_df['Population'].values
            try:
                fig = ff.create_distplot([pop_values], ['Population'],
                                       show_hist=True, show_rug=False, bin_size=1000)
                fig.update_layout(
                    title="Population Distribution Across Wards",
                    xaxis_title="Population",
                    yaxis_title="Density"
                )
            except Exception as e:
                # Fallback to simple histogram if KDE fails (e.g. singular matrix)
                fig = px.histogram(pop_df, x='Population', nbins=10,
                                 title="Population Distribution (Histogram)")
            
            st.plotly_chart(fig, use_container_width=True)

        with col_chart2:
            st.markdown("#### Population by Assembly Constituency")

            constituency_pop = pop_df.groupby('Assembly constituency')['Population'].sum().sort_values(ascending=False)

            import plotly.express as px
            fig = px.bar(
                x=constituency_pop.index,
                y=constituency_pop.values,
                title="Population by Assembly Constituency",
                labels={'x': 'Assembly Constituency', 'y': 'Total Population'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

        # Detailed ward analysis
        st.markdown("---")
        st.markdown("### 🔍 Ward-Level Demographics")

        # Top 10 most populated wards
        top_wards = pop_df.nlargest(10, 'Population')[['Ward Name', 'Population', 'Assembly constituency']].reset_index(drop=True)

        st.markdown("#### Top 10 Most Populated Wards")
        st.dataframe(top_wards, use_container_width=True)

        # Ward classification
        st.markdown("#### Ward Classification by Population Density")

        # Classify wards
        max_pop = pop_df['Population'].max()
        min_pop = pop_df['Population'].min()
        range_pop = max_pop - min_pop

        def classify_ward(pop):
            ratio = (pop - min_pop) / range_pop
            if ratio > 0.7:
                return "Very High Density"
            elif ratio > 0.5:
                return "High Density"
            elif ratio > 0.3:
                return "Medium Density"
            else:
                return "Low Density"

        pop_df['Density_Category'] = pop_df['Population'].apply(classify_ward)

        density_counts = pop_df['Density_Category'].value_counts()

        col_pie, col_stats = st.columns([1, 2])
        with col_pie:
            fig = px.pie(values=density_counts.values, names=density_counts.index,
                        title="Ward Distribution by Density")
            st.plotly_chart(fig, use_container_width=True)

        with col_stats:
            st.markdown("#### Ward Statistics by Category")
            category_stats = pop_df.groupby('Density_Category').agg({
                'Population': ['count', 'sum', 'mean']
            }).round(0)

            category_stats.columns = ['Wards', 'Total Population', 'Avg Population']
            st.dataframe(category_stats, use_container_width=True)
    else:
        st.info("Population data will be loaded automatically.")

with tab3:
    st.markdown("### 🎯 Strategic Location Recommendations")

    if all([k in st.session_state for k in ['reliance_stores', 'competitor_stores', 'ward_stats']]):
        # Location scoring algorithm
        st.markdown("#### 🎖️Location Scoring")

        with st.expander("🔍 Scoring Methodology", expanded=True):
            st.markdown("""
            **Multi-factor Location Scoring Model:**

            1. **Population Potential (40%)**: Ward population × accessibility factor
            2. **Strategic Distance (30%)**: Optimal distance from competitors (Reliance Fresh) (500m-2km)
            3. **Accessibility Index (20%)**: Proximity to major roads and transport
            4. **Socio-Economic Factors (10%)**:population ratio, income indicators

            **Dynamic Distance Requirements:**
            - High-density wards (>70%): Minimum 500m from Reliance Fresh
            - Medium-density wards (50-70%): Minimum 1km from Reliance Fresh
            - Low-density wards (<50%): Minimum 2km from Reliance Fresh
            """)

        # Recommendation controls
        col_controls1, col_controls2, col_controls3 = st.columns(3)

        with col_controls1:
            target_locations = st.number_input("Number of Recommendations", 5, 20, 10)

        with col_controls2:
            min_distance = st.selectbox("Minimum Distance from Reliance Fresh",
                                      ["500m", "1km", "2km", "Custom"], index=1)
            if min_distance == "Custom":
                custom_distance = st.slider("Custom Distance (km)", 0.2, 5.0, 1.0, 0.1)
                min_distance_km = custom_distance
            else:
                min_distance_km = float(min_distance.replace('km', '').replace('m', '')) / 1000 if 'm' in min_distance else float(min_distance.replace('km', ''))

        with col_controls3:
            roi_weight = st.slider("ROI Focus (Population vs Distance)", 0, 100, 50,
                                 help="Higher = More population-focused, Lower = More distance-focused")

        def generate_location_recommendations_local(reliance_stores, competitor_stores, population_data,
                                                   num_locations=10, min_distance_km=1.0, roi_weight=50):
            """Generate strategic location recommendations based on scoring algorithm"""

            from geopy.distance import geodesic

            recommendations = []

            # Convert competitor locations to list for distance calculations
            competitor_locs = [(row.geometry.y, row.geometry.x, row['name'])
                              for _, row in competitor_stores.iterrows()]

            for _, ward in population_data.iterrows():
                # Skip if population too low
                if ward['Population'] < 10000:
                    continue

                # Use real location from ward data
                ward_lat = ward['latitude']
                ward_lng = ward['longitude']

                # Calculate distance to nearest competitor
                distances_to_competitors = [geodesic((ward_lat, ward_lng), (lat, lng)).km
                                          for lat, lng, _ in competitor_locs]

                nearest_competitor_km = min(distances_to_competitors) if distances_to_competitors else 10.0

                # Skip if too close to competitors
                if nearest_competitor_km < min_distance_km:
                    continue

                # Population score (0-40 points)
                pop_ratio = ward['Population'] / population_data['Population'].max()
                population_score = pop_ratio * 40

                # Distance score (0-30 points) - optimal distance gives highest score
                if nearest_competitor_km < 0.5:
                    distance_score = 10  # Too close
                elif nearest_competitor_km < 1.0:
                    distance_score = 25  # Good distance
                elif nearest_competitor_km < 2.0:
                    distance_score = 30  # Optimal
                elif nearest_competitor_km < 3.0:
                    distance_score = 25  # Good but far
                else:
                    distance_score = 15  # Too far
                
                # Accessibility score (0-20 points) - simplified based on population proximity to center
                center_distance = geodesic((ward_lat, ward_lng), (12.9716, 77.5946)).km
                accessibility_score = max(5, 20 - center_distance * 2)  # Closer to center = higher score

                # Socio-economic score (0-10 points)
                # SC Population not available in new dataset, using default neutral score
                socio_score = 5 

                # Total score with ROI weight
                total_score = (population_score * (roi_weight/100) +
                              distance_score * ((100-roi_weight)/100) +
                              accessibility_score * 0.2 +
                              socio_score * 0.1)

                recommendations.append({
                    'ward_name': ward['Ward Name'],
                    'assembly': ward['Assembly constituency'],
                    'population': ward['Population'],
                    'lat': ward_lat,
                    'lng': ward_lng,
                    'nearest_competitor_km': nearest_competitor_km,
                    'population_score': population_score,
                    'distance_score': distance_score,
                    'accessibility_score': accessibility_score,
                    'socio_score': socio_score,
                    'total_score': total_score,
                    'nearby_competitors': sorted([
                        (lat, lng, name) for lat, lng, name in competitor_locs
                        if geodesic((ward_lat, ward_lng), (lat, lng)).km <= 2.0
                    ], key=lambda x: geodesic((ward_lat, ward_lng), (x[0], x[1])).km)[:3]
                })

            # Sort by total score and return top recommendations
            recommendations.sort(key=lambda x: x['total_score'], reverse=True)
            return recommendations[:num_locations]

        if st.button("🔍 Generate Recommendations", type="primary"):
            with st.spinner("Analyzing optimal locations..."):
                recommendations = generate_location_recommendations_local(
                    st.session_state.competitor_stores, # Target: KPN (Red)
                    st.session_state.reliance_stores,   # Competitor: Reliance (Blue)
                    st.session_state.ward_stats,
                    target_locations,
                    min_distance_km,
                    roi_weight
                )

                if recommendations:
                    st.session_state.gis_recommendations = recommendations
                    st.success(f"✅ Generated {len(recommendations)} optimal location recommendations!")

                    # Display recommendations
                    st.markdown("#### 🥇 Top Recommended Locations")

                    for i, rec in enumerate(recommendations[:10], 1):
                        with st.expander(f"#{i} Location Score: {rec['total_score']:.1f}", expanded=i<=3):
                            col_rec1, col_rec2 = st.columns([2, 1])

                            with col_rec1:
                                st.markdown(f"""
                                **📍 Ward:** {rec['ward_name']} ({rec['assembly']})
                                **👥 Population:** {rec['population']:,}
                                **🎯 Strategic Distance:** {rec['nearest_competitor_km']:.2f}km
                                **📈 Accessibility Score:** {rec['accessibility_score']:.1f}/10
                                """)

                            with col_rec2:
                                # Mini recommender map
                                rec_map = folium.Map(location=[rec['lat'], rec['lng']], zoom_start=13)
                                folium.Marker([rec['lat'], rec['lng']],
                                            popup=f"<b>Recommended Location</b><br>Ward: {rec['ward_name']}",
                                            icon=folium.Icon(color='green', icon='star', prefix='fa')).add_to(rec_map)

                                # Add nearby competitors
                                for comp in rec['nearby_competitors'][:3]:
                                    folium.Marker([comp[1], comp[0]],
                                                popup=f"<b>Competitor: {comp[2]}</b>",
                                                icon=folium.Icon(color='blue', icon='shopping-cart', prefix='fa')).add_to(rec_map)

                                folium_static(rec_map, width=300, height=200)

                    # Export option
                    st.markdown("---")
                    st.markdown("#### 📁 Download Location Recommendations")

                    # Convert recommendations to DataFrame
                    export_df = pd.DataFrame(recommendations)

                    # Create Excel file in memory
                    from io import BytesIO
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Sheet 1: Recommendations
                        export_df.to_excel(writer, index=False, sheet_name='KPN_Expansion_Recommendations')
                        
                        # Sheet 2: Network Statistics
                        stats_data = {
                            'Metric': [
                                'Target Brand',
                                'Competitor Brand',
                                'KPN Fresh Stores (Current)',
                                'Reliance Fresh Stores (Competitor)',
                                'Total Population Coverage',
                                'Average Ward Population'
                            ],
                            'Value': [
                                'KPN Fresh',
                                'Reliance Fresh',
                                len(st.session_state.competitor_stores),
                                len(st.session_state.reliance_stores),
                                st.session_state.ward_stats['Population'].sum(),
                                st.session_state.ward_stats['Population'].mean()
                            ]
                        }
                        pd.DataFrame(stats_data).to_excel(writer, index=False, sheet_name='Network_Statistics')
                        
                    excel_data = output.getvalue()

                    # Download button (this creates a proper downloadable link)
                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                    filename = f'kpn_fresh_expansion_plan_{timestamp}.xlsx'

                    st.download_button(
                        label="📥 Download KPN Expansion Report",
                        data=excel_data,
                        file_name=filename,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        help="Download strategic expansion recommendations for KPN Fresh"
                    )
                else:
                    st.warning("No optimal locations found with current criteria. Try adjusting parameters.")
    else:
        st.info("All GIS data needs to be loaded for location recommendations.")



with tab4:
    st.markdown("### 📈 Network Analysis & Performance Dashboard")

    if all([k in st.session_state for k in ['reliance_stores', 'competitor_stores', 'population_data']]):
        # Network analysis
        col_analysis1, col_analysis2 = st.columns(2)

        with col_analysis1:
            st.markdown("#### 📊 Market Share Analysis")

            total_stores = len(st.session_state.reliance_stores) + len(st.session_state.competitor_stores)
            reliance_share = len(st.session_state.reliance_stores) / total_stores * 100
            competitor_share = len(st.session_state.competitor_stores) / total_stores * 100

            fig = px.pie(values=[competitor_share, reliance_share],
                        names=['KPN Fresh', 'Reliance Fresh'],
                        title="Market Share by Store Count")
            st.plotly_chart(fig, use_container_width=True)

        with col_analysis2:
            st.markdown("#### 📍 Distance Distribution Analysis")

            from geopy.distance import geodesic

            # Calculate distances between all store pairs
            reliance_coords = [(row.geometry.y, row.geometry.x)
                             for _, row in st.session_state.reliance_stores.iterrows()]
            competitor_coords = [(row.geometry.y, row.geometry.x)
                               for _, row in st.session_state.competitor_stores.iterrows()]

            # KPN store distances (competitor_stores is KPN)
            kpn_distances = []
            for i, coord1 in enumerate(competitor_coords):
                for j, coord2 in enumerate(competitor_coords[i+1:], i+1):
                    kpn_distances.append(geodesic(coord1, coord2).km)

            # Cross-brand distances (nearest Reliance to each KPN store)
            cross_distances = []
            for kpn_coord in competitor_coords:
                nearest_reliance = min([geodesic(kpn_coord, rel_coord).km
                                  for rel_coord in reliance_coords])
                cross_distances.append(nearest_reliance)

            col_dist1, col_dist2 = st.columns(2)
            with col_dist1:
                fig1 = px.histogram(kpn_distances, nbins=20,
                                  title="KPN Store-to-Store Distances")
                fig1.update_xaxes(title="Distance (km)")
                st.plotly_chart(fig1, use_container_width=True)

            with col_dist2:
                fig2 = px.histogram(cross_distances, nbins=15,
                                  title="KPN to Nearest Reliance Fresh")
                fig2.update_xaxes(title="Distance (km)")
                st.plotly_chart(fig2, use_container_width=True)

        # Performance metrics
        st.markdown("---")
        st.markdown("### 📈 Key Performance Indicators")

        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

        with col_kpi1:
            # Average distance between KPN stores
            avg_kpn_distance = np.mean(kpn_distances) if kpn_distances else 0
            st.metric("Avg KPN Store Distance", f"{avg_kpn_distance:.1f}km")

        with col_kpi2:
            # Average distance to nearest Reliance
            avg_comp_distance = np.mean(cross_distances) if cross_distances else 0
            st.metric("Avg Dist to Reliance", f"{avg_comp_distance:.1f}km")

        with col_kpi3:
            # Population coverage estimate (simplified)
            population_per_store = (st.session_state.population_data['Population'].sum() /
                                   len(st.session_state.competitor_stores)) # Using KPN count
            st.metric("Est. Population per KPN Store", f"{population_per_store:,.0f}")

        with col_kpi4:
            # Market penetration
            ward_coverage = len([w for _, w in st.session_state.population_data.iterrows()
                               if w['Population'] > 30000])  # High-density wards
            coverage_ratio = len(st.session_state.competitor_stores) / ward_coverage * 100 # Using KPN count
            st.metric("KPN High-Density Coverage", f"{min(coverage_ratio, 100):.1f}%")
    else:
        st.info("Complete GIS data required for network analysis.")

# Footer with implementation status
st.markdown("---")
st.markdown("""
**🎯 **KPN Fresh GIS Analytics Dashboard** - Complete Strategic Location Intelligence Solution**
""")
