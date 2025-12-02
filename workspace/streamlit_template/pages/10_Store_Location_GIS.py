# workspace/streamlit_template/pages/10_Store_Location_GIS.py
"""
Robust Store Location GIS page for Unified Retail Intelligence Platform (URIP)

Notes:
- This page will run in degraded mode when heavy GIS libs (geopandas/fiona) are missing.
- For full GIS functionality locally or in Docker, install geopandas/fiona and system libs (GDAL/PROJ).
- To show maps on Streamlit Cloud without geopandas, preprocess KML -> GeoJSON locally and commit wards.geojson + wards_stats.csv into assets/.
"""

# -------------------------------
# set_page_config must be the FIRST Streamlit call
# -------------------------------
import streamlit as st

st.set_page_config(page_title="Store Location GIS", page_icon="🏪", layout="wide")

# -------------------------------
# Defensive imports for GIS libs
# -------------------------------
HAS_FOLIUM = False
HAS_GEO = False
HAS_FIONA = False

# Try to import folium and supporting libs (folium is often fine)
try:
    import folium
    from folium.plugins import MarkerCluster, HeatMap
    from streamlit_folium import folium_static
    HAS_FOLIUM = True
except Exception:
    folium = None
    folium_static = None

# geopandas / shapely (heavy)
try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEO = True
except Exception:
    gpd = None

# fiona (used for KML reading)
try:
    import fiona
    HAS_FIONA = True
except Exception:
    fiona = None

# Safe common imports
import pandas as pd
import numpy as np
import os
from pathlib import Path
import io
import json
import zipfile
import shutil
from datetime import datetime
import tempfile

# Local utils (optional - adjust if your utils lives elsewhere)
try:
    # try to import your app's sidebar renderer
    from utils.ui_components import render_sidebar
except Exception:
    # fallback: simple sidebar stub
    def render_sidebar():
        st.sidebar.markdown("**Sidebar**")
        st.sidebar.info("UI components not loaded (utils.ui_components missing)")

# -------------------------------
# Helper: safe CSS loader relative to page file
# -------------------------------
def load_css_for_page(page_file: str):
    """
    Load assets/custom.css relative to this page file.
    Falls back silently if file not present.
    """
    try:
        base_dir = Path(page_file).resolve().parent.parent  # workspace/streamlit_template/pages -> workspace/streamlit_template
        css_path = base_dir / "assets" / "custom.css"
        if css_path.exists():
            with open(css_path, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        else:
            # If there's a committed wards.geojson etc, it's fine — CSS optional
            pass
    except Exception:
        # don't break the page if css loader fails
        pass

# load CSS for this page
load_css_for_page(__file__)

# -------------------------------
# User-facing notice for missing GIS dependencies
# -------------------------------
def gis_missing_notice():
    st.warning(
        "🔧 GIS dependencies are not installed in this environment.\n\n"
        "- `geopandas` and/or `fiona` (system-dependent packages) or `folium` are missing.\n\n"
        "Options:\n\n"
        "1. **Run locally (recommended)** with conda: install GDAL/PROJ and geopandas/fiona. See README.\n"
        "2. **Use preprocessed GeoJSON/CSV**: Convert KML -> GeoJSON locally and commit `assets/wards.geojson` and `assets/wards_stats.csv` to the repo. The page will work with `folium` only.\n"
        "3. **Deploy with Docker or a host that supports system libraries** (GDAL/PROJ) for full GIS support.\n"
    )

# -------------------------------
# Data loader (defensive)
# -------------------------------
def load_gis_data():
    """
    Load and preprocess GIS datasets.
    Returns: (reliance_gdf, competitor_gdf, wards_gdf, wards_stats_df)
    May use GeoJSON/CSV fallbacks if KML parsing not possible.
    """
    try:
        repo_root = Path(__file__).resolve().parent.parent  # workspace/streamlit_template
        assets_dir = repo_root / "assets"

        # --- Load competitor/reliance store datasets ---
        # prefer files in assets, otherwise try root working dir
        reliance_csv = assets_dir / "reliance fresh dataset.csv"
        competitor_xlsx = assets_dir / "KPN fresh dataset.xlsx"

        if not reliance_csv.exists():
            # fallback to working directory filename
            reliance_csv = Path("reliance fresh dataset.csv")
        if not competitor_xlsx.exists():
            competitor_xlsx = Path("KPN fresh dataset.xlsx")

        # If files not present, warn and return None
        if not reliance_csv.exists() or not competitor_xlsx.exists():
            st.warning("Store datasets not found in assets or repo root. Expect limited functionality.")
            # return empty GeoDataFrames if geopandas available, else None
            if HAS_GEO:
                empty_gdf = gpd.GeoDataFrame(columns=['name', 'address', 'latitude', 'longitude', 'brand', 'geometry'], crs="EPSG:4326")
                return empty_gdf, empty_gdf.copy(), None, None
            else:
                return None, None, None, None

        reliance_df = pd.read_csv(reliance_csv)
        competitor_df = pd.read_excel(competitor_xlsx)

        # Standardize minimal columns
        # (Make these tolerant: map various possible column names)
        def pick_col(df, candidates, default=None):
            for c in candidates:
                if c in df.columns:
                    return c
            return default

        # Reliance
        lat_col_r = pick_col(reliance_df, ['latitude', 'lat', 'Latitude', 'LAT'])
        lon_col_r = pick_col(reliance_df, ['longitude', 'lon', 'lng', 'LONGITUDE', 'LONG'])
        name_col_r = pick_col(reliance_df, ['Name', 'name', 'store_name'])
        addr_col_r = pick_col(reliance_df, ['Address', 'address', 'addr'])

        if lat_col_r:
            reliance_df['latitude'] = pd.to_numeric(reliance_df[lat_col_r], errors='coerce')
        if lon_col_r:
            reliance_df['longitude'] = pd.to_numeric(reliance_df[lon_col_r], errors='coerce')
        reliance_df['name'] = reliance_df[name_col_r] if name_col_r in reliance_df.columns else reliance_df.get('name', 'Reliance Store')
        reliance_df['address'] = reliance_df[addr_col_r] if addr_col_r in reliance_df.columns else reliance_df.get('address', '')

        # Competitor (KPN)
        lat_col_c = pick_col(competitor_df, ['latitude', 'lat', 'Latitude'])
        lon_col_c = pick_col(competitor_df, ['longitude', 'lon', 'lng'])
        name_col_c = pick_col(competitor_df, ['name', 'Name', 'store_name'])
        addr_col_c = pick_col(competitor_df, ['address', 'Address', 'addr'])

        if lat_col_c:
            competitor_df['latitude'] = pd.to_numeric(competitor_df[lat_col_c], errors='coerce')
        if lon_col_c:
            competitor_df['longitude'] = pd.to_numeric(competitor_df[lon_col_c], errors='coerce')
        competitor_df['name'] = competitor_df[name_col_c] if name_col_c in competitor_df.columns else competitor_df.get('name', 'KPN Store')
        competitor_df['address'] = competitor_df[addr_col_c] if addr_col_c in competitor_df.columns else competitor_df.get('address', '')

        # add brand
        reliance_df['brand'] = 'Reliance Fresh'
        competitor_df['brand'] = 'KPN Fresh'

        # create GeoDataFrames if geopandas available, else retain DataFrames
        if HAS_GEO:
            reliance_gdf = gpd.GeoDataFrame(
                reliance_df.dropna(subset=['longitude', 'latitude']),
                geometry=gpd.points_from_xy(reliance_df.dropna(subset=['longitude', 'latitude'])['longitude'],
                                            reliance_df.dropna(subset=['longitude', 'latitude'])['latitude']),
                crs="EPSG:4326"
            )
            competitor_gdf = gpd.GeoDataFrame(
                competitor_df.dropna(subset=['longitude', 'latitude']),
                geometry=gpd.points_from_xy(competitor_df.dropna(subset=['longitude', 'latitude'])['longitude'],
                                            competitor_df.dropna(subset=['longitude', 'latitude'])['latitude']),
                crs="EPSG:4326"
            )
        else:
            # minimal dataframes with lat/lon
            reliance_gdf = reliance_df.dropna(subset=['longitude', 'latitude']).copy()
            competitor_gdf = competitor_df.dropna(subset=['longitude', 'latitude']).copy()

        # --- Load ward boundaries & stats ---
        # Prefer preprocessed GeoJSON + CSV under assets to avoid fiona/geopandas KML problems on cloud
        wards_geojson = assets_dir / "wards.geojson"
        wards_stats_csv = assets_dir / "bbmp_wards_full_merged.csv"  # same name as original script usage

        wards_gdf = None
        wards_stats_df = None

        if wards_geojson.exists() and wards_stats_csv.exists():
            # preferred: preprocessed GeoJSON & stats CSV (works even without fiona)
            if HAS_GEO:
                wards_gdf = gpd.read_file(wards_geojson)
            else:
                # read GeoJSON as dict for folium usage
                wards_gdf = None

            wards_stats_df = pd.read_csv(wards_stats_csv)
            # ensure numeric Population column
            if 'Population' in wards_stats_df.columns:
                wards_stats_df['Population'] = pd.to_numeric(wards_stats_df['Population'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            return reliance_gdf, competitor_gdf, wards_gdf, wards_stats_df

        # If no preprocessed files, attempt to parse KML (requires fiona/geopandas)
        kml_path = assets_dir / "bbmp_final_new_wards.kml"
        csv_stats_path = assets_dir / "bbmp_wards_full_merged.csv"
        if not kml_path.exists():
            # fallback to repo root
            kml_path = Path("bbmp_final_new_wards.kml")
        if not csv_stats_path.exists():
            csv_stats_path = Path("bbmp_wards_full_merged.csv")

        if kml_path.exists() and csv_stats_path.exists() and HAS_GEO:
            # attempt to read KML using fiona driver if available
            try:
                # Ensure KML driver declared for fiona
                if HAS_FIONA:
                    fiona.drvsupport.supported_drivers['KML'] = 'rw'
                wards_gdf = gpd.read_file(kml_path, driver='KML')

                # parse extended attributes via xml if present (best-effort)
                try:
                    import xml.etree.ElementTree as ET
                    tree = ET.parse(kml_path)
                    root = tree.getroot()
                    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
                    ward_data = []
                    for placemark in root.findall('.//kml:Placemark', ns):
                        data = {'Ward Name': None, 'Population': 0, 'Assembly constituency': None}
                        extended_data = placemark.find('kml:ExtendedData', ns)
                        if extended_data is not None:
                            schema_data = extended_data.find('kml:SchemaData', ns)
                            if schema_data is not None:
                                for sd in schema_data.findall('kml:SimpleData', ns):
                                    name = sd.get('name')
                                    value = sd.text
                                    if name and value:
                                        if name.lower() in ['name_en', 'ward_name', 'ward name']:
                                            data['Ward Name'] = value
                                        elif 'pop' in name.lower():
                                            try:
                                                data['Population'] = int(float(value))
                                            except:
                                                pass
                                        elif 'assembly' in name.lower():
                                            data['Assembly constituency'] = value
                        ward_data.append(data)
                except Exception:
                    ward_data = []

                # read CSV stats
                if csv_stats_path.exists():
                    wards_stats_df = pd.read_csv(csv_stats_path)
                    if 'Population' in wards_stats_df.columns:
                        wards_stats_df['Population'] = pd.to_numeric(wards_stats_df['Population'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

                    # Merge centroid coords from kml to csv stats (normalized names)
                    if 'Ward Name' in wards_gdf.columns:
                        wards_gdf = wards_gdf.to_crs(epsg=4326)
                        wards_gdf['latitude'] = wards_gdf.geometry.centroid.y
                        wards_gdf['longitude'] = wards_gdf.geometry.centroid.x
                        wards_gdf['Ward Name Clean'] = wards_gdf['Ward Name'].astype(str).str.strip().str.title()
                        wards_stats_df['Ward Name Clean'] = wards_stats_df['Ward Name'].astype(str).str.strip().str.title()
                        wards_stats_df = pd.merge(wards_stats_df, wards_gdf[['Ward Name Clean', 'latitude', 'longitude']].drop_duplicates(),
                                                 on='Ward Name Clean', how='left')
                        wards_stats_df = wards_stats_df.drop(columns=['Ward Name Clean'])
                    return reliance_gdf, competitor_gdf, wards_gdf, wards_stats_df
                else:
                    # no stats CSV but we have KML geometry -> return wards_gdf only
                    return reliance_gdf, competitor_gdf, wards_gdf, None
            except Exception as e:
                st.warning(f"Could not read KML with geopandas/fiona: {e}")
                # fallback below
        # Nothing found or cannot parse
        st.warning("No preprocessed GeoJSON/CSV found and KML parsing not possible in this environment.")
        return reliance_gdf, competitor_gdf, None, None

    except Exception as e:
        st.error(f"Error loading GIS data: {e}")
        return None, None, None, None

# -------------------------------
# QGIS export helper (unchanged, but defensive)
# -------------------------------
def create_qgis_project(reliance_stores, competitor_stores, population_data,
                       recommendations=None, include_demographics=True, project_name="KPN_Fresh_GIS"):
    """
    Create a zip with GeoJSON layers that can be loaded into QGIS.
    Returns (zip_bytes, success_bool)
    """
    try:
        temp_dir = tempfile.mkdtemp()
        project_dir = Path(temp_dir) / project_name.replace(' ', '_')
        project_dir.mkdir(parents=True, exist_ok=True)

        # Write reliance/competitor layers to GeoJSON (if GeoDataFrame provided, use to_json(), else try to assemble)
        def write_geojson(obj, path):
            if obj is None:
                return False
            if HAS_GEO and isinstance(obj, gpd.GeoDataFrame):
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(obj.to_json())
            else:
                # obj may be a regular DataFrame with lat/lon columns
                try:
                    features = []
                    for _, row in obj.iterrows():
                        lat = row.get('latitude') or row.get('lat') or None
                        lon = row.get('longitude') or row.get('lon') or None
                        if pd.isna(lat) or pd.isna(lon):
                            continue
                        features.append({
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                            "properties": {k: (str(v) if not pd.isna(v) else "") for k, v in row.to_dict().items()}
                        })
                    geojson = {"type": "FeatureCollection", "features": features}
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(geojson, f, indent=2)
                except Exception as e:
                    return False
            return True

        reliance_path = project_dir / "01_reliance_stores.geojson"
        competitor_path = project_dir / "02_competitor_stores.geojson"

        write_geojson(reliance_stores, reliance_path)
        write_geojson(competitor_stores, competitor_path)

        demo_paths = []
        if include_demographics and population_data is not None:
            demo_path = project_dir / "03_demographics.geojson"
            if HAS_GEO and isinstance(population_data, gpd.GeoDataFrame):
                with open(demo_path, 'w', encoding='utf-8') as f:
                    f.write(population_data.to_json())
                demo_paths.append(demo_path)
            else:
                # if population_data is None or not GeoDataFrame, skip
                pass

        rec_paths = []
        if recommendations:
            rec_geo = {
                "type": "FeatureCollection",
                "features": []
            }
            for rec in recommendations:
                rec_geo["features"].append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [rec['lng'], rec['lat']]},
                    "properties": {
                        "ward_name": rec.get('ward_name'),
                        "assembly": rec.get('assembly'),
                        "population": rec.get('population'),
                        "total_score": rec.get('total_score'),
                        "distance_km": rec.get('nearest_competitor_km')
                    }
                })
            rec_path = project_dir / "04_recommendations.geojson"
            with open(rec_path, 'w', encoding='utf-8') as f:
                json.dump(rec_geo, f, indent=2)
            rec_paths.append(rec_path)

        # Create zip
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for p in [reliance_path, competitor_path] + demo_paths + rec_paths:
                if p.exists():
                    zipf.write(p, p.name)
            # add README
            readme = project_dir / "README.txt"
            with open(readme, 'w', encoding='utf-8') as f:
                f.write(f"Generated: {datetime.now().isoformat()}\n")
            zipf.write(readme, "README.txt")

        data = zip_buffer.getvalue()
        shutil.rmtree(temp_dir, ignore_errors=True)
        return data, True
    except Exception as e:
        st.error(f"Error creating QGIS export: {e}")
        return None, False

# -------------------------------
# Page UI and logic
# -------------------------------
# Authentication (keep same guard as other pages)
if not st.session_state.get('authenticated', False):
    st.error("Please login first!")
    st.stop()

# Sidebar from utils (if exists)
render_sidebar()

st.title("🏪 Store Location GIS Analytics")
st.markdown("**Strategic Location Intelligence for KPN Fresh Expansion**")

# Load data (cached in session to avoid repeated heavy IO)
if 'reliance_stores' not in st.session_state or 'data_timestamp' not in st.session_state:
    reliance_gdf, competitor_gdf, wards_gdf, wards_stats_df = load_gis_data()
    st.session_state['reliance_stores'] = reliance_gdf
    st.session_state['competitor_stores'] = competitor_gdf
    st.session_state['population_data'] = wards_gdf
    st.session_state['ward_stats'] = wards_stats_df
    st.session_state['data_timestamp'] = datetime.now().timestamp()
else:
    reliance_gdf = st.session_state.get('reliance_stores')
    competitor_gdf = st.session_state.get('competitor_stores')
    wards_gdf = st.session_state.get('population_data')
    wards_stats_df = st.session_state.get('ward_stats')

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Live Map View",
    "📊 Population Intelligence",
    "🎯 Location Recommendations",
    "📈 Analysis Dashboard"
])

# -------------------------------
# Tab 1: Live map view (only if folium available)
# -------------------------------
with tab1:
    st.markdown("### 🗺️ Interactive Bangalore Map")
    if not HAS_FOLIUM:
        gis_missing_notice()
        st.info("You can still view population and recommendation tabs if ward stats are present.")
    else:
        # Map controls
        col_map1, col_map2 = st.columns([2, 1])
        with col_map1:
            map_style = st.selectbox("Map Style", ["OpenStreetMap", "Satellite"])
        with col_map2:
            st.write("")  # spacer

        # Filters (minimal, rely on ward_stats)
        pop_range = (0, 10_000_000)
        buffer_distance = 1.0
        ward_filter = ["High Population", "Medium Population", "Low Population"]
        if st.session_state.get('ward_stats') is not None:
            pop_df = st.session_state['ward_stats']
            try:
                pop_min = int(pop_df['Population'].min())
                pop_max = int(pop_df['Population'].max())
                pop_range = st.slider("Population Range", pop_min, pop_max, (pop_min, min(pop_max, int(pop_min + (pop_max-pop_min)/3))))
                buffer_distance = st.slider("Distance from Competitors (km)", 0.5, 5.0, 1.0, 0.1)
                ward_filter = st.multiselect("Ward Types", ["High Population", "Medium Population", "Low Population"], default=["High Population", "Medium Population", "Low Population"])
            except Exception:
                pass

        try:
            # Initialize map
            if map_style == "OpenStreetMap":
                tiles = "OpenStreetMap"
                attr = None
            else:
                tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                attr = "Tiles © Esri"

            m = folium.Map(location=[12.9716, 77.5946], zoom_start=10, tiles=tiles, attr=attr)

            # Wards layer: use ward_stats_df + wards_gdf if available
            if st.session_state.get('ward_stats') is not None:
                ward_stats = st.session_state['ward_stats'].copy()

                # If wards_gdf available (with geometry), use it; otherwise if wards.geojson exists, read it into python dict and draw
                if st.session_state.get('population_data') is not None and HAS_GEO and isinstance(st.session_state['population_data'], gpd.GeoDataFrame):
                    wards_geo = st.session_state['population_data']
                    wards_geo['Population'] = pd.to_numeric(wards_geo.get('Population', ward_stats.get('Population', 0)), errors='coerce').fillna(0)
                    # Filter by pop_range if provided
                    wards_geo = wards_geo[(wards_geo['Population'] >= pop_range[0]) & (wards_geo['Population'] <= pop_range[1])]
                    # Add GeoJson
                    try:
                        import branca.colormap as cm
                        pop_min = ward_stats['Population'].min()
                        pop_max = ward_stats['Population'].max()
                        colormap = cm.linear.YlOrRd_09.scale(pop_min, pop_max)
                        folium.GeoJson(
                            wards_geo,
                            style_function=lambda feature, colormap=colormap: {
                                'fillColor': colormap(feature['properties'].get('Population', 0)),
                                'color': 'black',
                                'weight': 0.7,
                                'fillOpacity': 0.5
                            },
                            tooltip=folium.GeoJsonTooltip(fields=['Ward Name', 'Population', 'Assembly constituency'],
                                                          aliases=['Ward:', 'Population:', 'Assembly:'])
                        ).add_to(m)
                        colormap.caption = "Ward Population"
                        colormap.add_to(m)
                    except Exception:
                        pass
                else:
                    # fallback: try reading a committed wards.geojson file (no geopandas required)
                    repo_assets = Path(__file__).resolve().parent.parent / "assets"
                    geojson_path = repo_assets / "wards.geojson"
                    if geojson_path.exists():
                        try:
                            folium.GeoJson(str(geojson_path)).add_to(m)
                        except Exception:
                            pass

            # Add reliance markers (blue)
            if st.session_state.get('reliance_stores') is not None:
                rel = st.session_state['reliance_stores']
                # support both GeoDataFrame and plain DataFrame
                if HAS_GEO and isinstance(rel, gpd.GeoDataFrame):
                    for _, row in rel.iterrows():
                        lat = row.geometry.y
                        lon = row.geometry.x
                        popup = f"<b>{row.get('name','Reliance')}</b><br>{row.get('address','')}"
                        folium.Marker([lat, lon], popup=popup, icon=folium.Icon(color='blue', icon='shopping-cart', prefix='fa')).add_to(m)
                else:
                    for _, row in rel.iterrows():
                        lat = row.get('latitude')
                        lon = row.get('longitude')
                        if pd.isna(lat) or pd.isna(lon):
                            continue
                        popup = f"<b>{row.get('name','Reliance')}</b><br>{row.get('address','')}"
                        folium.Marker([lat, lon], popup=popup, icon=folium.Icon(color='blue', icon='shopping-cart', prefix='fa')).add_to(m)

            # Add competitor (KPN) markers (red) and buffer circles
            if st.session_state.get('competitor_stores') is not None:
                comp = st.session_state['competitor_stores']
                if HAS_GEO and isinstance(comp, gpd.GeoDataFrame):
                    for _, row in comp.iterrows():
                        lat = row.geometry.y
                        lon = row.geometry.x
                        popup = f"<b>{row.get('name','KPN')}</b><br>{row.get('address','')}"
                        folium.Marker([lat, lon], popup=popup, icon=folium.Icon(color='red', icon='store', prefix='fa')).add_to(m)
                        folium.Circle([lat, lon], radius=buffer_distance * 1000, color='red', fill=True, fill_opacity=0.08).add_to(m)
                else:
                    for _, row in comp.iterrows():
                        lat = row.get('latitude')
                        lon = row.get('longitude')
                        if pd.isna(lat) or pd.isna(lon):
                            continue
                        popup = f"<b>{row.get('name','KPN')}</b><br>{row.get('address','')}"
                        folium.Marker([lat, lon], popup=popup, icon=folium.Icon(color='red', icon='store', prefix='fa')).add_to(m)
                        folium.Circle([lat, lon], radius=buffer_distance * 1000, color='red', fill=True, fill_opacity=0.08).add_to(m)

            folium.LayerControl().add_to(m)
            folium_static(m, width=1200, height=800)
        except Exception as e:
            st.error(f"Error creating map: {e}")

# -------------------------------
# Tab 2: Population Intelligence
# -------------------------------
with tab2:
    st.markdown("### 📊 Population Intelligence Dashboard")

    if st.session_state.get('ward_stats') is not None:
        pop_df = st.session_state['ward_stats'].copy()

        # ensure Population numeric
        if 'Population' in pop_df.columns:
            pop_df['Population'] = pd.to_numeric(pop_df['Population'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Bangalore Population", f"{int(pop_df['Population'].sum()):,}")
        with col2:
            st.metric("Average Ward Population", f"{int(pop_df['Population'].mean()):,}")
        with col3:
            try:
                st.metric("Highest Ward Population", pop_df.loc[pop_df['Population'].idxmax(), 'Ward Name'])
            except Exception:
                st.metric("Highest Ward Population", "N/A")
        with col4:
            try:
                st.metric("Lowest Ward Population", pop_df.loc[pop_df['Population'].idxmin(), 'Ward Name'])
            except Exception:
                st.metric("Lowest Ward Population", "N/A")

        st.markdown("---")
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            import plotly.figure_factory as ff
            import plotly.express as px
            pop_values = pop_df['Population'].values
            try:
                fig = ff.create_distplot([pop_values], ['Population'], show_hist=True, show_rug=False, bin_size=1000)
                fig.update_layout(title="Population Distribution Across Wards", xaxis_title="Population", yaxis_title="Density")
            except Exception:
                fig = px.histogram(pop_df, x='Population', nbins=20, title="Population Distribution (Histogram)")
            st.plotly_chart(fig, use_container_width=True)

        with col_chart2:
            st.markdown("#### Population by Assembly Constituency")
            try:
                constituency_pop = pop_df.groupby('Assembly constituency')['Population'].sum().sort_values(ascending=False)
                fig = px.bar(x=constituency_pop.index, y=constituency_pop.values, title="Population by Assembly Constituency", labels={'x': 'Assembly', 'y': 'Population'})
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Assembly constituency data not available or malformed.")
        st.markdown("---")
        st.markdown("#### Top 10 Most Populated Wards")
        try:
            top_wards = pop_df.nlargest(10, 'Population')[['Ward Name', 'Population', 'Assembly constituency']].reset_index(drop=True)
            st.dataframe(top_wards, use_container_width=True)
        except Exception:
            st.info("Cannot compute top wards.")
    else:
        st.info("Population ward stats not available. Upload or commit `assets/bbmp_wards_full_merged.csv` (or preprocessed ward file) to enable this tab.")

# -------------------------------
# Tab 3: Recommendations
# -------------------------------
with tab3:
    st.markdown("### 🎯 Strategic Location Recommendations")

    # Basic preconditions
    has_required = all([
        st.session_state.get('reliance_stores') is not None,
        st.session_state.get('competitor_stores') is not None,
        st.session_state.get('ward_stats') is not None
    ])
    if not has_required:
        st.info("Load competitor, reliance and ward stats to generate recommendations. Use preprocessed assets for cloud deployments.")
        if not HAS_GEO:
            st.caption("Tip: Commit assets/wards.geojson and assets/bbmp_wards_full_merged.csv to avoid heavy geopandas dependencies on the host.")
    else:
        # controls
        col_controls1, col_controls2, col_controls3 = st.columns(3)
        with col_controls1:
            target_locations = st.number_input("Number of Recommendations", 5, 50, 10)
        with col_controls2:
            min_distance_choice = st.selectbox("Minimum Distance from Reliance Fresh", ["500m", "1km", "2km", "Custom"], index=1)
            if min_distance_choice == "Custom":
                min_distance_km = st.slider("Custom Distance (km)", 0.2, 5.0, 1.0, 0.1)
            else:
                if 'm' in min_distance_choice:
                    min_distance_km = float(min_distance_choice.replace('m', '')) / 1000.0
                else:
                    min_distance_km = float(min_distance_choice.replace('km', ''))
        with col_controls3:
            roi_weight = st.slider("ROI Focus (Population vs Distance)", 0, 100, 50)

        def generate_location_recommendations_local(competitor_stores, reliance_stores, ward_stats, num_locations=10, min_distance_km=1.0, roi_weight=50):
            from geopy.distance import geodesic
            recommendations = []

            # competitor_stores: KPN (targets); reliance_stores: Reliance (competitors)
            # Convert competitor/reliance coords to lists
            def coords_from_obj(obj):
                coords = []
                if HAS_GEO and isinstance(obj, gpd.GeoDataFrame):
                    for _, r in obj.iterrows():
                        coords.append((r.geometry.y, r.geometry.x, r.get('name', '')))
                else:
                    for _, r in obj.iterrows():
                        lat = r.get('latitude')
                        lon = r.get('longitude')
                        if pd.isna(lat) or pd.isna(lon):
                            continue
                        coords.append((float(lat), float(lon), r.get('name', '')))
                return coords

            reliance_coords = coords_from_obj(reliance_stores)
            competitor_coords = coords_from_obj(competitor_stores)

            for _, ward in ward_stats.iterrows():
                try:
                    pop = ward.get('Population', 0)
                    if pd.isna(pop) or pop < 10000:
                        continue
                    ward_lat = ward.get('latitude') or ward.get('lat') or None
                    ward_lng = ward.get('longitude') or ward.get('lng') or None
                    if not ward_lat or not ward_lng or pd.isna(ward_lat) or pd.isna(ward_lng):
                        continue
                    # nearest reliance distance
                    distances = [geodesic((ward_lat, ward_lng), (lat, lng)).km for lat, lng, _ in reliance_coords] if reliance_coords else [10.0]
                    nearest_rel = min(distances) if distances else 10.0
                    if nearest_rel < min_distance_km:
                        continue
                    pop_ratio = pop / ward_stats['Population'].max()
                    population_score = pop_ratio * 40
                    if nearest_rel < 0.5:
                        distance_score = 10
                    elif nearest_rel < 1.0:
                        distance_score = 25
                    elif nearest_rel < 2.0:
                        distance_score = 30
                    elif nearest_rel < 3.0:
                        distance_score = 25
                    else:
                        distance_score = 15
                    center_distance = geodesic((ward_lat, ward_lng), (12.9716, 77.5946)).km
                    accessibility_score = max(5, 20 - center_distance * 2)
                    socio_score = 5
                    total_score = (population_score * (roi_weight/100) + distance_score * ((100-roi_weight)/100) + accessibility_score * 0.2 + socio_score * 0.1)
                    # nearby competitors within 2km
                    nearby_comp = sorted([
                        (lat, lng, name) for lat, lng, name in reliance_coords
                        if geodesic((ward_lat, ward_lng), (lat, lng)).km <= 2.0
                    ], key=lambda x: geodesic((ward_lat, ward_lng), (x[0], x[1])).km)[:3]
                    recommendations.append({
                        'ward_name': ward.get('Ward Name', ward.get('Ward', 'Unknown')),
                        'assembly': ward.get('Assembly constituency', ward.get('Assembly', 'Unknown')),
                        'population': int(pop),
                        'lat': float(ward_lat),
                        'lng': float(ward_lng),
                        'nearest_competitor_km': nearest_rel,
                        'population_score': population_score,
                        'distance_score': distance_score,
                        'accessibility_score': accessibility_score,
                        'socio_score': socio_score,
                        'total_score': total_score,
                        'nearby_competitors': nearby_comp
                    })
                except Exception:
                    continue

            recommendations.sort(key=lambda x: x['total_score'], reverse=True)
            return recommendations[:num_locations]

        if st.button("🔍 Generate Recommendations", type="primary"):
            with st.spinner("Analyzing optimal locations..."):
                recs = generate_location_recommendations_local(
                    st.session_state['competitor_stores'],
                    st.session_state['reliance_stores'],
                    st.session_state['ward_stats'],
                    num_locations=target_locations,
                    min_distance_km=min_distance_km,
                    roi_weight=roi_weight
                )
                if recs:
                    st.session_state['gis_recommendations'] = recs
                    st.success(f"✅ Generated {len(recs)} recommendations.")
                else:
                    st.warning("No locations found with the current criteria. Try relaxing filters.")

        if 'gis_recommendations' in st.session_state and st.session_state['gis_recommendations']:
            st.markdown("#### 🥇 Top Recommendations")
            for i, rec in enumerate(st.session_state['gis_recommendations'], 1):
                with st.expander(f"#{i} - Score: {rec['total_score']:.1f}", expanded=(i <= 3)):
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.write(f"**Ward:** {rec['ward_name']} ({rec['assembly']})")
                        st.write(f"**Population:** {rec['population']:,}")
                        st.write(f"**Distance to nearest Reliance:** {rec['nearest_competitor_km']:.2f} km")
                    with col_b:
                        if HAS_FOLIUM:
                            mini_map = folium.Map(location=[rec['lat'], rec['lng']], zoom_start=13)
                            folium.Marker([rec['lat'], rec['lng']], popup=f"Recommended: {rec['ward_name']}", icon=folium.Icon(color='green', icon='star', prefix='fa')).add_to(mini_map)
                            for comp in rec.get('nearby_competitors', [])[:3]:
                                folium.Marker([comp[0], comp[1]], popup=f"Competitor: {comp[2]}", icon=folium.Icon(color='blue', icon='shopping-cart', prefix='fa')).add_to(mini_map)
                            folium_static(mini_map, width=300, height=200)
                        else:
                            st.write("Map preview requires folium (not available in this environment).")

            # Export Excel
            if st.button("📥 Download Recommendations as Excel"):
                try:
                    df_export = pd.DataFrame(st.session_state['gis_recommendations'])
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_export.to_excel(writer, index=False, sheet_name='Recommendations')
                        # add simple stats sheet
                        stats = {
                            'Metric': ['Total Population Covered', 'KPN Stores', 'Reliance Stores'],
                            'Value': [int(st.session_state['ward_stats']['Population'].sum()), len(st.session_state['competitor_stores']), len(st.session_state['reliance_stores'])]
                        }
                        pd.DataFrame(stats).to_excel(writer, index=False, sheet_name='Stats')
                    st.download_button("Download Excel", data=output.getvalue(), file_name=f"kpn_recommendations_{datetime.now().strftime('%Y%m%d')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except Exception as e:
                    st.error(f"Export failed: {e}")

# -------------------------------
# Tab 4: Analysis dashboard
# -------------------------------
with tab4:
    st.markdown("### 📈 Network Analysis & Performance Dashboard")
    if st.session_state.get('reliance_stores') is None or st.session_state.get('competitor_stores') is None or st.session_state.get('population_data') is None:
        st.info("GIS data incomplete. Use preprocessed assets or run locally with geopandas for full analysis.")
    else:
        try:
            # Compute some simple KPIs
            reliance_count = len(st.session_state['reliance_stores']) if st.session_state['reliance_stores'] is not None else 0
            kpn_count = len(st.session_state['competitor_stores']) if st.session_state['competitor_stores'] is not None else 0
            total = (reliance_count + kpn_count) if (reliance_count + kpn_count) > 0 else 1
            reliance_share = reliance_count / total * 100
            kpn_share = kpn_count / total * 100

            col1, col2 = st.columns(2)
            with col1:
                import plotly.express as px
                fig = px.pie(values=[kpn_share, reliance_share], names=['KPN Fresh', 'Reliance Fresh'], title="Market Share by Store Count")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                # distance distributions
                from geopy.distance import geodesic
                # get coords lists
                def coords_list(obj):
                    coords = []
                    if HAS_GEO and isinstance(obj, gpd.GeoDataFrame):
                        for _, r in obj.iterrows():
                            coords.append((r.geometry.y, r.geometry.x))
                    else:
                        for _, r in obj.iterrows():
                            lat = r.get('latitude'); lon = r.get('longitude')
                            if pd.isna(lat) or pd.isna(lon): continue
                            coords.append((float(lat), float(lon)))
                    return coords
                kpn_coords = coords_list(st.session_state['competitor_stores'])
                reliance_coords = coords_list(st.session_state['reliance_stores'])
                kpn_distances = []
                for i in range(len(kpn_coords)):
                    for j in range(i+1, len(kpn_coords)):
                        try:
                            kpn_distances.append(geodesic(kpn_coords[i], kpn_coords[j]).km)
                        except Exception:
                            pass
                cross_distances = []
                for kc in kpn_coords:
                    try:
                        nearest = min([geodesic(kc, rc).km for rc in reliance_coords]) if reliance_coords else 0
                        cross_distances.append(nearest)
                    except Exception:
                        pass
                col_a, col_b = st.columns(2)
                with col_a:
                    fig1 = px.histogram(kpn_distances, nbins=20, title="KPN Store-to-Store Distances")
                    st.plotly_chart(fig1, use_container_width=True)
                with col_b:
                    fig2 = px.histogram(cross_distances, nbins=15, title="KPN to Nearest Reliance")
                    st.plotly_chart(fig2, use_container_width=True)

            st.markdown("---")
            st.markdown("### 📈 Key Metrics")
            colk1, colk2, colk3, colk4 = st.columns(4)
            with colk1:
                avg_kpn_dist = np.mean(kpn_distances) if kpn_distances else 0
                st.metric("Avg KPN Distance", f"{avg_kpn_dist:.2f} km")
            with colk2:
                avg_cross = np.mean(cross_distances) if cross_distances else 0
                st.metric("Avg Dist to Reliance", f"{avg_cross:.2f} km")
            with colk3:
                pop_total = st.session_state['ward_stats']['Population'].sum() if st.session_state.get('ward_stats') is not None else 0
                est_pop_per_store = pop_total / max(kpn_count, 1)
                st.metric("Est. Population per KPN Store", f"{int(est_pop_per_store):,}")
            with colk4:
                ward_high = len([w for _, w in st.session_state['population_data'].iterrows() if w.get('Population', 0) > 30000]) if st.session_state.get('population_data') is not None else 0
                coverage_ratio = min((kpn_count / max(ward_high, 1) * 100), 100)
                st.metric("KPN High-Density Coverage", f"{coverage_ratio:.1f}%")
        except Exception as e:
            st.error(f"Analysis error: {e}")

# Footer
st.markdown("---")
st.markdown("**KPN Fresh GIS Analytics Dashboard** — use preprocessed GeoJSON/CSV for cloud deploys, or run locally with conda for full GIS support.")
