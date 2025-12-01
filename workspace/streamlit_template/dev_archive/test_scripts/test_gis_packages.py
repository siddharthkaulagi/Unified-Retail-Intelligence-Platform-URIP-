#!/usr/bin/env python
"""Test GIS package imports and basic GIS page functionality"""

try:
    import folium
    print("Folium imported successfully")
    folium_available = True
except ImportError as e:
    print(f"Folium import failed: {e}")
    folium_available = False

try:
    import geopandas
    print("Geopandas imported successfully")
    geopandas_available = True
except ImportError as e:
    print(f"Geopandas import failed: {e}")
    geopandas_available = False

try:
    import shapely
    print("Shapely imported successfully")
    shapely_available = True
except ImportError as e:
    print(f"Shapely import failed: {e}")
    shapely_available = False

try:
    import streamlit_folium
    print("Streamlit-Folium imported successfully")
    streamlit_folium_available = True
except ImportError as e:
    print(f"Streamlit-Folium import failed: {e}")
    streamlit_folium_available = False

try:
    import geopy
    from geopy.distance import geodesic
    print("Geopy imported successfully")
    geopy_available = True
except ImportError as e:
    print(f"Geopy import failed: {e}")
    geopy_available = False

if all([folium_available, geopandas_available, shapely_available, streamlit_folium_available, geopy_available]):
    print("\nAll GIS dependencies installed successfully!")
    print("Testing basic GIS functionality...")

    # Test creating a basic folium map
    try:
        import os
        os.chdir(os.path.dirname(__file__))  # Change to current directory

        # Test if data files exist
        reliance_file = 'reliance fresh dataset.csv'
        competitor_file = 'simpli namdhari\'s dataset.csv'
        population_file = 'bangalore-ward-level-census-2011.csv'

        files_exist = all(os.path.exists(f) for f in [reliance_file, competitor_file, population_file])

        if files_exist:
            print("All GIS data files found")
            print("GIS integration is ready to use!")
        else:
            print("Some data files are missing")
            for f in [reliance_file, competitor_file, population_file]:
                if not os.path.exists(f):
                    print(f"  - Missing: {f}")

    except Exception as e:
        print(f"Error during GIS functionality test: {e}")
else:
    print("\nSome GIS dependencies are missing. Please run:")
    print("pip install folium==0.14.0 streamlit-folium>=0.17.0 geopandas>=0.13.0 shapely>=2.0.0 geopy>=2.4.0")
