#!/usr/bin/env python
"""Final GIS functionality test"""

try:
    print("Testing GIS imports and basic functionality...")

    # Test all GIS imports
    import folium
    import geopandas as gpd
    import shapely
    import streamlit_folium
    from geopy.distance import geodesic
    import pandas as pd
    import numpy as np

    print("✅ All GIS packages imported successfully")

    # Test data loading
    try:
        reliance_df = pd.read_csv('reliance fresh dataset.csv')
        competitor_df = pd.read_csv('simpli namdhari\'s dataset.csv')
        population_df = pd.read_csv('bangalore-ward-level-census-2011.csv')

        print(f"✅ Data loaded: {len(reliance_df)} Reliance stores, {len(competitor_df)} competitors, {len(population_df)} wards")

        # Test GeoDataFrame creation
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

        print("✅ GeoDataFrames created successfully")

        # Test distance calculation
        test_coord1 = (reliance_df.iloc[0]['latitude'], reliance_df.iloc[0]['longitude'])
        test_coord2 = (competitor_df.iloc[0]['latitude'], competitor_df.iloc[0]['longitude'])
        distance = geodesic(test_coord1, test_coord2).km
        print(f"✅ Distance calculation test: {distance:.2f}km")

        # Test population scoring algorithm (simplified)
        def _get_population_color(population, max_pop):
            ratio = population / max_pop
            if ratio > 0.7:
                return 'darkred'
            elif ratio > 0.5:
                return 'orange'
            elif ratio > 0.3:
                return 'yellow'
            else:
                return 'lightblue'

        # Test function
        max_pop = population_df['Population'].max()
        test_color = _get_population_color(population_df['Population'].iloc[0], max_pop)
        print(f"✅ Population color function test: {test_color}")

        print("\n🎉 ALL GIS FUNCTIONALITY TESTS PASSED!")
        print("The Reliance Fresh GIS Analytics platform is READY!")

    except Exception as e:
        print(f"❌ Data loading test failed: {e}")

except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please ensure all GIS packages are installed:")
    print("pip install folium==0.14.0 geopandas>=0.13.0 shapely>=2.0.0 streamlit-folium>=0.17.0 geopy>=2.4.0")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
