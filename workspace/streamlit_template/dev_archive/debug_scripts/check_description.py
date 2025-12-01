import geopandas as gpd
import fiona

try:
    fiona.drvsupport.supported_drivers['KML'] = 'rw'
    gdf = gpd.read_file('bbmp_final_new_wards.kml', driver='KML')
    print("First Description:", gdf['Description'].iloc[0])
except Exception as e:
    print(f"Error reading KML: {e}")
