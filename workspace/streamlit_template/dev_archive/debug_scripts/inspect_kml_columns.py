import geopandas as gpd
import fiona

try:
    fiona.drvsupport.supported_drivers['KML'] = 'rw'
    gdf = gpd.read_file('bbmp_final_new_wards.kml', driver='KML')
    print("Columns:", gdf.columns.tolist())
    if not gdf.empty:
        print("First row properties:", gdf.iloc[0].to_dict())
except Exception as e:
    print(f"Error: {e}")
