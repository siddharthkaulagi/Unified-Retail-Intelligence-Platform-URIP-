import geopandas as gpd
import pandas as pd
import fiona

try:
    fiona.drvsupport.supported_drivers['KML'] = 'rw'
    gdf = gpd.read_file('bbmp_final_new_wards.kml', driver='KML')
    print("KML Names:", gdf['Name'].head(5).tolist())
except Exception as e:
    print(f"Error reading KML: {e}")

try:
    df = pd.read_csv('bbmp_wards_full_merged.csv')
    print("CSV Ward Names:", df['Ward Name'].head(5).tolist())
except Exception as e:
    print(f"Error reading CSV: {e}")
