import fiona

try:
    fiona.drvsupport.supported_drivers['KML'] = 'rw'
    # Force KML driver
    with fiona.open('bbmp_final_new_wards.kml', 'r', driver='KML') as source:
        print("Driver:", source.driver)
        print("First feature properties keys:", list(next(iter(source))['properties'].keys()))
        print("First feature properties values:", list(next(iter(source))['properties'].values()))
except Exception as e:
    print(f"Error: {e}")
