import fiona

try:
    fiona.drvsupport.supported_drivers['KML'] = 'rw'
    with fiona.open('bbmp_final_new_wards.kml', 'r') as source:
        print("Driver:", source.driver)
        print("Schema:", source.schema)
        print("First feature properties:", next(iter(source))['properties'])
except Exception as e:
    print(f"Error: {e}")
