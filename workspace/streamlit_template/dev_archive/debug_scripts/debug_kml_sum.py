import xml.etree.ElementTree as ET
import pandas as pd

try:
    print("Parsing KML...")
    tree = ET.parse('bbmp_final_new_wards.kml')
    root = tree.getroot()
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    total_pop = 0
    count_pop = 0
    zero_pop_count = 0
    missing_pop_count = 0
    
    all_fields = set()

    ward_details = []

    for placemark in root.findall('.//kml:Placemark', ns):
        pop_value = 0
        ward_name = "Unknown"
        
        extended_data = placemark.find('kml:ExtendedData', ns)
        if extended_data is not None:
            schema_data = extended_data.find('kml:SchemaData', ns)
            if schema_data is not None:
                found_pop = False
                for simple_data in schema_data.findall('kml:SimpleData', ns):
                    name = simple_data.get('name')
                    value = simple_data.text
                    all_fields.add(name)
                    
                    if name in ['name_en', 'WARD_NAME']:
                        ward_name = value
                    
                    if name in ['population', 'POP_TOTAL', 'Population']:
                        try:
                            pop_value = int(float(value))
                            found_pop = True
                        except:
                            pass
                
                if not found_pop:
                    missing_pop_count += 1
        
        if pop_value > 0:
            total_pop += pop_value
            count_pop += 1
        else:
            zero_pop_count += 1
            ward_details.append((ward_name, pop_value))

    print(f"Total Population Sum: {total_pop:,}")
    print(f"Wards with Population > 0: {count_pop}")
    print(f"Wards with Population = 0: {zero_pop_count}")
    print(f"Wards with Missing Population Field: {missing_pop_count}")
    print(f"All Fields Found: {sorted(list(all_fields))}")
    
    if zero_pop_count > 0:
        print("\nSample Wards with 0 Population:")
        for name, pop in ward_details[:10]:
            print(f"- {name}: {pop}")

except Exception as e:
    print(f"Error: {e}")
