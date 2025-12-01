import xml.etree.ElementTree as ET

try:
    tree = ET.parse('bbmp_final_new_wards.kml')
    root = tree.getroot()
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    field_names = set()
    
    count = 0
    for placemark in root.findall('.//kml:Placemark', ns):
        extended_data = placemark.find('kml:ExtendedData', ns)
        if extended_data is not None:
            schema_data = extended_data.find('kml:SchemaData', ns)
            if schema_data is not None:
                for simple_data in schema_data.findall('kml:SimpleData', ns):
                    field_names.add(simple_data.get('name'))
        count += 1
        if count > 5: break
        
    print("Available fields:", field_names)

except Exception as e:
    print(f"Error: {e}")
