import xml.etree.ElementTree as ET

try:
    tree = ET.parse('bbmp_final_new_wards.kml')
    root = tree.getroot()
    
    # KML namespace
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    # Find first Placemark
    placemark = root.find('.//kml:Placemark', ns)
    if placemark is not None:
        print("Found Placemark")
        # Look for ExtendedData/SchemaData/SimpleData
        extended_data = placemark.find('kml:ExtendedData', ns)
        if extended_data is not None:
            schema_data = extended_data.find('kml:SchemaData', ns)
            if schema_data is not None:
                for simple_data in schema_data.findall('kml:SimpleData', ns):
                    name = simple_data.get('name')
                    text = simple_data.text
                    print(f"SimpleData: {name} = {text}")
            else:
                print("No SchemaData")
        else:
            print("No ExtendedData")
    else:
        print("No Placemark found")

except Exception as e:
    print(f"Error: {e}")
