import xml.etree.ElementTree as ET

tree = ET.parse('bbmp_final_new_wards.kml')
root = tree.getroot()
ns = {'kml': 'http://www.opengis.net/kml/2.2'}

# Find first Placemark and list all SimpleData names
placemark = root.find('.//kml:Placemark', ns)
if placemark is not None:
    extended_data = placemark.find('kml:ExtendedData', ns)
    if extended_data is not None:
        schema_data = extended_data.find('kml:SchemaData', ns)
        if schema_data is not None:
            print("Available SimpleData fields:")
            for simple_data in schema_data.findall('kml:SimpleData', ns):
                print(f"- {simple_data.get('name')}: {simple_data.text}")
