import pandas as pd
import xml.etree.ElementTree as ET

# 1. Load CSV Names
df = pd.read_csv('bbmp_wards_full_merged.csv')
csv_names = set(df['Ward Name'].str.strip().tolist())

# 2. Load KML Names
tree = ET.parse('bbmp_final_new_wards.kml')
root = tree.getroot()
ns = {'kml': 'http://www.opengis.net/kml/2.2'}

kml_names = []
for placemark in root.findall('.//kml:Placemark', ns):
    ward_name = None
    extended_data = placemark.find('kml:ExtendedData', ns)
    if extended_data is not None:
        schema_data = extended_data.find('kml:SchemaData', ns)
        if schema_data is not None:
            for simple_data in schema_data.findall('kml:SimpleData', ns):
                if simple_data.get('name') == 'WARD_NAME':
                    ward_name = simple_data.text
                    break
    if ward_name:
        kml_names.append(ward_name.strip())

kml_names_set = set(kml_names)

# 3. Compare
missing_in_csv = kml_names_set - csv_names

print("\n--- Names in KML but NOT in CSV (Potential Mismatches) ---")
for name in sorted(list(missing_in_csv)):
    print(f"'{name}'")
