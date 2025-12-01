import xml.etree.ElementTree as ET
import pandas as pd

try:
    tree = ET.parse('bbmp_final_new_wards.kml')
    root = tree.getroot()
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    populations = []
    
    for placemark in root.findall('.//kml:Placemark', ns):
        extended_data = placemark.find('kml:ExtendedData', ns)
        if extended_data is not None:
            schema_data = extended_data.find('kml:SchemaData', ns)
            if schema_data is not None:
                for simple_data in schema_data.findall('kml:SimpleData', ns):
                    name = simple_data.get('name')
                    if name == 'population':
                        populations.append(simple_data.text)
                        
    print(f"Found {len(populations)} population values.")
    print("First 10 values:", populations[:10])
    
    # Check if they convert to numbers
    numeric_pops = []
    for p in populations:
        try:
            numeric_pops.append(float(p))
        except:
            pass
            
    if numeric_pops:
        s = pd.Series(numeric_pops)
        print("Min:", s.min())
        print("Max:", s.max())
        print("Mean:", s.mean())
        print("Unique values:", s.nunique())
    else:
        print("No numeric population values found.")

except Exception as e:
    print(f"Error: {e}")
