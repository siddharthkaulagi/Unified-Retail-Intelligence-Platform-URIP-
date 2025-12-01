import pandas as pd

try:
    df = pd.read_csv('bbmp_wards_full_merged.csv')
    print("CSV Columns:", df.columns.tolist())
    if 'Population' in df.columns:
        # Clean population column (remove commas if string)
        if df['Population'].dtype == object:
             df['Population'] = df['Population'].astype(str).str.replace(',', '')
        
        total_pop = pd.to_numeric(df['Population'], errors='coerce').sum()
        print(f"CSV Total Population: {total_pop:,}")
        print(f"CSV Row Count: {len(df)}")
    else:
        print("Population column not found in CSV")

except Exception as e:
    print(f"Error: {e}")
