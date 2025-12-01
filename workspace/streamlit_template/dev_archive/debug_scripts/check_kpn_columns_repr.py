import pandas as pd

try:
    df = pd.read_excel('KPN fresh dataset.xlsx')
    print("Columns repr:", [repr(c) for c in df.columns])
except Exception as e:
    print(f"Error reading excel: {e}")
