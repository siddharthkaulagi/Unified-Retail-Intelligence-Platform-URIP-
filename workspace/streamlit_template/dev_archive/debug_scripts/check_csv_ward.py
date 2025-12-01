import pandas as pd
df = pd.read_csv('bbmp_wards_full_merged.csv')
print("Is 'Chowdeswari Ward' in CSV?", 'Chowdeswari Ward' in df['Ward Name'].values)
