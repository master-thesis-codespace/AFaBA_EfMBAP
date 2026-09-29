import pandas as pd

df = pd.read_csv('some path/file ...')
df_bas1 = df[df['session'] == 'BAS1']

df_bas1.to_csv('some path/file ...', index=False)
print(f"Rows kept: {len(df_bas1)}")