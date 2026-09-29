import pandas as pd

ids_to_drop = [
    'some identifiers ...', 
]

df = pd.read_csv('some path/file ...')                                   # Load the CSV

print(f"Original shape: {df.shape}")
df_filtered = df[~df['pat.id'].isin(ids_to_drop)]                                                   #drop rows where 'pat.id' is in the list

print(f"Filtered shape: {df_filtered.shape}")
print(f"Rows dropped: {df.shape[0] - df_filtered.shape[0]}")

df_filtered.to_csv('some path/file ...v', index=False)           # Save the result
print("Saved to 'some path/file ...'")