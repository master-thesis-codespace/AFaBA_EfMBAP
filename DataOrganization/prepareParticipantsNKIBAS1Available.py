import pandas as pd

subjects_to_remove = [
    "some identifiers ...",
]

df = pd.read_csv('some path/file ...')
df_filtered = df[~df['subject'].isin(subjects_to_remove)]

print(f"Rows before: {len(df)}")
print(f"Rows after: {len(df_filtered)}")
print(f"Removed: {len(df) - len(df_filtered)}")

df_filtered.to_csv('some path/file ...', index=False)