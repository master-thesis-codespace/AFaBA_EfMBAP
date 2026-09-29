import pandas as pd

df = pd.read_csv('some path/file ...')

counts = df['subject'].value_counts()                                           #get subjects that appear exactly once
unique_subjects = counts[counts == 1].index
df_clean = df[df['subject'].isin(unique_subjects)].reset_index(drop=True)

print(f"Rows before: {len(df)}")
print(f"Rows after: {len(df_clean)}")
print(f"Removed: {len(df) - len(df_clean)}")

df_clean.to_csv('some path/file ...', index=False)