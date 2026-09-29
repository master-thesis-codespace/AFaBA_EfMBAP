import pandas as pd

df = pd.read_excel('some path/file ...', engine='odf')          #load the ODS file

df = df.drop(columns=['ehi', 'filepath'])                                       #drop 'ehi' and 'filepath' columns
df = df.dropna(subset=['subject', 'session', 'gender', 'age'])                  #keep only rows where all four columns have valid values
df = df.drop_duplicates(subset=['subject', 'session', 'gender', 'age'])         #deduplicate: one row per unique combination of subject-session-gender-age
df = df.reset_index(drop=True)                                                  #reset index

print("Shape:", df.shape)
print(df.head(10))

df.to_csv('some path/file ...', index=False)                               #save to CSV