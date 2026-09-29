import pandas as pd

df = pd.read_csv('some path/file ...')

counts = df['Subject'].value_counts()
over_4 = counts[counts > 4]

if over_4.empty:
    print("All subjects occur at most 4 times.")
else:
    print("Subjects occurring more than 4 times:")
    for subject, count in over_4.items():
        print(f"{subject}: {count}")