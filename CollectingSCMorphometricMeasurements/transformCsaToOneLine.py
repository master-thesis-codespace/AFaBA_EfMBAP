import pandas as pd
import numpy as np

INPUT_FILE = 'some path/file ...'
OUTPUT_FILE = f'{INPUT_FILE}_oneLined.csv'

DROP_COLS = ['SCT Version', 'Slice (I->S)', 'DistancePMJ']                                      # Columns to drop

MEAS_COLS = [                                                                                   # Measurement columns (without vertebral-level suffix)
    'MEAN(area)', 'STD(area)', 
    'MEAN(angle_AP)', 'STD(angle_AP)',
    'MEAN(angle_RL)', 'STD(angle_RL)',
    'MEAN(diameter_AP)', 'STD(diameter_AP)',
    'MEAN(diameter_RL)', 'STD(diameter_RL)',
    'MEAN(eccentricity)', 'STD(eccentricity)',
    'MEAN(orientation)', 'STD(orientation)',
    'MEAN(solidity)', 'STD(solidity)',
    'SUM(length)',
]

ID_COLS = ['Dataset', 'Subject', 'Age', 'Gender', 'Filename']                                   # Identity columns kept once per subject row

df = pd.read_csv(INPUT_FILE)

df = df.drop(columns=DROP_COLS)

vert_levels = sorted(df['VertLevel'].unique())                                                  # [1, 2, 3, 4]

rows = []
for (dataset, subject), grp in df.groupby(['Dataset', 'Subject'], sort=False):
    row = {col: grp[col].iloc[0] for col in ID_COLS}                                            # ID cols from first row
    for lvl in vert_levels:
        lvl_data = grp[grp['VertLevel'] == lvl]
        for col in MEAS_COLS:
            row[f'{col}_{lvl}'] = (lvl_data[col].values[0] if not lvl_data.empty else np.nan)
    rows.append(row)

result = pd.DataFrame(rows)

DROP_COLS2 = ['MEAN(area)_4', 'STD(area)_4', 'MEAN(angle_AP)_4', 'STD(angle_AP)_4', 'MEAN(angle_RL)_4', 'STD(angle_RL)_4', 'MEAN(diameter_AP)_4', 'STD(diameter_AP)_4', 'MEAN(diameter_RL)_4', 'STD(diameter_RL)_4', 'MEAN(eccentricity)_4', 'STD(eccentricity)_4', 'MEAN(orientation)_4', 'STD(orientation)_4', 'MEAN(solidity)_4', 'STD(solidity)_4', 'SUM(length)_4',
              'MEAN(angle_AP)_1', 'STD(angle_AP)_1', 'MEAN(angle_RL)_1', 'STD(angle_RL)_1',	'MEAN(orientation)_1', 'STD(orientation)_1', 'MEAN(angle_AP)_2', 'STD(angle_AP)_2', 'MEAN(angle_RL)_2', 'STD(angle_RL)_2', 'MEAN(orientation)_2', 'STD(orientation)_2', 'MEAN(angle_AP)_3', 'STD(angle_AP)_3', 'MEAN(angle_RL)_3', 'STD(angle_RL)_3', 'MEAN(orientation)_3', 'STD(orientation)_3']

result = result.drop(columns=DROP_COLS2)

result.to_csv(OUTPUT_FILE, index=False)
print(f"{result.shape[0]} rows x {result.shape[1]} cols to {OUTPUT_FILE}")