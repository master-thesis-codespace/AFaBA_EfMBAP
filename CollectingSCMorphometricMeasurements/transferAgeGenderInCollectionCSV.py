import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

extract = pd.read_excel('./participants.ods', engine='odf')
insert = pd.read_csv('some path/file ...')

#print(insert.head)

for dataset in ['MMRR', 'ABRIM','SALD','Swedish','Noddi','SWU-slim','HBN']:
    subjects = extract.loc[extract['group2'] == dataset, ['ID']]
    #print(subjects)

    for currentSubject in subjects['ID']:
        #print(currentSubject)
        if dataset == 'MMRR':
            token = currentSubject.split('MMRR')
            insertSubject = f"sub-{token[1]}"
            #print(insertSubject)
        elif dataset == 'ABRIM':
            token = currentSubject.split('ABRIM1')
            insertSubject = f"sub-{token[1]}"
            #print(insertSubject)
        elif dataset == 'SALD':
            token = currentSubject.split('SALD')
            insertSubject = f"sub-{token[1]}"
            #print(insertSubject)
        elif dataset == 'Swedish':
            token = currentSubject.split('SS')
            insertSubject = f"sub-Ctrl{token[1]}"
            #print(insertSubject)
        elif dataset == 'Noddi':
            token = currentSubject.split('UCSF_')
            token2 = token[1].split('.')
            insertSubject = f"sub-{token2[0]}"
            #print(insertSubject)
        elif dataset == 'SWU-slim':
            token = currentSubject.split('SWU')
            insertSubject = f"sub-{token[1]}"
            #print(insertSubject)
        elif dataset == 'HBN':
            token = currentSubject.split('HBN_')
            insertSubject = f"sub-{token[1]}"
            #print(insertSubject)
        #print(insertSubject)

        ageGender = extract.loc[extract['ID'] == currentSubject, ['Age','Sex']]
        insert.loc[insert['Subject'] == insertSubject, 'Age'] = ageGender['Age'].iloc[0]
        insert.loc[insert['Subject'] == insertSubject, 'Gender'] = ageGender['Sex'].iloc[0]


for dataset in ['AHDC','Calgary_p','Calgary_s','Calgary_g','DVPFC','IXI','MR-ART','NIMH','OAS3','OAS4','PAINGEN','QTIM']:
    subjects = extract.loc[extract['group2'] == dataset, ['ID']]
    #print(subjects)

    for currentSubject in subjects['ID']:
        #print(currentSubject)
        ageGender = extract.loc[extract['ID'] == currentSubject, ['Age','Sex']]
        insert.loc[insert['Subject'] == currentSubject, 'Age'] = ageGender['Age'].iloc[0]
        insert.loc[insert['Subject'] == currentSubject, 'Gender'] = ageGender['Sex'].iloc[0]

insert.to_csv("some path/file ...", index=False)