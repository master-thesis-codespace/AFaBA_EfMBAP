import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

df = pd.read_excel('./SAFcontrol_20260212.ods', engine='odf')

def age_statistics(df):
    age = df['Age'].dropna()
    
    metrics = {'Count': len(age), 'Mean': age.mean(), 'Median': age.median(), 'Std Dev': age.std(), 'Variance': age.var(), 'Min': age.min(), 'Max': age.max(), 'Range': age.max() - age.min(),}
    print("Age Distribution:")
    for key, val in metrics.items():
        if key == 'Count':
            print(f"  {key:<18}: {int(val)}")
        else:
            print(f"  {key:<18}: {val:.4f}")    
    return metrics

#used for Fig. 6 and 9
#bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, np.inf]                                                  #define age bins and labels
#labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+']

#used for Fig. 14
#bins = [30, 40, 50, 60, 70, 80, 90, np.inf]                                                             #define age bins and labels
#labels = ['30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+']

#used for Fig. 12
bins = [60, 70, 80, 90, np.inf]                                                                         #define age bins and labels
labels = ['60-69', '70-79', '80-89', '90+']

df['age_range'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
groups = sorted(df['group2'].unique())

def get_distinct_colors(n):
    colors = []
    cmaps = ['tab20', 'tab20b', 'tab20c']
    for cmap_name in cmaps:
        cmap = plt.cm.get_cmap(cmap_name)
        colors += [cmap(i / 20) for i in range(20)]
        if len(colors) >= n:
            break
    return colors[:n]

colors = get_distinct_colors(len(groups))

pivot = df.groupby(['age_range', 'group2'], observed=True).size().unstack(fill_value=0)
pivot = pivot.reindex(labels)

fig, ax = plt.subplots(figsize=(14, 6))
bottom = np.zeros(len(pivot))
bars = []
for i, group in enumerate(pivot.columns):
    values = pivot[group].values
    b = ax.bar(pivot.index, values, bottom=bottom, color=colors[i], label=group, edgecolor='white', linewidth=0.5)
    bars.append(b)
    bottom += values

totals = pivot.sum(axis=1).values                                                                       #calculate totals per age bin and annotate percentages above each bar
grand_total = totals.sum()

for i, total in enumerate(totals):
    point = 100 * total / grand_total
    ax.text(i, total + 5, f'{point:.2f}%', ha='center', va='bottom')

ax.set_xlabel('Age range [years]')
ax.set_ylabel('Absolute number of participants')
ax.set_title(f'SAF Healthy Controls Age distribution ({grand_total} subjects)')
ax.legend(title='Dataset', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('SAFHCAgeDistribution08032026.png', dpi=150, bbox_inches='tight')
plt.show()

sex_counts = df.groupby(['group2', 'Sex']).size().unstack(fill_value=0)
sex_counts = sex_counts.reindex(columns=[0, 1, 9], fill_value=0)

totalFemale = np.sum(sex_counts[0])
totalMale = np.sum(sex_counts[1])
totalOther = np.sum(sex_counts[9])
allParticipants = totalFemale + totalMale + totalOther

x = np.arange(len(sex_counts))
width = 0.35

fig2, ax2 = plt.subplots(figsize=(14, 6))                                                               #plot the gender distribution per cohort
bars0 = ax2.bar(x - width, sex_counts[0], width, label=f'Female ({totalFemale/allParticipants*100:.2f}%)', color='red', edgecolor='white')
bars1 = ax2.bar(x, sex_counts[1], width, label=f'Male ({totalMale/allParticipants*100:.2f}%)', color='blue', edgecolor='white')
bars2 = ax2.bar(x + width, sex_counts[9], width, label=f'Other ({totalOther/allParticipants*100:.2f}%)', color='green', edgecolor='white')

ax2.set_xlabel('Dataset')
ax2.set_ylabel('Absolute number of participants')
ax2.set_title(f'SAF Healthy Controls Gender distribution and participants per dataset ({allParticipants} subjects)')
ax2.set_xticks(x)
ax2.set_xticklabels(sex_counts.index, rotation=45, ha='right', fontsize=9)
ax2.legend(title='Gender')
plt.tight_layout()
plt.savefig('SAFHCGenderDistribution08032026.png', dpi=150, bbox_inches='tight')
plt.show()

age_statistics(df)                                                                                      #call it after loading the dataframe

fig = plt.figure(figsize=(16, 14))
#ax1 = fig.add_subplot(2, 2, (1, 2))                                                                     #one pie chart top, two bottom Fig. 8 and Fig. 11
ax1 = fig.add_subplot(1,1,1)                                                                            #only one pie chart bottom Fig. 13
#ax2 = fig.add_subplot(2, 2, 3)                                                                          #one pie chart top, two bottom Fig. 8 and Fig. 11
#ax3 = fig.add_subplot(2, 2, 4)                                                                          #one pie chart top, two bottom Fig. 8 and Fig. 11
#axes = [ax1, ax2, ax3]
axes = [ax1]

overall = df['Sex'].value_counts().sort_index()                                                         #firts pie chart with all genders
sex_label_map = {0: 'Female', 1: 'Male', 9: 'Other'}
sex_labels = [sex_label_map.get(s, str(s)) for s in overall.index]
sex_colors  = {0: 'red', 1: 'blue', 9: 'green'}
pie_colors  = [sex_colors.get(s, 'grey') for s in overall.index]

axes[0].pie(overall, labels=sex_labels, autopct='%1.2f%%', colors=pie_colors, startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}, textprops={'fontsize': 11})
axes[0].set_title(f'SAF Healthy Controls Gender Distribution ({grand_total} subjects)', pad = 15)

sex0_df = df[df['Sex'] == 0]                                                                            #second pie chart with contribution of cohorts to female
sex0_counts = sex0_df['group2'].value_counts().sort_index()
n0 = len(sex0_counts)
cmap0 = cm.Blues(np.linspace(0.3, 0.95, n0))
sex1_df = df[df['Sex'] == 1]                                                                            #third pie chart with contribution of cohorts to female
sex1_counts = sex1_df['group2'].value_counts().sort_index()
n1 = len(sex1_counts)
cmap1 = cm.Oranges(np.linspace(0.3, 0.95, n1))

wedges0, texts0, autotexts0 = axes[1].pie(sex0_counts.values, labels=sex0_counts.index, autopct='%1.0f%%', colors=cmap1, startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.2}, textprops={'fontsize': 8}, pctdistance=0.9)
for at in autotexts0:
    at.set_fontsize(7)
axes[1].set_title(f'SAF HC Dataset Contributions to #Female = {len(sex0_df)}', pad=15)

wedges1, texts1, autotexts1 = axes[2].pie(sex1_counts.values, labels=sex1_counts.index, autopct='%1.0f%%', colors=cmap0, startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.2}, textprops={'fontsize': 8}, pctdistance=0.9)
for at in autotexts1:
    at.set_fontsize(7)
axes[2].set_title(f'SAF HC Dataset Contributions to #Male = {len(sex1_df)}', pad=15)

plt.tight_layout()
plt.savefig('SAFHCGenderPieDistribution06032026.png', dpi=150, bbox_inches='tight')
plt.show()