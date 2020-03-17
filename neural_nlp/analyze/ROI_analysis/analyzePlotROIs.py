# Imports
import pickle
import os
import numpy as np
from collections import Counter
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.style.use('ggplot')

f_name = 'ROIanalysis_Pereira2018_encoding_gpt2_xl.pkl'

# Load object
with open(f_name, 'rb') as f:  
    result = pickle.load(f)
    
d = result[0]

# GPT-xl
ROInames_mean = []
ROIscores_mean = []
for item in d.items():
    kval = item[0]
    sval = item[1]
    if kval.endswith('mean') == True:
        ROInames_mean.append(kval)
        ROIscores_mean.append(sval)

ROInames_plot = [name[ : -5] for name in ROInames_mean]

# Original ordering
plt.figure(figsize=(10, 5))
plt.bar(range(len(ROInames_plot)), ROIscores_mean, align='edge', width=0.3)
plt.xticks(range(len(ROInames_plot)), ROInames_plot, size='small', rotation=90)
plt.show()

# Intended order: 
names_ordered = ['LH_PostTemp', 'LH_AntTemp', 'LH_IFG', 'LH_IFGorb', 'LH_MFG', \
                 'LH_AngG', 'RH_PostTemp', 'RH_AntTemp', 'RH_IFG', 'RH_IFGorb', \
                 'RH_MFG', 'RH_AngG', 'LH_postParietal', 'LH_midParietal', 'LH_antParietal', \
                 'LH_supFrontal', 'LH_Precentral_A_PrecG', 'LH_Precental_B_IFGop', 'LH_midFrontal', \
                 'LH_midFrontalOrb', 'LH_insula', 'LH_medialFrontal', 'RH_postParietal', \
                 'RH_midParietal', 'RH_antParietal', 'RH_supFrontal', 'RH_Precentral_A_PrecG', \
                 'RH_Precental_B_IFGop', 'RH_midFrontal', 'RH_midFrontalOrb', 'RH_insula', 'RH_medialFrontal', \
                 'LH_FrontalMed', 'LH_PostCing', 'LH_TPJ', 'LH_MidCing', 'LH_STGorInsula', \
                 'LH_AntTemp', 'RH_FrontalMed', 'RH_PostCing', 'RH_TPJ', 'RH_MidCing', 'RH_STGorInsula', \
                 'RH_AntTemp', 'LH_TE11', 'LH_TE12', 'RH_TE11', 'RH_TE12', 'LH_OccipInf', 'LH_OccipMid', \
                 'LH_OccipSup', 'RH_OccipInf', 'RH_OccipMid', 'RH_OccipSup']  
                 
# First 12: lang, 20 MD, 12 DMN, 4 aud, 6 vis

# Test ROI discrepany
list(set(names_ordered) - set(ROInames_plot))

s = set(ROInames_plot)
temp3 = [x for x in names_ordered if x not in s]

# Re-order names
name_score = list(zip(ROInames_plot, ROIscores_mean))
name_score2 = name_score.copy()
name_score2.sort(key=lambda x: names_ordered.index(x[0]))

color_lst = ['brown']*12 + ['orange']*20 + ['olivedrab']*10 + ['dodgerblue']*4 + ['purple']*6 # SHOULD BE 12 DMN


custom_lines = [Line2D([0], [0], color='brown', lw=2),
                Line2D([0], [0], color='orange', lw=2),
                Line2D([0], [0], color='olivedrab', lw=2),
                Line2D([0], [0], color='dodgerblue', lw=2),
                Line2D([0], [0], color='purple', lw=2)]


names = [item[0] for item in name_score2] 
scores = [item[1] for item in name_score2] 

plt.figure(figsize=(10, 5))
plt.bar(range(len(names)), scores, align='edge', width=0.3, color = color_lst)
plt.xticks(range(len(names)), names, size='small', rotation=90)
plt.title('GPT2-xl Normalized Scores Across ROIs')
plt.ylabel('Normalized Predictivity (r / c)')
plt.legend(custom_lines, ['Language', 'Multiple Demand', 'Default Mode', 'Auditory', 'Visual'], loc=2)
plt.show()
