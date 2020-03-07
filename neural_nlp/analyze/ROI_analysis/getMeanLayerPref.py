import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Function inputs
score_name = 'benchmark=Pereira2018-encoding,model=gpt2-xl,subsample=None.pkl'

# Define variables and file names
SPM_dim = (79,95,69)
fname = score_name.split('=')
matfile_name = 'mean_layer_pref_' + fname[1][:-5] + fname[2][:-10] + '.mat'
matfile_name = matfile_name.replace(',', '_')
matfile_name = matfile_name.replace('-', '_')

# Load score objects
with open(score_name, 'rb') as f:  
    result = pickle.load(f)

score = result['data']

roi_names_all = score.raw.roi.values
roi_names = np.unique(roi_names_all)

## FOR MODEL ##
center_score = score[{'aggregation': [agg == 'center' for agg in score['aggregation'].values]}].squeeze('aggregation')
model2 = center_score.raw.mean(['split', 'experiment']) # Not grouped by neuroid

_, index_model = np.unique(model2['neuroid_id'], return_index = True)
model3 = model2.isel(neuroid = np.sort(index_model))

# Squeeze the layer dim
model4 = model3.squeeze()

# Sort by unique neuroid_id
model5 = model4.sortby('neuroid_id')

# Remove NaNs 
if np.isnan(model5).any() == True:
    nan_sum = np.isnan(model5).sum()
    print(nan_sum)
    
    model6 = model5.fillna(0)
else:
    model6 = np.copy(model5)

# Fetch the layer preference 
model7 = model6.argmax('layer', skipna=True) # Argmax over layer dimension

# Normalize scores 
model7_norm = model7/model7.max()

# Group by ROIs and average across subjects and ROIs 
d = {}
for roiID in roi_names:
    model7_roi = model7_norm[{'neuroid': [roi == roiID for roi in model7_norm['roi'].values]}]
    roi_mean = model7_roi.mean()
    # print('Mean of ROI ', roiID, roi_mean)
    d[roiID + '_mean'] = roi_mean.mean().values


ROInames_mean = []
ROIscores_mean = []
for item in d.items():
    kval = item[0]
    sval = item[1]
    ROInames_mean.append(kval)
    ROIscores_mean.append(sval)
    
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

ROInames_plot = [name[ : -5] for name in ROInames_mean]

name_score = list(zip(ROInames_plot, ROIscores_mean))
name_score2 = name_score.copy()
name_score2.sort(key=lambda x: names_ordered.index(x[0]))

plt.style.use('ggplot')

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
plt.title('GPT2-xl - Normalized Layer Preference Across ROIs (All Subjects)')
plt.ylabel('Layer Number') 
plt.ylim(0,1)
plt.legend(custom_lines, ['Language', 'Multiple Demand', 'Default Mode', 'Auditory', 'Visual'], loc=3)
plt.show()


