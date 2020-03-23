import pickle
import os
import numpy as np
from collections import Counter
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
plt.style.use('ggplot')

def getMeanLayerPref_perSubject(subjectID = '018',
                                score_name = 'benchmark=Pereira2018-encoding,model=gpt2-xl,subsample=None.pkl',
                                metric = 'mean',
                                categorize = False, # Split into early, mid, late
                                plotROIs = True,
                                plotNetworks = False):

    # Define variables and file names
    fname = score_name.split('=')
    pngfile_name = subjectID + '_ROI_' + metric + '_LayerPref_' + fname[1][:-5] + fname[2][:-10] + '.png'
    pngfile_name = pngfile_name.replace(',', '_')
    pngfile_name = pngfile_name.replace('-', '_')
    
    if plotNetworks:
        pngfile_name2 = subjectID + '_Network_' + metric + '_LayerPref_' + fname[1][:-5] + fname[2][:-10] + '.png'
        pngfile_name2 = pngfile_name2.replace(',', '_')
        pngfile_name2 = pngfile_name2.replace('-', '_')
    
    # Load score objects
    with open(score_name, 'rb') as f:  
        result = pickle.load(f)
    
    score = result['data']

    roi_names_all = score.raw.roi.values
    roi_names = np.unique(roi_names_all)

    center_score = score[{'aggregation': [agg == 'center' for agg in score['aggregation'].values]}].squeeze('aggregation')
    s = center_score.raw[{'neuroid': [subject == subjectID for subject in score.raw['subject'].values]}] 
    
    model2 = s.mean(['split', 'experiment']) # Not grouped by neuroid
    
    _, index_model = np.unique(model2['neuroid_id'], return_index = True)
    model3 = model2.isel(neuroid = np.sort(index_model))
    
    # Squeeze the layer dim
    model4 = model3.squeeze()
    
    # Sort by unique neuroid_id
    model5 = model4.sortby('neuroid_id')
    
    # Remove NaNs for subject 018
    if np.isnan(model5).any():
        nan_sum = np.isnan(model5).sum()
        print(nan_sum)
        
        model6 = model5.fillna(0)
    else:
        model6 = model5
    
    # Fetch the layer preference 
    model7 = model6.argmax('layer', skipna=True) # Argmax over layer dimension
    
    # Normalize scores or categorize
    if not categorize:
        model7_norm = model7/model7.max()
    if categorize:
        model7_norm = model7
        # Categorize into early (0-16), mid (17-32) and late (33-48) layers
        model7_norm[model7_norm <= 16] = 1
        model7_norm[(model7_norm >= 17) & (model7_norm <= 32)] = 2 
        model7_norm[model7_norm >= 33] = 3
 
    # Group by ROIs and average across ROIs 
    d = {}
    for roiID in roi_names:
        model7_roi = model7_norm[{'neuroid': [roi == roiID for roi in model7_norm['roi'].values]}]
        if metric == 'mean':
            roi_metric = model7_roi.mean().values
        if metric == 'median':
            roi_metric = model7_roi.median().values
        if metric == 'mode':
            roi_metric = stats.mode(model7_roi.values)
            roi_metric = roi_metric[0][0]
        roi_std = model7_roi.std().values
        
        d[roiID + '_metric'] = [roi_metric, roi_std]

    ROInames_metric = []
    ROIscores_metric = []
    for item in d.items():
        kval = item[0]
        sval = item[1]
        ROInames_metric.append(kval)
        ROIscores_metric.append(sval)
    
    if plotROIs:
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
        
        ROInames_plot = [name[ : -7] for name in ROInames_metric]
        
        name_score = list(zip(ROInames_plot, ROIscores_metric))
        name_score2 = name_score.copy()
        name_score2.sort(key=lambda x: names_ordered.index(x[0]))
        
        color_lst = ['brown']*12 + ['orange']*20 + ['olivedrab']*10 + ['dodgerblue']*4 + ['purple']*6 # SHOULD BE 12 DMN
        
        custom_lines = [Line2D([0], [0], color='brown', lw=2),
                        Line2D([0], [0], color='orange', lw=2),
                        Line2D([0], [0], color='olivedrab', lw=2),
                        Line2D([0], [0], color='dodgerblue', lw=2),
                        Line2D([0], [0], color='purple', lw=2)]
        
        names = [item[0] for item in name_score2] 
        scores = [item[1][0] for item in name_score2] 
        stds = [item[1][1] for item in name_score2] 
        
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(names)), scores, yerr = stds,
                error_kw = dict(lw=1, capsize=1, capthick=1),
                ecolor = 'grey', align = 'edge', width = 0.4, color = color_lst)
        plt.xticks(range(len(names)), names, size='small', rotation=90)
        plt.ylabel('Layer Number') 
        if not categorize:
            plt.ylim(0,1.1)
            plt.title('GPT2-xl - Normalized Layer Preference Across ROIs (' + metric + '), Subject ' + subjectID)
        if categorize:
            plt.ylim(0,4)
            plt.title('GPT2-xl - Layer Preference (Early, Mid, Late) Across ROIs (' + metric + '), Subject ' + subjectID)
        # plt.legend(custom_lines, ['Language', 'Multiple Demand', 'Default Mode', 'Auditory', 'Visual'], loc='best')
        plt.tight_layout()
        plt.savefig(pngfile_name, dpi=240)

    # For plotting networks
    if plotNetworks:
        lang_score = np.mean(scores[0:12])
        lang_std = np.std(scores[0:12])
        
        md_score = np.mean(scores[12:32])
        md_std = np.std(scores[12:32])
        
        dmn_score = np.mean(scores[32:42])
        dmn_std = np.std(scores[32:42])
        
        aud_score = np.mean(scores[42:46])
        aud_std = np.std(scores[42:46])
        
        vis_score = np.mean(scores[46:])
        vis_std = np.std(scores[46:])
        
        plt.figure(figsize=(10, 5))
        plt.bar(range(5), [lang_score, md_score, dmn_score, aud_score, vis_score],
                yerr = [lang_std, md_std, dmn_std, aud_std, vis_std], align='edge',
                error_kw = dict(lw=1, capsize=1, capthick=1), ecolor='grey',
                width=0.4, color = ['brown', 'orange', 'olivedrab', 'dodgerblue', 'purple'])
        plt.xticks(range(5), ['Language', 'Multiple Demand', 'Default Mode', 'Auditory', 'Visual'], rotation=90)#, alignment='center')
        # plt.xlabel(['Language', 'Multiple Demand', 'Default Mode', 'Auditory', 'Visual'], rotation=90) # alignment='center')
        plt.title('Normalized Layer Preference Across Networks (' + metric +'), Subject ' + subjectID)
        plt.ylabel('Layer Number') 
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(pngfile_name2, dpi=240)

