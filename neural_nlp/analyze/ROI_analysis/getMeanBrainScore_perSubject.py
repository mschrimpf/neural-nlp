import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import datetime 
import seaborn
from stats import is_significant, _permutation_test
import xlsxwriter 
  
# Plot specifications
seaborn.set(context='talk')
seaborn.set_style("whitegrid", {'axes.grid': False})
plt.rc('axes', edgecolor='black')
plt.rc('axes', edgecolor='black')
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

def getMeanBrainScore_perSubject(subjectID = '018', 
                                score_name = 'benchmark=Pereira2018-encoding,model=gpt2-xl,subsample=None.pkl',
                                plotROIs = True,
                                plotNetworks = False):
    # Extracts brainscores for a particular subject. Means across each ROI. For the best model layer (similar to plotting)
    # Computes permutation statistics for lang versus the rest of the networks.
    
    # Define variables and file names
    timestamp = '{:%Y-%m-%d}'.format(datetime.datetime.now())
    fname = score_name.split('=')
    pngfile_name = subjectID + '_ROI_' + timestamp + fname[1][:-5] + fname[2][:-10] + '.png'
    pngfile_name = pngfile_name.replace(',', '_')
    pngfile_name = pngfile_name.replace('-', '_')
    
#    csvfile_name = subjectID + '_ROI_sizes_' + timestamp + '_' + fname[2][:-10] + '.csv'
    statsfile_name = subjectID + '_stats_' + timestamp + '_' + fname[2][:-10] + '.csv'
    workbook = xlsxwriter.Workbook(subjectID + '_stats_' + timestamp + '_' + fname[2][:-10] + '.xlsx') 

    if plotNetworks:
        pngfile_name2 = subjectID + '_Network_BrainScore_' + timestamp + '_' + fname[1][:-5] + fname[2][:-10] + '.png'
        pngfile_name2 = pngfile_name2.replace(',', '_')
        pngfile_name2 = pngfile_name2.replace('-', '_')
    
    # Load score objects
    with open(score_name, 'rb') as f:  
        result = pickle.load(f)
    
    score = result['data']
    
    roi_names_all = score.raw.roi.values
    roi_names = np.unique(roi_names_all)
    
    # Find best layer
    center_score = score[{'aggregation': [agg == 'center' for agg in score['aggregation'].values]}].squeeze('aggregation')
    layer_score = center_score.mean(['atlas', 'experiment'])
    best_layer = layer_score['layer'][layer_score == layer_score.max()].values[0]
    
    # Extract best layer and mean over splits and experiments
    model = score.raw[{'layer': [layer == best_layer for layer in score.raw['layer'].values]}] 
    model2 = model.mean(['split', 'experiment']) # Not grouped by neuroid
    
    # Extract subject of interest
    model2_sub = model2[{'neuroid': [subject == subjectID for subject in model2['subject'].values]}] 
    
    # Omit filtering for unique neuroIDs when not plotting (include all values in analysis)
#    _, index_model = np.unique(model2_sub['neuroid_id'], return_index = True)
#    model3 = model2_sub.isel(neuroid = np.sort(index_model))
    
    # Squeeze the layer dim
    model4 = model2_sub.squeeze() # If filtering for unique neuroIDs: model3.squeeze()
    
    # Sort by unique neuroid_id
    model5 = model4.sortby('neuroid_id')
    
    # Remove NaNs
    if np.isnan(model5).any() == True:
        nan_sum = np.isnan(model5).sum()
        print(nan_sum)
        
        model6 = model5.fillna(0)
    else:
        model6 = model5
    
    # Group by ROIs and average across ROIs 
    d = {}
    for roiID in roi_names:
        model6_roi = model6[{'neuroid': [roi == roiID for roi in model6['roi'].values]}]
        
        unique_atlas = np.unique(model6_roi.atlas.values)
        if len(unique_atlas) > 1:

            # Disentangle LH/RH AntTemp for lang and DMN
            model6_roi_lang = model6_roi[{'neuroid': [atlas == 'language' for atlas in model6_roi['atlas'].values]}]
            model6_roi_dmn = model6_roi[{'neuroid': [atlas == 'DMN' for atlas in model6_roi['atlas'].values]}]
            
            roiID_lang = roiID + '_lang'
            roiID_dmn = roiID + '_DMN'
            
            # Split into language voxels
            roi_mean_lang = model6_roi_lang.mean().values
            roi_std_lang = model6_roi_lang.std().values
            roi_sem_lang = roi_std_lang/np.sqrt(len(model6_roi_lang.values))
            d[roiID_lang + '_mean_std'] = [roi_mean_lang, roi_std_lang, roi_sem_lang, len(model6_roi_lang.values), model6_roi_lang.values]

            # Split into DMN voxels
            roi_mean_dmn = model6_roi_dmn.mean().values
            roi_std_dmn = model6_roi_dmn.std().values
            roi_sem_dmn = roi_std_dmn/np.sqrt(len(model6_roi_dmn.values))
            d[roiID_dmn + '_mean_std'] = [roi_mean_dmn, roi_std_dmn, roi_sem_dmn, len(model6_roi_dmn.values), model6_roi_dmn.values]        
        
        if len(unique_atlas) == 1:
            roi_mean = model6_roi.mean().values
            roi_std = model6_roi.std().values
            roi_sem = roi_std/np.sqrt(len(model6_roi.values))
            d[roiID + '_mean_std'] = [roi_mean, roi_std, roi_sem, len(model6_roi.values), model6_roi.values]
    

    # Extract ROI names and corresponding values (ROI mean, ROI std, ROI size, ROI values)      
    ROInames = []
    ROIvalues = []
    for item in d.items():
        kval = item[0]
        sval = item[1]
        ROInames.append(kval)
        ROIvalues.append(sval)
    
    if plotROIs:
        # Intended order: 
        names_ordered = ['LH_PostTemp', 'LH_AntTemp_lang', 'LH_IFG', 'LH_IFGorb', 'LH_MFG', \
                'LH_AngG', 'RH_PostTemp', 'RH_AntTemp_lang', 'RH_IFG', 'RH_IFGorb', \
                 'RH_MFG', 'RH_AngG', 'LH_postParietal', 'LH_midParietal', 'LH_antParietal', \
                 'LH_supFrontal', 'LH_Precentral_A_PrecG', 'LH_Precental_B_IFGop', 'LH_midFrontal', \
                 'LH_midFrontalOrb', 'LH_insula', 'LH_medialFrontal', 'RH_postParietal', \
                 'RH_midParietal', 'RH_antParietal', 'RH_supFrontal', 'RH_Precentral_A_PrecG', \
                 'RH_Precental_B_IFGop', 'RH_midFrontal', 'RH_midFrontalOrb', 'RH_insula', 'RH_medialFrontal', \
                 'LH_FrontalMed', 'LH_PostCing', 'LH_TPJ', 'LH_MidCing', 'LH_STGorInsula', \
                 'LH_AntTemp_DMN', 'RH_FrontalMed', 'RH_PostCing', 'RH_TPJ', 'RH_MidCing', 'RH_STGorInsula', 'RH_AntTemp_DMN', \
                 'LH_TE11', 'LH_TE12', 'RH_TE11', 'RH_TE12', 'LH_OccipInf', 'LH_OccipMid', 'LH_OccipSup', \
                 'RH_OccipInf', 'RH_OccipMid', 'RH_OccipSup'] 
    
        ROInames_plot = [name[ : -9] for name in ROInames]
        
        # Sort the ROI names to the list above (intended ROI order)
        name_score = list(zip(ROInames_plot, ROIvalues))
        name_score_sorted = name_score.copy()
        name_score_sorted.sort(key=lambda x: names_ordered.index(x[0]))
        
        names = [item[0] for item in name_score_sorted] 
        scores = [item[1][0] for item in name_score_sorted] 
        stds = [item[1][1] for item in name_score_sorted] 
        sems = [item[1][2] for item in name_score_sorted] 
        sizes = [item[1][3] for item in name_score_sorted] 
        brain_scores = [item[1][4] for item in name_score_sorted] 
        
        # Save stats
        row = 1
        col = 0
        
        worksheet = workbook.add_worksheet("ROI") 
        worksheet.write(0, col, 'ROI_name')
        worksheet.write(0, 1, 'Mean')
        worksheet.write(0, 2, 'Std')
        worksheet.write(0, 3, 'Sem')
        worksheet.write(0, 4, 'Size')
        
        for idx, name in enumerate(names): 
            worksheet.write(row, col, name) 
            worksheet.write(row, col + 1, scores[idx]) 
            worksheet.write(row, col + 2, stds[idx]) 
            worksheet.write(row, col + 3, sems[idx]) 
            worksheet.write(row, col + 4, sizes[idx]) 

            row += 1
        
        
        # Save ROI sizes to csv files
#        name_size = list(zip(names, sizes))
#        
#        # With ROI annotations
#        f = open(csvfile_name,'w')
#        for ROI_element in name_size:
#            f.write(str(ROI_element) + '\n')
#        f.close()
        
        # Without ROI annotations
#        f2 = open('noAnnot_' + csvfile_name,'w')
#        for ROI_size in sizes:
#            f2.write(str(ROI_size) + '\n')
#        f2.close()
        
        # Remove suffix from names for plotting
        names[1] = 'LH_AntTemp'
        names[7] = 'RH_AntTemp'
        names[37] = 'LH_AntTemp'
        names[43] = 'RH_AntTemp'
        
        names_new = []
        for name in names:
            name_new = name.replace('_', ' ')
            substr = name_new[2:4]
            substr_up = substr.upper()
            name_new_up = name_new.replace(substr, substr_up)
            names_new.append(name_new_up)

        color_lst = [(241/255, 150/255, 135/255)]*12 + [(240/255, 211/255, 128/255)]*20 + \
        [(136/255, 192/255, 133/255)]*12 + ['lightsteelblue']*4 + ['thistle']*6
        
        xticks = list(range(len(names)))
        xtick_lst = [x + .33 for x in xticks] 
        
        plt.figure(figsize=(16, 8))
        plt.bar(range(len(names)), scores, yerr = stds, error_kw = dict(lw=2, capsize=0, capthick=1), \
                ecolor='black', align='edge', width=0.7, color = color_lst)
        plt.xticks(xtick_lst, names_new, size='small', rotation=90)
        plt.title('GloVe - Brain-Scores Across ROIs, Subject ' + subjectID)
        plt.ylabel('Brain-Score (r)', size='small') 
        plt.ylim(0, 0.2) # Same limits across subjects 0, 0.55
        plt.tight_layout()
        plt.savefig(pngfile_name, dpi=240)
    
    # For plotting networks and computing permutation stats
    if plotNetworks:
        
        network_names = ['Language', 'Multiple Demand', 'Default Mode', 'Auditory', 'Visual']
        
        # Define all neuroIDs within each network
        lang_values = [i for subl in brain_scores[0:12] for i in subl]
        md_values = [i for subl in brain_scores[12:32] for i in subl]
        dmn_values = [i for subl in brain_scores[32:44] for i in subl]
        aud_values = [i for subl in brain_scores[44:48] for i in subl]
        vis_values = [i for subl in brain_scores[48:] for i in subl]
        
        lang_score = np.mean(lang_values)
        lang_std = np.std(lang_values)
        lang_sem = lang_std/np.sqrt(len(lang_values))

        md_score = np.mean(md_values)
        md_std = np.std(md_values)
        md_sem = md_std/np.sqrt(len(md_values))
        
        dmn_score = np.mean(dmn_values)
        dmn_std = np.std(dmn_values)
        dmn_sem = dmn_std/np.sqrt(len(dmn_values))
        
        aud_score = np.mean(aud_values)
        aud_std = np.std(aud_values)
        aud_sem = aud_std/np.sqrt(len(aud_values))
        
        vis_score = np.mean(vis_values)
        vis_std = np.std(vis_values)
        vis_sem = vis_std/np.sqrt(len(vis_values))
        
        netw_scores = [lang_score, md_score, dmn_score, aud_score, vis_score]
        netw_stds = [lang_std, md_std, dmn_std, aud_std, vis_std]
        netw_sems = [lang_sem, md_sem, dmn_sem, aud_sem, vis_sem]
        netw_sizes = [len(lang_values), len(md_values), len(dmn_values), len(aud_values), len(vis_values)]


        # Save stats
        row = 1
        col = 0
        
        worksheet = workbook.add_worksheet("Network") 
        worksheet.write(0, col, 'Network_name')
        worksheet.write(0, 1, 'Mean')
        worksheet.write(0, 2, 'Std')
        worksheet.write(0, 3, 'Sem')
        worksheet.write(0, 4, 'Size')
        
        for idx, name in enumerate(network_names): 
            worksheet.write(row, col, name) 
            worksheet.write(row, col + 1, netw_scores[idx]) 
            worksheet.write(row, col + 2, netw_stds[idx]) 
            worksheet.write(row, col + 3, netw_sems[idx]) 
            worksheet.write(row, col + 4, netw_sizes[idx]) 

            row += 1
        
        # Plot
        plt.figure(figsize=(7, 14)) 
        plt.bar([0, .2, .4, .6, .8], netw_scores, \
        yerr = netw_stds, \
        error_kw = dict(lw=2, capsize=0, capthick=1), ecolor='black', align='edge', \
        width = .1, color = [(241/255, 150/255, 135/255), (240/255, 211/255, 128/255), \
                             (136/255, 192/255, 133/255), 'lightsteelblue', 'thistle'])
        plt.xticks([0.05, .25, .45, .65, .85], \
           network_names, \
           rotation=90, size='medium')
        plt.title('GloVe - Brain-Scores Across Networks, Subject ' + subjectID)
        plt.ylabel('Brain-Score (r)', size='medium') # 'Layer Number'
        plt.ylim(0, 0.15) #0, 0.47
        plt.tight_layout()
        plt.savefig(pngfile_name2, dpi=240)
        
        # Permutation testing
        row = 1
        col = 0
        
        worksheet = workbook.add_worksheet("Permutation") 
        worksheet.write(0, col, 'Comparison')
        worksheet.write(0, 1, 'Delta')
        worksheet.write(0, 2, 'Estimate_mean')
        worksheet.write(0, 3, 'p')
        
        worksheet.write(1, col, 'Lang vs MD')
        worksheet.write(2, col, 'Lang vs DMN')
        worksheet.write(3, col, 'Lang vs aud')
        worksheet.write(4, col, 'Lang vs vis')

        tests = [md_values, dmn_values, aud_values, vis_values]
        
        for idx, test in enumerate(tests):
            delta, est, p = is_significant(lang_values, test)
            worksheet.write(row, col + 1, delta) 
            worksheet.write(row, col + 2, est) 
            worksheet.write(row, col + 3, p) 

            row += 1
        
#        
#        f3 = open(statsfile_name, 'w')
#        f3.write('Lang vs MD: ' + str(is_significant(lang_values, md_values)) + '\n')
#        f3.write('Lang vs DMN: ' + str(is_significant(lang_values, dmn_values)) + '\n')
#        f3.write('Lang vs aud: ' + str(is_significant(lang_values, aud_values)) + '\n')
#        f3.write('Lang vs vis: ' + str(is_significant(lang_values, vis_values)) + '\n')
#        
#        f3.close()
#        
    


