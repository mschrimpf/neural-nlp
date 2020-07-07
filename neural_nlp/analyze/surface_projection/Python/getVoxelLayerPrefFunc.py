# Imports
import pickle
import os
import numpy as np
from collections import Counter
import scipy.io as sio

def getVoxelLayerPref(subjectID = '018', score_name = 'benchmark=Pereira2018-encoding,model=gpt2-xl,subsample=None.pkl'):

    # Define variables and file names
    SPM_dim = (79,95,69)
    fname = score_name.split('=')
    matfile_name = subjectID + '_layer_pref_' + fname[1][:-5] + fname[2][:-10] + '.mat'
    matfile_name = matfile_name.replace(',', '_')
    matfile_name = matfile_name.replace('-', '_')
    
    # Load score object
    with open(score_name, 'rb') as f:  
        result = pickle.load(f)
    
    score = result['data']
    
    # center_score = score[{'aggregation': [agg == 'center' for agg in score['aggregation'].values]}].squeeze('aggregation')
    
    # Extract brain-scores for subjects
    s = score.raw[{'neuroid': [subject == subjectID for subject in score.raw['subject'].values]}] 
    
    # Mean across experiment and split 
    t3 = s.mean(['split', 'experiment']) # Not grouped by neuroid
    
    # Group by (unique) neuroid
    _, index = np.unique(t3['neuroid_id'], return_index = True)
    t4 = t3.isel(neuroid = np.sort(index))
    
    # Test NaNs
    if np.isnan(t4).any():
        nan_sum = np.isnan(t4).sum()
        nan_perc = nan_sum / (t4.shape[0] * t4.shape[1]) * 100
    
        t4 = t4.fillna(0)
    
        log_file = open('layer_preference_nan_log.txt', 'a')
        log_file.write('Percentage NaN values for subject {0}: {1}'.format(subjectID, nan_perc.values.round(3)))

    t5 = t4.argmax('layer',skipna=True) # Argmax over layer dimension
    
    # Normalize
    t6 = t5/t5.max()

    # Create empty brain matrix
    brain = np.empty(SPM_dim) # The original data dimensions from that particular subject
    brain[:] = np.nan
    
    for idx, element in enumerate(t6.values):
        brain[(t6.col_to_coord_1.values[idx])-1, (t6.col_to_coord_2.values[idx])-1, (t6.col_to_coord_3.values[idx])-1] = element
    
    # Save brain matrix
    sio.savemat(matfile_name, {'brain_matrix':brain})
