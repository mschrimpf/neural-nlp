# Imports
import pickle
import os
import numpy as np
from collections import Counter
import scipy.io as sio

def createBrainMatrix(subjectID = '018', score_name = 'benchmark=Pereira2018-encoding,model=glove,subsample=None.pkl'):

    # Define variables and file names
    SPM_dim = (79,95,69)
    fname = score_name.split('=')
    matfile_name = subjectID + '_' + fname[1][:-5] + fname[2][:-10] + '.mat'
    matfile_name = matfile_name.replace(',', '_')
    matfile_name = matfile_name.replace('-', '_')
    
    # Load score object
    with open(score_name, 'rb') as f:  
        result = pickle.load(f)
    
    score = result['data']
    subjects = np.unique(score.raw.subject.values)
    
    # Find the best model layer
    center_score = score[{'aggregation': [agg == 'center' for agg in score['aggregation'].values]}].squeeze('aggregation')
    
    # Mean score over atlas and experiment
    layer_score = center_score.mean(['atlas', 'experiment'])
    best_layer = layer_score['layer'][layer_score == layer_score.max()].values[0]
    
    # Extract brain-scores for subjects
    s = score.raw[{'neuroid': [subject == subjectID for subject in score.raw['subject'].values]}] 
    
    # Fetch score for the best layer
    s2 = s[{'layer': [layer == best_layer for layer in s['layer'].values]}]
    
    # s2.mean()
    
    # Mean across experiment and split 
    s3 = s2.mean(['split', 'experiment']) # Not grouped by neuroid
    
    # Group by (unique) neuroid
    _, index = np.unique(s3['neuroid_id'], return_index = True)
    
    # Maintain the order of neuroids
    s4 = s3.isel(neuroid = np.sort(index))
    
    s5 = s4.squeeze('layer')
    
    # Create empty brain matrix
    brain = np.empty(SPM_dim) # The original data dimensions from that particular subject
    brain[:] = np.nan
    
    for idx, element in enumerate(s5.values):
    #     print(s5.col_to_coord_1.values[idx])
    #     print(s5.indices_in_3d.values[idx]) # Corresponds to MATLAB
        brain[(s5.col_to_coord_1.values[idx])-1, (s5.col_to_coord_2.values[idx])-1, (s5.col_to_coord_3.values[idx])-1] = element
    
    # Save brain matrix
    sio.savemat(matfile_name, {'brain_matrix':brain}, {'mean_brainscore':s5.mean().values, 'max_brainscore':s5.max().values, 'min_brainscore':s5.min().values, 'best_layer':s5.layer.values})
