# Imports
import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def extractBrainScoreAndCeiling(score_name = 'benchmark=Pereira2018-encoding,model=gpt2-xl,subsample=None.pkl',
                                ceil_name = 'neural_nlp.benchmarks.neural.PereiraEncoding.ceiling.pkl'):
    
    # Extracts the brain-score for each ROI, across all subject. Can either be ceiled or not. Saves pkl file.

    # Define variables and file names
    fname = score_name.split('=')
    pklfile = 'ROI_analysis_CEILED_' + fname[1][:-5] + fname[2][:-10] + '.pkl'
    pklfile = pklfile.replace(',', '_')
    pklfile = pklfile.replace('-', '_')
    
    if not ceil_name:
        pklfile = 'ROI_analysis_NOT_CEILED_' + fname[1][:-5] + fname[2][:-10] + '.pkl'
        pklfile = pklfile.replace(',', '_')
        pklfile = pklfile.replace('-', '_')
    
    # Load score objects
    with open(score_name, 'rb') as f:  
        result = pickle.load(f)
    
    if ceil_name:
        with open(ceil_name, 'rb') as f:  
            c_result = pickle.load(f)
            score_ceil = c_result['data']

    score = result['data']
    
    # Extract ROI names
    roi_names_all = score.raw.roi.values
    roi_names = np.unique(roi_names_all)
    
    # Find best layer
    center_score = score[{'aggregation': [agg == 'center' for agg in score['aggregation'].values]}].squeeze('aggregation')
    layer_score = center_score.mean(['atlas', 'experiment'])
    best_layer = layer_score['layer'][layer_score == layer_score.max()].values[0]
    
    ## FOR MODEL ##
    
    model = score.raw[{'layer': [layer == best_layer for layer in score.raw['layer'].values]}] 
    model2 = model.mean(['split', 'experiment']) # Not grouped by neuroid
     
    _, index_model = np.unique(model2['neuroid_id'], return_index = True)
    model3 = model2.isel(neuroid = np.sort(index_model))
    
    # Squeeze the layer dim
    model4 = model3.squeeze()
    
    # Sort by unique neuroid_id
    model5 = model4.sortby('neuroid_id')
    
    print('Mean of model: ', score_name, 'is: ', model5.mean(), '\n Size of model is: ', np.shape(model5))
      
    ##  For ceiling ## 
    if ceil_name:
        ceil2 = score_ceil.raw.mean(['split', 'experiment']) # Not grouped by neuroid
        
        _, index_ceil = np.unique(ceil2['neuroid_id'], return_index = True)
        ceil3 = ceil2.isel(neuroid = np.sort(index_ceil))
        
        # Sort by unique neuroid_id
        ceil4 = ceil3.sortby('neuroid_id')
        
        ## SANITY CHECK AND ASSERT ## 
        model_ids = model5.neuroid_id.values
        ceil_ids = ceil4.neuroid_id.values
        
        model_vox = model5.voxel_num.values
        ceil_vox = ceil4.voxel_num.values
        
        if not np.array_equal(model_vox, ceil_vox):
            raise ValueError('The voxel ids do not match')
            
        if not np.array_equal(model_ids, ceil_ids):
            raise ValueError('The neuro ids do not match')
        
        final_model = model5 / ceil4.values
        
    if not ceil_name:
        final_model = model5
        
    # Take abs value
    # final_model = abs(final_model)
    
    # Save the array values and mean/std values
    d = {}
    
    for roiID in roi_names:
        ceil_roi = final_model[{'neuroid': [roi == roiID for roi in final_model['roi'].values]}]
        roi_mean = ceil_roi.mean().values
        roi_std = ceil_roi.std().values
        d[roiID + '_mean_std'] = [roi_mean, roi_std]
    
    # Save
    pkl_arr = [d]
    
    # PICKLE TIME
    with open(pklfile, 'wb') as fout:
        pickle.dump(pkl_arr, fout)
    
