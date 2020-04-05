# Imports
import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import datetime 

def extractBrainScoreAndCeiling(score_name = 'benchmark=Pereira2018-encoding,model=gpt2-xl,subsample=None.pkl',
                                ceil_name = 'neural_nlp.benchmarks.neural.PereiraEncoding.ceiling.pkl'):
    
    # Extracts the brain-score for each ROI, across all subjects. Can either be ceiled or not. Saves pkl file.

    # Define variables and file names
    timestamp = '{:%Y-%m-%d}'.format(datetime.datetime.now())
    fname = score_name.split('=')
    pklfile = 'ROI_analysis_' + timestamp + '_CEILED_' + fname[1][:-5] + fname[2][:-10] + '.pkl'
    pklfile = pklfile.replace(',', '_')
    pklfile = pklfile.replace('-', '_')
    
    if not ceil_name:
        pklfile = 'ROI_analysis_' + timestamp + '_NOT_CEILED_' + fname[1][:-5] + fname[2][:-10] + '.pkl'
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
     
#    _, index_model = np.unique(model2['neuroid_id'], return_index = True)
#    model3 = model2.isel(neuroid = np.sort(index_model))
    
    # Squeeze the layer dim
    model4 = model2.squeeze()
    
    # Sort by unique neuroid_id
    model5 = model4.sortby('neuroid_id')
    
    print('Mean of model: ', score_name, 'is: ', model5.mean(), '\n Size of model is: ', np.shape(model5))
    
    # Remove NaNs
    if np.isnan(model5).any() == True:
        nan_sum = np.isnan(model5).sum()
        print('Number of NaNs: ' + str(nan_sum))
        
        model6 = model5.fillna(0)
    else:
        model6 = model5
    
    ##  For ceiling, todo: not updated, needs work to align and run ## 
    if ceil_name:
        ceil2 = score_ceil.raw.mean(['split', 'experiment']) # Not grouped by neuroid
        
#        _, index_ceil = np.unique(ceil2['neuroid_id'], return_index = True)
#        ceil3 = ceil2.isel(neuroid = np.sort(index_ceil))
        
        # Sort by unique neuroid_id
        ceil4 = ceil2.sortby('neuroid_id')
        
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
        final_model = model6
        
    # Save the array values and mean/std values
    d = {}
    
    for roiID in roi_names:
        model6_roi = final_model[{'neuroid': [roi == roiID for roi in final_model['roi'].values]}]
        
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
            d[roiID_lang + '_mean_std'] = [roi_mean_lang, roi_std_lang, len(model6_roi_lang.values), model6_roi_lang.values]

            # Split into DMN voxels
            roi_mean_dmn = model6_roi_dmn.mean().values
            roi_std_dmn = model6_roi_dmn.std().values
            d[roiID_dmn + '_mean_std'] = [roi_mean_dmn, roi_std_dmn, len(model6_roi_dmn.values), model6_roi_dmn.values]
        
        if len(unique_atlas) == 1:
            roi_mean = model6_roi.mean().values
            roi_std = model6_roi.std().values
            d[roiID + '_mean_std'] = [roi_mean, roi_std, len(model6_roi.values), model6_roi.values]
    
    # Save
    pkl_arr = [d]
    
    # PICKLE TIME
    with open(pklfile, 'wb') as fout:
        pickle.dump(pkl_arr, fout)
    
