# Imports
import pickle
import os
import numpy as np
from collections import Counter
import scipy.io as sio
import matplotlib.pyplot as plt

# Function inputs
#score_name = 'benchmark=Pereira2018-encoding,model=gpt2-xl,subsample=None.pkl'
score_name = 'benchmark=Pereira2018-encoding,model=glove,subsample=None.pkl'
ceil_name = 'neural_nlp.benchmarks.neural.PereiraEncoding.ceiling.pkl'

# Define variables and file names
fname = score_name.split('=')
pklfile = 'ROIanalysis_' + fname[1][:-5] + fname[2][:-10] + '.pkl'
pklfile = pklfile.replace(',', '_')
pklfile = pklfile.replace('-', '_')

# Load score objects
with open(score_name, 'rb') as f:  
    result = pickle.load(f)

with open(ceil_name, 'rb') as f:  
    c_result = pickle.load(f)

score = result['data']
score_ceil = c_result['data']

# Inv score object
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

ceiled_model = model5 / ceil4.values

# Take abs value
#ceiled_model_abs = abs(ceiled_model)

# Save the array values and a mean val 
d = {}

for roiID in roi_names:
    ceil_roi = ceiled_model[{'neuroid': [roi == roiID for roi in ceiled_model['roi'].values]}]
    ceil_mean = ceil_roi.mean()
    # print('Mean of ROI ', roiID, ceil_mean)
    d[roiID + '_arr'] = ceil_roi.values
    d[roiID + '_mean'] = ceil_mean.mean().values


#plt.figure
#plt.plot(ceiled_model_abs.values)

# Save
pkl_arr = [d]

# PICKLE TIME
with open(pklfile, 'wb') as fout:
    pickle.dump(pkl_arr, fout)

