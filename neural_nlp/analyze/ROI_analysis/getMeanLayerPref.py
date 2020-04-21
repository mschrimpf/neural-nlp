import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import datetime 

# Function inputs
score_name = 'benchmark=Pereira2018-encoding,model=gpt2-xl,subsample=None.pkl'
metric = 'mean'

timestamp = '{:%Y-%m-%d}'.format(datetime.datetime.now())
fname = score_name.split('=')
pklfile = 'AllSubs_ROI_analysis_LayerPref_' + timestamp + '_' + metric + '_' + fname[1][:-5] + fname[2][:-10] + '.pkl'
pklfile = pklfile.replace(',', '_')
pklfile = pklfile.replace('-', '_')

# Load score objects
with open(score_name, 'rb') as f:  
    result = pickle.load(f)

score = result['data']

roi_names_all = score.raw.roi.values
roi_names = np.unique(roi_names_all)

## FOR MODEL ##
center_score = score[{'aggregation': [agg == 'center' for agg in score['aggregation'].values]}].squeeze('aggregation')
model2 = center_score.raw.mean(['split', 'experiment']) # Not grouped by neuroid

#_, index_model = np.unique(model2['neuroid_id'], return_index = True)
#model3 = model2.isel(neuroid = np.sort(index_model))

# Squeeze the layer dim
model4 = model2.squeeze()

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

# Group by ROIs and average across ROIs 
d = {}
for roiID in roi_names:
    model7_roi = model7_norm[{'neuroid': [roi == roiID for roi in model7_norm['roi'].values]}]
    
    unique_atlas = np.unique(model7_roi.atlas.values)
    
    if len(unique_atlas) > 1:

        # Disentangle LH/RH AntTemp for lang and DMN
        model7_roi_lang = model7_roi[{'neuroid': [atlas == 'language' for atlas in model7_roi['atlas'].values]}]
        model7_roi_dmn = model7_roi[{'neuroid': [atlas == 'DMN' for atlas in model7_roi['atlas'].values]}]
        
        roiID_lang = roiID + '_lang'
        roiID_dmn = roiID + '_DMN'
        
        # Split into language voxels
        if metric == 'mean':
            roi_metric_lang = model7_roi_lang.mean().values
        if metric == 'median':
            roi_metric_lang = model7_roi_lang.median().values
        if metric == 'mode':
            roi_metric_lang = stats.mode(model7_roi_lang.values)
            roi_metric_lang = roi_metric_lang[0][0]
        roi_std_lang = model7_roi_lang.std().values
        d[roiID_lang + '_metric'] = [roi_metric_lang, roi_std_lang, len(model7_roi_lang.values), model7_roi_lang.values]

        # Split into DMN voxels
        if metric == 'mean':
            roi_metric_dmn = model7_roi_dmn.mean().values
        if metric == 'median':
            roi_metric_dmn = model7_roi_dmn.median().values
        if metric == 'mode':
            roi_metric_dmn = stats.mode(model7_roi_dmn.values)
            roi_metric_dmn = roi_metric_dmn[0][0]
        roi_std_dmn = model7_roi_dmn.std().values
        d[roiID_dmn + '_metric'] = [roi_metric_dmn, roi_std_dmn, len(model7_roi_dmn.values), model7_roi_dmn.values]
    
    if len(unique_atlas) == 1:
        if metric == 'mean':
            roi_metric = model7_roi.mean().values
        if metric == 'median':
            roi_metric = model7_roi.median().values
        if metric == 'mode':
            roi_metric = stats.mode(model7_roi.values)
            roi_metric = roi_metric[0][0]
        roi_std = model7_roi.std().values
        d[roiID + '_metric'] = [roi_metric, roi_std, len(model7_roi.values), model7_roi.values]

# Save
pkl_arr = [d]

# PICKLE TIME
with open(pklfile, 'wb') as fout:
    pickle.dump(pkl_arr, fout)
    