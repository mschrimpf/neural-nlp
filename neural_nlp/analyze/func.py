#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import datetime 
import seaborn
from stats import is_significant, _permutation_test
import xlsxwriter 
from scipy import stats
import xarray as xr
from matplotlib import cm

  
# Plot specifications
seaborn.set(context='talk')
seaborn.set_style("whitegrid", {'axes.grid': False})
plt.rc('axes', edgecolor='black')
plt.rc('axes', edgecolor='black')
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False



def getCenter(ceil):
    x = ceil.raw[{'aggregation': [agg == 'center' for agg in ceil.raw['aggregation'].values]}].squeeze('aggregation')
    return x

def getCenter2(obj):
    # same as above, just more generic
    x = obj[{'aggregation': [agg == 'center' for agg in obj['aggregation'].values]}].squeeze('aggregation')
    return x

def assertCeiling(obj,cobj):
    assert set(obj['neuroid_id'].values) == set(cobj['neuroid_id'].values)
    assert (obj['neuroid_id'].values.all()) == (cobj['neuroid_id'].values.all())
    assert (obj['voxel_num'].values.all()) == (cobj['voxel_num'].values.all())
    align = cobj[{'neuroid': [obj['neuroid_id'].values.tolist().index(neuroid_id)
                           for neuroid_id in obj['neuroid_id'].values]}]  # align
    
    assert align.all()==cobj.all()
    
def bestLangLayer(score):
    x = score[{'aggregation': [agg == 'center' for agg in score['aggregation'].values]}].squeeze('aggregation')
    best_lang_layer = x['layer'][x.argmax()].layer.values
    
    print(best_lang_layer)
    return best_lang_layer

def meanSplitExp(score):
    x = score.raw.raw.raw
    x1 = x.mean(['split', 'experiment'])
    return x1

def freezeBestLangLayer(obj, layerInfo):
    return obj[{'layer': [layer == layerInfo for layer in obj['layer'].values]}].squeeze('layer')

def extractSubject(obj, subjID):
    return obj[{'neuroid': [subject == subjID for subject in obj['subject'].values]}] 

def sortBy(obj, coord='neuroid_id'):
    return obj.sortby(coord)

def removeNaNs(obj):
    subj = obj.subject.values
    if np.isnan(obj).any() == True:
        nan_sum = np.isnan(obj).sum()
        print('Number of NaNs: ' + str(nan_sum) + 'subjID: ' + subj)
#         x = obj.fillna(0)
        x = obj.dropna('neuroid')

    else:
        x = obj
        
    return x

def getUniqueNeuroIDs(obj):
    _, index_u = np.unique(obj['neuroid_id'], return_index = True)
    x = obj.isel(neuroid = np.sort(index_u))
    return x

def getLayerPref(obj):
    x = obj.argmax('layer', skipna=True) # Argmax over layer dimension
    # Normalize scores 
    x_norm = x/x.max()
    return x_norm

def dropDMN(obj, cobj):
    
    for i, roiID in enumerate(['LH_AntTemp', 'RH_AntTemp']):
        if i == 1:
            obj = x
            cobj = y
            
        roi = obj[{'neuroid': [roi == roiID for roi in obj['roi'].values]}]
        cobj_roi = cobj[{'neuroid': [roi == roiID for roi in cobj['roi'].values]}]

        unique_atlas = np.unique(roi.atlas.values)
        print('More than one unique atlas: ', unique_atlas)

        # Disentangle LH/RH AntTemp for lang and DMN
        roi_dmn = roi[{'neuroid': [atlas == 'DMN' for atlas in roi['atlas'].values]}]
        cobj_roi_dmn = cobj_roi[{'neuroid': [atlas == 'DMN' for atlas in cobj_roi['atlas'].values]}]

        cobj_intersec = np.in1d(cobj.neuroid_id.values, cobj_roi_dmn.neuroid_id.values)
        cobj_n = np.array([not y for y in cobj_intersec])
        cobj_xi = xr.DataArray(cobj_n.reshape(cobj.shape),dims=cobj.dims,coords=cobj.coords)
        y = cobj.where(cobj_xi,drop=True)
        
        intersec = np.in1d(obj.neuroid_id.values, roi_dmn.neuroid_id.values)
        n = np.array([not y for y in intersec])
        xi = xr.DataArray(n.reshape(obj.shape),dims=obj.dims,coords=obj.coords)
        x = obj.where(xi,drop=True)
        
    return x, y

def assertCeiling(obj,cobj):
    assert set(obj['neuroid_id'].values) == set(cobj['neuroid_id'].values)
    assert (obj['neuroid_id'].values.all()) == (cobj['neuroid_id'].values.all())
    assert (obj['voxel_num'].values.all()) == (cobj['voxel_num'].values.all())
    align = cobj[{'neuroid': [obj['neuroid_id'].values.tolist().index(neuroid_id)
                           for neuroid_id in obj['neuroid_id'].values]}]  # align
    
    assert align.all()==cobj.all()
    
    
def getROIs(obj, cobj, filename='brain.mat', atlas='language'):
    roi_names_all = obj.roi.values
    roi_names = np.unique(roi_names_all)
    
    roi_atlas = obj.roi[{'neuroid': [l == atlas for l in obj['atlas'].values]}].values
    roi_atlas_u = np.unique(roi_atlas)

    SPM_dim = (79,95,69)

    # Create empty brain matrix
    brain = np.empty(SPM_dim) # The original data dimensions from that particular subject
    brain[:] = np.nan

    assert set(obj.roi.values)==set(cobj.roi.values)

    d = {}
    for roiID in roi_atlas_u:
        roi = obj[{'neuroid': [roi == roiID for roi in obj['roi'].values]}]
        cobj_roi = cobj[{'neuroid': [roi == roiID for roi in cobj['roi'].values]}]

        unique_atlas = np.unique(roi.atlas.values)

        assert len(unique_atlas) == 1

        # Ceil by median ROI val
        cobj_roi_med = cobj_roi.median().values
        c_vals = roi/cobj_roi_med

        # Not ceiled
        roi_mean = roi.mean().values
        roi_med = roi.median().values

        roi_std = roi.std().values
        roi_sem = roi_std/np.sqrt(len(roi.values))

        roi_mad = stats.median_absolute_deviation(roi)
        roi_mad_m = roi_mad/np.sqrt(len(roi.values))

        # Ceiled
        croi_med = c_vals.median().values
        croi_mad = stats.median_absolute_deviation(c_vals)
        croi_mad_m = croi_mad/np.sqrt(len(c_vals.values))


        # obj = obj[{'neuroid': [roi == roiID for roi in obj['roi'].values]}]/cobj_roi_med

        for idx, element in enumerate(c_vals.values):
            brain[(c_vals.col_to_coord_1.values[idx])-1, (c_vals.col_to_coord_2.values[idx])-1, \
                  (c_vals.col_to_coord_3.values[idx])-1] = element


        d[roiID + '_save'] = [roi_mean, roi_med, roi_std, roi_sem, roi_mad, roi_mad_m, \
                              croi_med, croi_mad, croi_mad_m, cobj_roi_med, 
                              len(c_vals.values), c_vals]

    # Save brain matrix
    sio.savemat(filename, {'brain_matrix':brain})
    
    return d

def writeStats(d, names_ordered, workbook_name):
    ROInames = []
    ROIvalues = []
    for item in d.items():
        kval = item[0]
        sval = item[1]
        ROInames.append(kval)
        ROIvalues.append(sval)

    ROInames_plot = [name[ : -5] for name in ROInames]

    # Sort the ROI names to the list (intended ROI order)
    name_score = list(zip(ROInames_plot, ROIvalues))
    name_score_sorted = name_score.copy()
    name_score_sorted.sort(key=lambda x: names_ordered.index(x[0]))

    names = [item[0] for item in name_score_sorted] 
    means = [item[1][0] for item in name_score_sorted] 
    meds = [item[1][1] for item in name_score_sorted] 

    stds = [item[1][2] for item in name_score_sorted] 
    sems = [item[1][3] for item in name_score_sorted] 
    
    mads = [item[1][4] for item in name_score_sorted] 
    mads_m = [item[1][5] for item in name_score_sorted] 
    
    # Ceiled
    medsc = [item[1][6] for item in name_score_sorted] 
    madsc = [item[1][7] for item in name_score_sorted] 
    mads_mc = [item[1][8] for item in name_score_sorted] 
    
    ceil_val = [item[1][9] for item in name_score_sorted]
    
    sizes = [item[1][10] for item in name_score_sorted] 
    brain_scores = [item[1][11] for item in name_score_sorted] 

    names_new = []
    for name in names:
        name_new = name.replace('_', ' ')
        substr = name_new[2:4]
        substr_up = substr.upper()
        name_new_up = name_new.replace(substr, substr_up)
        names_new.append(name_new_up)
    
    if len(names_ordered) > 12:
        netw = ['Language']*12 + ['Multiple Demand']*20 + ['Default Mode']*10 + ['Auditory']*4 + ['Visual']*6
    else:
        netw = ['Language']*12
    
    # Save stats - ROI
    row = 1
    col = 0

    timestamp = '{:%Y-%m-%d}'.format(datetime.datetime.now())
    workbook = xlsxwriter.Workbook(workbook_name)
    
    worksheet = workbook.add_worksheet("ROI") 
    worksheet.write(0, col, 'ROI_name')
    worksheet.write(0, 1, 'Network')
    worksheet.write(0, 2, 'Mean')
    worksheet.write(0, 3, 'Median')
    worksheet.write(0, 4, 'Std')
    worksheet.write(0, 5, 'Sem')
    worksheet.write(0, 6, 'Mad')
    worksheet.write(0, 7, 'Mad_div_sqrt_obs')
    worksheet.write(0, 8, 'Consistency_median')
    worksheet.write(0, 9, 'Consistency_mad')
    worksheet.write(0, 10, 'Consistency_mad_div_sqrt_obs')
    worksheet.write(0, 11, 'Ceiling_value_median')
    worksheet.write(0, 12, 'Size_num_neuroIDs')

    for idx, name in enumerate(names_new): 
        worksheet.write(row, col, name) 
        worksheet.write(row, col + 1, netw[idx]) 
        worksheet.write(row, col + 2, means[idx]) 
        worksheet.write(row, col + 3, meds[idx]) 
        worksheet.write(row, col + 4, stds[idx]) 
        worksheet.write(row, col + 5, sems[idx])
        worksheet.write(row, col + 6, mads[idx]) 
        worksheet.write(row, col + 7, mads_m[idx]) 
        
        worksheet.write(row, col + 8, medsc[idx])
        worksheet.write(row, col + 9, madsc[idx]) 
        worksheet.write(row, col + 10, mads_mc[idx])
        worksheet.write(row, col + 11, ceil_val[idx])

        worksheet.write(row, col + 12, sizes[idx]) 

        row += 1
    
    workbook.close()
    
    return medsc, madsc, names_new 
   
def plotBars(bars, err, names, colors, model, filename, subjectID):    

    xticks = list(range(len(names)))
    xtick_lst = [x + .33 for x in xticks] 

    plt.figure(figsize=(8, 8))
    plt.bar(range(len(names)), bars, yerr = err, error_kw = dict(lw=2, capsize=0, capthick=1), \
            ecolor='black', edgecolor='black', linewidth=0.5, align='edge', width=0.7, color = colors)

    plt.xticks(xtick_lst, names, size='medium', rotation=90)
    plt.title(model +' - Normalized Predictivity, Language ROIs, Subject ' + subjectID)
    plt.ylabel('Normalized Predictivity (r/c)', size='large') 
    plt.ylim(0, 1.5) # Same limits across subjects 0, 0.55
    plt.yticks([0.0, 0.2, .4, .6, .8, 1], ['0.0','0.2','0.4','0.6','0.8','1.0'])
    plt.tight_layout()
    plt.savefig(filename, dpi=240)
    
def getLayerPref(obj):
    x = obj.argmax('layer', skipna=True) # Argmax over layer dimension
    # Normalize scores 
    x_norm = x/x.max()
    return x_norm


def plotLayerPref(obj, title, filename):
    cmhot = cm.get_cmap("hot")

    plt.figure(figsize=(14, 8))
    n, bins, patches = plt.hist(obj.values, bins=48, edgecolor='black', linewidth=1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cmhot(c))

    plt.xlabel('Normalized Layer Number')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=240)

def writeToBrain(obj, filename):
    
    
    SPM_dim = (79,95,69)

    # Create empty brain matrix
    brain = np.empty(SPM_dim) # The original data dimensions from that particular subject
    brain[:] = np.nan
    
    for idx, element in enumerate(obj.values):
            brain[(obj.col_to_coord_1.values[idx])-1, (obj.col_to_coord_2.values[idx])-1, \
                  (obj.col_to_coord_3.values[idx])-1] = element
            
    # Save brain matrix
    sio.savemat(filename, {'brain_matrix':brain})

def aggregateScores(obj, coord='subject'):
    subject_scores = obj.groupby(coord).median()
    center = subject_scores.median(coord)
    # subject_values = np.nan_to_num(subject_scores.values, nan=0)
    return center
    
def computeDrop(score,ceil,comparison):
    b=bestLangLayer(score)
    o=meanSplitExp(score)
    o1=freezeBestLangLayer(o,b)
    o_lang=extractCoord(o1,'atlas','language')
    o_comp=extractCoord(o1,'atlas',comparison)

    center_lang=aggregateScores(o_lang)
    center_comp=aggregateScores(o_comp)

    # Ceiling
    c=getCenter(ceil)
#    c_lang=extractCoord(c,'atlas','language').median()
#    c_comp=extractCoord(c,'atlas',comparison).median()

#    ceiled_lang = center_lang/c_lang
#    ceiled_comp = center_comp/c_comp

    c_lang=extractCoord(c,'atlas','language')#.median()
    c_comp=extractCoord(c,'atlas',comparison)#.median()
    
    ceil_lang=aggregateScores(c_lang)
    ceil_comp=aggregateScores(c_comp)

    ceiled_lang = center_lang/ceil_lang
    ceiled_comp = center_comp/ceil_comp

    diff = ceiled_lang - ceiled_comp
    mult = ceiled_lang / ceiled_comp
    perc = diff / ceiled_lang

    return diff, mult, perc, ceiled_lang, ceiled_comp, b 

def computeDropNoCeil(score,comparison):
    b=bestLangLayer(score)
    o=meanSplitExp(score)
    o1=freezeBestLangLayer(o,b)
    o_lang=extractCoord(o1,'atlas','language')
    o_comp=extractCoord(o1,'atlas',comparison)

    center_lang=aggregateScores(o_lang)
    center_comp=aggregateScores(o_comp)

    diff = center_lang - center_comp
    mult = center_lang / center_comp
    perc = diff / center_lang

    return diff, mult, perc, center_lang, center_comp, b 

def computeDropFedorenko(score_l,score_nl,ceil=True):
    b=bestLangLayer(score_l)
    if ceil:
        o_l=freezeBestLangLayer(score_l,b)
        o_comp=freezeBestLangLayer(score_nl,b) # freeze same layer
        center_lang=getCenter2(o_l).values
        center_comp=getCenter2(o_comp).values
    if not ceil:
        center_lang=freezeBestLangLayer(score_l.raw.raw,b).median()
        center_comp=freezeBestLangLayer(score_nl.raw.raw,b).median()

    diff = center_lang - center_comp
    mult = center_lang / center_comp
    perc = (diff / center_lang)*100

    return diff, mult, perc, center_lang, center_comp, b 

def extractCoord(obj, coord, ID):
    return obj[{'neuroid': [x == ID for x in obj[coord].values]}] 