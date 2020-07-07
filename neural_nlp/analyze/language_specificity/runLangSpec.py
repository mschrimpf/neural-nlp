import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import datetime 
import seaborn
import xlsxwriter 
import os

ceil_name = '/om/user/msch/share/neural_nlp/identifier=Pereira2018-encoding.pkl'

# Load ceil objects
with open(ceil_name, 'rb') as f:  
    resultc = pickle.load(f)

ceil = resultc['data']

from func import *

models_pkl = []
for path,dir,file in os.walk('/braintree/home/msch/.result_caching/neural_nlp.score/'):
    for fileNames in file:
        if fileNames.startswith("benchmark=Pereira2018-encoding"):
            if 'untrained' not in fileNames:
                fileName = str(os.path.join(path,fileNames))
                print(fileName)
                models_pkl.append(fileName)

comparison='auditory'
timestamp = '{:%Y-%m-%d}'.format(datetime.datetime.now())
workbook = xlsxwriter.Workbook('Pereira2018-lang-'+comparison+'-specificity-'+timestamp+'.xlsx')

col=0

worksheet = workbook.add_worksheet("Language vs "+comparison) 
worksheet.write(0, col, 'Model_name')
worksheet.write(0, 1, 'Best_layer')
worksheet.write(0, 2, 'Comparison_1')
worksheet.write(0, 3, 'Comparison_2')
worksheet.write(0, 4, 'Normalized_Comparison1_Score')
worksheet.write(0, 5, 'Normalized_Comparison2_Score')
worksheet.write(0, 6, 'Median_Difference')
worksheet.write(0, 7, 'Median_Mult_Drop')    
worksheet.write(0, 8, 'Median_Perc_Drop')

for idx, model in enumerate(models_pkl):

    fname = models_pkl[idx]
    fname1= fname.split('=')
    fname2=fname1[-2].split(',')
    
    print(fname)
    
    with open(fname, 'rb') as f:  
        result = pickle.load(f)

    score = result['data']
    
    diff, mult, perc, ceiled_lang, ceiled_comp, best_layer = computeDrop(score,ceil,comparison)
 
    worksheet.write(idx+1, col, fname2[0]) 
    worksheet.write(idx+1, col + 1, str(best_layer)) 
    worksheet.write(idx+1, col + 2, 'Language') 
    worksheet.write(idx+1, col + 3, comparison) 
    worksheet.write(idx+1, col + 4, ceiled_lang) 
    worksheet.write(idx+1, col + 5, ceiled_comp) 
    worksheet.write(idx+1, col + 6, diff)
    worksheet.write(idx+1, col + 7, mult) 
    worksheet.write(idx+1, col + 8, perc) 

workbook.close()