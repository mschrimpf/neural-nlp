# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:50:44 2020

@author: greta
"""
from __future__ import division #Ensure division returns float
import pandas as pd
from scipy import stats
import os
from math import sqrt
import sys


def cohen_d(x,y):
    return (np.mean(x) - np.mean(y)) / sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2.0)
	
# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = np.mean(d1), np.mean(d2)
	# calculate the effect size
	return (u1 - u2) / s

## PEREIRA2018 ##

#%% MD - ceiled
data = pd.read_excel (r'P:\\Research2020\\Brain-Score_Lang\\Lang_Specificity\\Pereira2018-lang-MD-specificity-2020-05-19.xlsx') 
df = pd.DataFrame(data, columns= ['Normalized_Comparison1_Score','Normalized_Comparison2_Score', 'Best_layer'])
c1 = df['Normalized_Comparison1_Score'].values.tolist()[:-5]
c2 = df['Normalized_Comparison2_Score'].values.tolist()[:-5]

pmdc = df['Best_layer'].values.tolist()[:-5]

stats.ttest_ind(c1,c2)

assert(len(c1)==len(c2))
assert(len(c1)==43)

# Compute cohens d
print(cohen_d(c1,c2))
print(cohend(c1,c2))


#%% MD - non-ceiled
data = pd.read_excel (r'P:\\Research2020\\Brain-Score_Lang\\Lang_Specificity\\Pereira2018-lang-MD-specificity-not-ceiled-2020-05-21.xlsx') 
df = pd.DataFrame(data, columns= ['Comparison1_Predictivity','Comparison2_Predictivity', 'Best_layer'])
c1 = df['Comparison1_Predictivity'].values.tolist()[:-5]
c2 = df['Comparison2_Predictivity'].values.tolist()[:-5]

pmdnc = df['Best_layer'].values.tolist()[:-5]

assert(len(c1)==len(c2))
assert(len(c1)==43)

print(stats.ttest_ind(c1,c2))

set(pmdc)==set(pmdnc)

# Compute cohens d
print(cohen_d(c1,c2))

#%% DMN - ceiled
data = pd.read_excel (r'P:\\Research2020\\Brain-Score_Lang\\Lang_Specificity\\Pereira2018-lang-DMN-specificity-2020-05-19.xlsx') 
df = pd.DataFrame(data, columns= ['Normalized_Comparison1_Score','Normalized_Comparison2_Score','Best_layer'])
c1 = df['Normalized_Comparison1_Score'].values.tolist()[:-3]
c2 = df['Normalized_Comparison2_Score'].values.tolist()[:-3]

pdmnc = df['Best_layer'].values.tolist()[:-3]

assert(len(c1)==len(c2))
assert(len(c1)==43)

stats.ttest_ind(c1,c2)

set(pmdc)==set(pdmnc)

# Compute cohens d
print(cohen_d(c1,c2))

#%% DMN - non-ceiled

data = pd.read_excel (r'P:\\Research2020\\Brain-Score_Lang\\Lang_Specificity\\Pereira2018-lang-DMN-specificity-not-ceiled-2020-05-21.xlsx') 
df = pd.DataFrame(data, columns= ['Comparison1_Predictivity','Comparison2_Predictivity', 'Best_layer'])
c1 = df['Comparison1_Predictivity'].values.tolist()[:-3]
c2 = df['Comparison2_Predictivity'].values.tolist()[:-3]

pdmnnc = df['Best_layer'].values.tolist()[:-3]

assert(len(c1)==len(c2))
assert(len(c1)==43)

stats.ttest_ind(c1,c2)

set(pdmnc)==set(pdmnnc)
set(pdmnc)==set(pmdnc)

# Compute cohens d
print(cohen_d(c1,c2))

#%% FEDORENKO2016 ##

#%% Ceiled
data = pd.read_excel (r'P:\\Research2020\\Brain-Score_Lang\\Lang_Specificity\\Fedorenko2016v3\\Fedorenko2016-Language-Non-Language-specificity-ceiled-2020-06-03.xlsx') 
df = pd.DataFrame(data, columns= ['Normalized_Comparison1_Score','Normalized_Comparison2_Score', 'Best_layer'])
c1 = df['Normalized_Comparison1_Score'].values.tolist()[:-3]
c2 = df['Normalized_Comparison2_Score'].values.tolist()[:-3]

fc = df['Best_layer'].values.tolist()[:-3]
assert(set(pdmnc)==set(fc)) #false. legit.

stats.ttest_ind(c1,c2)

assert(len(c1)==len(c2))
assert(len(c1)==43)

# Compute cohens d
cohen_d(c1,c2)

#%% Non-Ceiled
#data = pd.read_excel (r'P:\\Research2020\\Brain-Score_Lang\\Lang_Specificity\\Fedorenko2016v3\\Fedorenko2016-Language-Non-Language-specificity-non-ceiled-2020-06-03.xlsx') 
#df = pd.DataFrame(data, columns= ['Predictivity_Comparison1','Predictivity_Comparison2'])
#c1 = df['Predictivity_Comparison1'].values.tolist()[:-3]
#c2 = df['Predictivity_Comparison2'].values.tolist()[:-3]
#
#stats.ttest_ind(c1,c2)

# Non-Ceiled - 06212020 (new median computation)
data = pd.read_excel (r'P:\\Research2020\\Brain-Score_Lang\\Lang_Specificity\\Fedorenko2016v3\\Fedorenko2016-Language-Non-Language-specificity-non-ceiled-2020-06-21.xlsx') 
df = pd.DataFrame(data, columns= ['Predictivity_Comparison1','Predictivity_Comparison2','Best_layer'])
c1 = df['Predictivity_Comparison1'].values.tolist()[:-3]
c2 = df['Predictivity_Comparison2'].values.tolist()[:-3]

fnc = df['Best_layer'].values.tolist()[:-3]

set(fc)==set(fnc)

stats.ttest_ind(c1,c2)

assert(len(c1)==len(c2))
assert(len(c1)==43)

# Compute cohens d
print(cohen_d(c1,c2))