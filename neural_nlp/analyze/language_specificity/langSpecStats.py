# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:50:44 2020

@author: greta
"""

import pandas as pd
from scipy import stats
import os

## PEREIRA2018 ##

# MD
data = pd.read_excel (r'P:\\Research2020\\Brain-Score_Lang\\Lang_Specificity\\Pereira2018-lang-MD-specificity-2020-05-19.xlsx') 
df = pd.DataFrame(data, columns= ['Normalized_Comparison1_Score','Normalized_Comparison2_Score'])
c1 = df['Normalized_Comparison1_Score'].values.tolist()[:-5]
c2 = df['Normalized_Comparison2_Score'].values.tolist()[:-5]

stats.ttest_ind(c1,c2)

data = pd.read_excel (r'P:\\Research2020\\Brain-Score_Lang\\Lang_Specificity\\Pereira2018-lang-MD-specificity-not-ceiled-2020-05-21.xlsx') 
df = pd.DataFrame(data, columns= ['Comparison1_Predictivity','Comparison2_Predictivity'])
c1 = df['Comparison1_Predictivity'].values.tolist()[:-5]
c2 = df['Comparison2_Predictivity'].values.tolist()[:-5]

stats.ttest_ind(c1,c2)

# DMN
data = pd.read_excel (r'P:\\Research2020\\Brain-Score_Lang\\Lang_Specificity\\Pereira2018-lang-DMN-specificity-2020-05-19.xlsx') 
df = pd.DataFrame(data, columns= ['Normalized_Comparison1_Score','Normalized_Comparison2_Score'])
c1 = df['Normalized_Comparison1_Score'].values.tolist()[:-3]
c2 = df['Normalized_Comparison2_Score'].values.tolist()[:-3]

stats.ttest_ind(c1,c2)

data = pd.read_excel (r'P:\\Research2020\\Brain-Score_Lang\\Lang_Specificity\\Pereira2018-lang-DMN-specificity-not-ceiled-2020-05-21.xlsx') 
df = pd.DataFrame(data, columns= ['Comparison1_Predictivity','Comparison2_Predictivity'])
c1 = df['Comparison1_Predictivity'].values.tolist()[:-3]
c2 = df['Comparison2_Predictivity'].values.tolist()[:-3]

stats.ttest_ind(c1,c2)

## FEDORENKO2016 ##

# Ceiled
data = pd.read_excel (r'P:\\Research2020\\Brain-Score_Lang\\Lang_Specificity\\Fedorenko2016v3\\Fedorenko2016-Language-Non-Language-specificity-ceiled-2020-06-03.xlsx') 
df = pd.DataFrame(data, columns= ['Normalized_Comparison1_Score','Normalized_Comparison2_Score'])
c1 = df['Normalized_Comparison1_Score'].values.tolist()[:-3]
c2 = df['Normalized_Comparison2_Score'].values.tolist()[:-3]

stats.ttest_ind(c1,c2)

# Non-Ceiled
data = pd.read_excel (r'P:\\Research2020\\Brain-Score_Lang\\Lang_Specificity\\Fedorenko2016v3\\Fedorenko2016-Language-Non-Language-specificity-non-ceiled-2020-06-03.xlsx') 
df = pd.DataFrame(data, columns= ['Predictivity_Comparison1','Predictivity_Comparison2'])
c1 = df['Predictivity_Comparison1'].values.tolist()[:-3]
c2 = df['Predictivity_Comparison2'].values.tolist()[:-3]

stats.ttest_ind(c1,c2)
