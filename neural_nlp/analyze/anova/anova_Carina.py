import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-bn", "--benchmark_name", help="Benchmark name \
(should be one of Pereira2018-encoding/Fedorenko2016-encoding/stories_froi_bold4s-encoding)")
args = parser.parse_args()

glue_scores = pd.read_csv('GLUE_ANOVA.csv')
list_of_models = glue_scores["model"].tolist()

#Select benchmark to run
benchmark_score = pd.read_csv('model-scores_{}.csv'.format(args.benchmark_name)).filter(["model","score"])
#Fedorenko2016-encoding
#Pereira2018
#stories_froi_bold4s-encoding

#define overlapping model set (t5-11b, e.g., not in this set for Pereira2018.)
list_of_models_benchmark = benchmark_score["model"].tolist()
my_model_list = sorted(set(list_of_models).intersection(list_of_models_benchmark))
benchmark_score = benchmark_score[(benchmark_score.model.isin(my_model_list))]

#Keep only the rows with the highest Brainscore for each model
benchmark_score = benchmark_score.sort_values('score').drop_duplicates(["model"],keep='last')

#restrict to intersection of model lists
glue_scores = glue_scores[(glue_scores.model.isin(my_model_list))]
dataset = glue_scores.join(benchmark_score.set_index("model"), on = "model")
print(dataset)

# Variables to run the ANOVA on
var_names = ['CoLA_MCC','SST_2_Acc','MRPC_Acc','STS_B_PCC','QQP_Acc','MNLI_m_Acc','QNLI_Acc','RTE_Acc']

n_var = len(var_names)
ev_vars = np.zeros(n_var)
n_models = dataset["model"].shape[0] #counts # of rows

# Lists to store all the variable combinations
formulas_list=[]   # ANOVA formulas
formulas_list.append([]) #now list of lists
lm_list=[]         # Linear model objects
lm_list.append([])
# Numpy arrays
rsqr_list = np.zeros((len(var_names), len(var_names)))      # R-sqr
fvalue_list = np.zeros((len(var_names), len(var_names)))    # F-statistic
inc_pvalue = np.ones((len(var_names), len(var_names)))      # Incremental pValue of adding variable to the ANOVA
best_var=np.zeros(len(var_names)).astype(int)            # Variable that explains most variance (indice)
best_var_names = []                                      # Variable that explains most variance (name)
inc_fvalue = np.ones(len(var_names))                     # Incremental F-statistic of adding variable
var_names_list=[]
var_names_list.append(var_names)
additional_explained_variance = []

# First cycle
# Runs through all variables and checks how much variance is explained by a single variable
for var_ind, var in enumerate(var_names):
    formulas_list[0].append('score ~ '+var)      # Formula for anova on single variable (C() would turn integer variable into categorical)
    lm_list[0].append(smf.ols(formula=formulas_list[0][var_ind], data=dataset).fit())    # Linear model object
    #here, it only stores the according values into the matrices
    rsqr_list[0,var_ind] = lm_list[0][var_ind].rsquared_adj  # Adjusted Rsqr of ANOVA
    fvalue_list[0,var_ind] = lm_list[0][var_ind].fvalue      # F-statistic
    # Incremental p-value of the model: for the first variable, it's the p-value of the anova model
    inc_pvalue[0,var_ind] = sm.stats.anova_lm(lm_list[0][var_ind],typ=1)['PR(>F)'][0] #P-value for significance comparing to previous model in args
    print('Incremental p value for: ',var, ' is value: ', inc_pvalue[0,var_ind],'\n')
# Best variable is defined as the one with smallest p-Value
best_var[0] = np.argmin(inc_pvalue[0])
print('\nThe best var according to inc_pvalue has index: ',best_var[0])
best_var_names.append(var_names_list[0][best_var[0]])   # Name of the best variable
print('Name of best variable: ', var_names_list[0][best_var[0]])
print(best_var_names)
var_names_list.append(list(var_names_list[0]))    # Creates list with second order variables
print(var_names_list)
del(var_names_list[1][best_var[0]])               # Deletes best variable for level 1 from the second order list
#Set f-value for best model in list
print('F-value for best model in list (before): ', inc_fvalue[0])
inc_fvalue[0] = lm_list[0][best_var[0]].f_pvalue  # p-value of the best var
print('F-value for best model in list (after): ', inc_fvalue[0],'\n\n')
# Now iterate through rest of list (the one from which you've deleted the best variable name)
exp_var = lm_list[0][best_var[0]].rsquared
print("Initial r^2: " , exp_var)
additional_explained_variance.append(exp_var)
for n_iter in range(n_var-1):
    formulas_list.append(list(var_names_list[n_iter+1])) # Creates new list to store the formulas for the next level of the ANOVA
    lm_list.append([]) # Creates new list to store the linear models
    for var_ind in range(len(var_names_list[n_iter+1])):  # For each remaining var that has not been selected in a previous level
        # Creates formula by adding to the previous level best formula the current variable name
        formulas_list[n_iter+1][var_ind] = formulas_list[n_iter][best_var[n_iter]] + '+ '+ var_names_list[n_iter+1][var_ind]
        print('Formula :', formulas_list[n_iter+1][var_ind])
        lm_list[n_iter+1].append(smf.ols(formula=formulas_list[n_iter+1][var_ind], data=dataset).fit()) # Linear model with current variable
        rsqr_list[n_iter+1,var_ind] = lm_list[n_iter+1][var_ind].rsquared_adj # Rsqr of ANOVA
        fvalue_list[n_iter+1,var_ind] = lm_list[n_iter+1][var_ind].fvalue # F-statistic
        # Incremental p-value of the model: is obtained by doing an ANOVA test between the two linear models, the best one from the previous level
        # and the new one by adding the current variable
        inc_pvalue[n_iter+1,var_ind] = sm.stats.anova_lm(lm_list[n_iter][best_var[n_iter]],lm_list[n_iter+1][var_ind],typ=1)['Pr(>F)'][1]
        print('Incremental p-value: ', inc_pvalue[n_iter+1,var_ind])
    # Best variable is defined with the lowest incremental p-value
    best_var[n_iter+1] = np.argmin(inc_pvalue[n_iter+1])
    print('\nBest variable (lowest inc_pvalue): ', best_var[n_iter+1])
    # Incremental p-value of the best model
    inc_fvalue[n_iter+1] = sm.stats.anova_lm(lm_list[n_iter][best_var[n_iter]],lm_list[n_iter+1][best_var[n_iter+1]],typ=1)['Pr(>F)'][1]
    print('Incremental p-value of the best model:',inc_fvalue[n_iter+1])
    my_exp_var = lm_list[n_iter+1][best_var[n_iter+1]].rsquared
    print('Current r^2: ', my_exp_var,'\n')
    additional_explained_variance.append(my_exp_var-exp_var)
    exp_var = my_exp_var
    var_names_list.append(list(var_names_list[n_iter+1])) # Creates list with next level variables
    best_var_names.append(var_names_list[n_iter+1][best_var[n_iter+1]]) # Current best variable
    del(var_names_list[n_iter+2][best_var[n_iter+1]])  # Deletes current level best variable from the next level list
anova_res = sm.stats.anova_lm(lm_list[-1][0], typ=1) # Final complete ANOVA model with all the variables ordered by importance
vars_ord = best_var_names
sum_sqrs = anova_res['sum_sq'].values  # Sum of squared errors of the final ANOVA model
r_sqr = lm_list[-1][0].rsquared_adj # R-square of the final model
print('%%%%%')
print('Model summary')
print(lm_list[-1][0].summary())

print('%%%%%')
print('R-sqr: ',r_sqr)
print('%%%%%')
print('Anova results')
print(anova_res)
print('\n%%%%%')
print('Additional explained variance with each model, added in order of importance: ')
mylist = list(zip(best_var_names,additional_explained_variance))
print(mylist)
