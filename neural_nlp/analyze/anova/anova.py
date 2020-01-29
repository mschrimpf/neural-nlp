import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
ev_vars = np.zeros(n_var)
n_models = scores.shape[0]
dataset = pd.DataFrame({'scores': scores,
                        'base_model': base_model_cat,
                        'tot_depth': tot_depth_cat,
                        'degrees': deg_cat,
                        'rel_depth': rel_depth_cat,
                        'rf_deg': rf_deg_cat,
                        'layer_type': layer_type_cat,
                        'depth': depth_cat,
                        'rf_px': rf_px_cat,
                        'features': features_cat,
                        'size': size_cat,
                        'units_num': units_num_cat,
                        })
# Variables to run the ANOVA on
var_names = ['tot_depth','degrees', 'depth', 'rf_deg', 'layer_type']


# Lists to store all the variable combinations
formulas_list=[]   # ANOVA formulas
formulas_list.append([])
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
# First cycle
# Runs through all variables and checks how much variance is explained by a single variable
for var_ind, var in enumerate(var_names):
    formulas_list[0].append('scores ~ C('+var+')')      # Formula for anova on single variable
    lm_list[0].append(smf.ols(formula=formulas_list[0][var_ind], data=dataset).fit())    # Linear model object
    rsqr_list[0,var_ind] = lm_list[0][var_ind].rsquared_adj  # Rsqr of ANOVA
    fvalue_list[0,var_ind] = lm_list[0][var_ind].fvalue      # F-statistic
    # Incremental p-value of the model: for the first variable, it's the p-value of the anova model
    inc_pvalue[0,var_ind] = sm.stats.anova_lm(lm_list[0][var_ind],typ=1)['PR(>F)'][0]
# Best variable is defined as the one with smallest p-Value
best_var[0] = np.argmin(inc_pvalue[0])
best_var_names.append(var_names_list[0][best_var[0]])   # Name of the best variable
var_names_list.append(list(var_names_list[0]))    # Creates list with second order variables
del(var_names_list[1][best_var[0]])               # Deletes best variable for level 1 from the second order list
inc_fvalue[0] = lm_list[0][best_var[0]].f_pvalue  # p-value of the best var
for n_iter in range(n_var-1):
    formulas_list.append(list(var_names_list[n_iter+1])) # Creates new list to store the formulas for the next level of the ANOVA
    lm_list.append([]) # Creates new list to store the linear models
    for var_ind in range(len(var_names_list[n_iter+1])):  # For each remaining var that has not been selected in a previous level
        # Creates formula by adding to the previous level best formula the current variable name
        formulas_list[n_iter+1][var_ind] = formulas_list[n_iter][best_var[n_iter]] + '+ C('+ var_names_list[n_iter+1][var_ind] +')'
        lm_list[n_iter+1].append(smf.ols(formula=formulas_list[n_iter+1][var_ind], data=dataset).fit()) # Linear model with current variable
        rsqr_list[n_iter+1,var_ind] = lm_list[n_iter+1][var_ind].rsquared_adj # Rsqr of ANOVA
        fvalue_list[n_iter+1,var_ind] = lm_list[n_iter+1][var_ind].fvalue # F-statistic
        # Incremental p-value of the model: is obtained by doing an ANOVA test between the two linear models, the best one from the previous level
        # and the new one by adding the current variable
        inc_pvalue[n_iter+1,var_ind] = sm.stats.anova_lm(lm_list[n_iter][best_var[n_iter]],lm_list[n_iter+1][var_ind],typ=1)['Pr(>F)'][1]
    # Best variable is defined with the lowest incremental p-value
    best_var[n_iter+1] = np.argmin(inc_pvalue[n_iter+1])
    # Incremental p-value of the best model
    inc_fvalue[n_iter+1] = sm.stats.anova_lm(lm_list[n_iter][best_var[n_iter]],lm_list[n_iter+1][best_var[n_iter+1]],typ=1)['Pr(>F)'][1]
    var_names_list.append(list(var_names_list[n_iter+1])) # Creates list with next level variables
    best_var_names.append(var_names_list[n_iter+1][best_var[n_iter+1]]) # Current best variable
    del(var_names_list[n_iter+2][best_var[n_iter+1]])  # Deletes current level best variable from the next level list
anova_res = sm.stats.anova_lm(lm_list[-1][0], typ=1) # Final complete ANOVA model with all the variables ordered by importance
vars_ord = best_var_names
sum_sqrs = anova_res['sum_sq'].values  # Sum of squared errors of the final ANOVA model
r_sqr = lm_list[-1][0].rsquared_adj # R-square of the final model
print('%%%%%')
print('R-sqr: '+r_sqr)
print(anova_res)
print('%%%%%')
for v_ind, v in enumerate(var_names):
    aux_ind = np.argwhere(np.array(vars_ord) == v)[0,0]
    # Explained variance by each variable: Defined by the sum squared errors of each variable divided by the sum of squared errors
    ev_vars[v_ind] = sum_sqrs[aux_ind] / sum_sqrs[s_ind].sum()
