import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#data (for each benchmark: (1) ordered list of tasks; (2) additional variance explained by adding the variable to LM in specified order)
#from anova_Carina.py script
stories_names = ['QQP_Acc', 'STS_B_PCC', 'SST_2_Acc', 'QNLI_Acc', 'CoLA_MCC', 'MNLI_m_Acc', 'RTE_Acc', 'MRPC_Acc']
stories_r2_diff = [0.6404136845000258, 0.025199070326026196, 0.005658985359343394, 0.11876633141622006, 0.10443457709162718, 0.017351905550674118, 0.00895594661999366, 0.0035178559602137005]

fedorenko_names = ['STS_B_PCC', 'CoLA_MCC', 'SST_2_Acc', 'QQP_Acc', 'MNLI_m_Acc', 'MRPC_Acc', 'QNLI_Acc', 'RTE_Acc']
fedorenko_r2_diff = [0.19183573133080745, 0.23313968897501935, 0.09142335875638563, 0.028918244457928455, 0.005521402775523376, 0.012612544721174901, 0.026189741584257797, 0.0012249225300056121]

pereira_names = ['MNLI_m_Acc', 'CoLA_MCC', 'MRPC_Acc', 'QNLI_Acc', 'STS_B_PCC', 'QQP_Acc', 'SST_2_Acc', 'RTE_Acc']
pereira_r2_diff = [0.6416725193062608, 0.10284595338439761, 0.028297126455801402, 0.024875618889072637, 0.01751584193267286, 0.007368228955993317, 0.013203801200430854, 3.354835385627197e-05]


#create dataframe from data
task = stories_names + fedorenko_names + pereira_names

N = len(stories_names)
benchmark = ['stories']*N
benchmark = benchmark + ['Fedorenko']*N
benchmark = benchmark + ['Pereira']*N

scores = stories_r2_diff + fedorenko_r2_diff + pereira_r2_diff

data = [benchmark, task, scores]

rows = list(zip(data[0], data[1], data[2]))
headers = ['benchmark', 'task', 'scores']
df = pd.DataFrame(rows, columns=headers)

df


#create barplot from dataframe
fig, ax = plt.subplots(figsize=(10,7))

my_task = df['task'].drop_duplicates()
margin_bottom = np.zeros(len(df['benchmark'].drop_duplicates()))
colors = ["#f7fbff","#deebf7","#c6dbef","#9ecae1","#6baed6","#4292c6","#2171b5","#084594"]

for num, tsk in enumerate(my_task):
    values = list(df[df['task'] == tsk].loc[:, 'scores'])

    df[df['task'] == tsk].plot.bar(x='benchmark',y='scores', ax=ax, stacked=True,
                                    bottom = margin_bottom, color=colors[num], label=tsk)
    margin_bottom += values

ax.legend(loc='lower left')
plt.ylabel('Additional variance explained (r-squared)')
plt.xlabel('Benchmark (encoding)')
#plt.show()

plt.savefig('/Users/carina.kauf/Desktop/MIT/NeuralNLP_ResearchProject/img/stackedBarplot_explVar.png', bbox_inches='tight')
