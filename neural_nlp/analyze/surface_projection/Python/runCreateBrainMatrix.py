from createBrainMatrixFunc import createBrainMatrix

for sub in ['018', '199', '215', '288', '289', '296', '343', '366', '407','426']:
    createBrainMatrix(subjectID = sub, score_name = 'benchmark=Pereira2018-encoding,model=gpt2-xl,subsample=None.pkl')
    print('Generated brain matrix for subject: ', sub)
