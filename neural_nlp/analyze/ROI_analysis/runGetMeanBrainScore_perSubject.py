from getMeanBrainScore_perSubject import getMeanBrainScore_perSubject

for sub in ['018', '199', '215', '288', '289', '296', '343', '366', '407','426']:
    getMeanBrainScore_perSubject(subjectID = sub,
                                 score_name = 'benchmark=Pereira2018-encoding,model=gpt2-xl,subsample=None.pkl',
                                 plotROIs = True,
                                 plotNetworks = True)
    
    
    print('Generated ROI-wise brain-score analysis for subject: ' + sub)
