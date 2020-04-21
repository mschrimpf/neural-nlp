from getMeanLayerPref_perSubject import getMeanLayerPref_perSubject

for sub in ['018', '199', '215', '288', '289', '296', '343', '366', '407','426']:
    getMeanLayerPref_perSubject(subjectID = sub,
                                score_name = 'benchmark=Pereira2018-encoding,model=gpt2-xl,subsample=None.pkl',
                                metric = 'mean',
                                categorize = False, 
                                plotROIs = True,
                                plotNetworks = True)
    
    
print('Generated ROI-wise layer preference analysis for subject: ' + sub)
