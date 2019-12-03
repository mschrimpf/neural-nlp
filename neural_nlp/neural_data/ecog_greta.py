import xarray as xr
import logging
import os
import re
from glob import glob
import numpy as np
import pandas as pd
from result_caching import store
from pathlib import Path
import scipy.io as sio

# neural_data_dir = Path(os.getcwd()) / 'ressources' / 'neural_data' / 'ecog' / 'Fedorenko2016' # TO DO create a setup file

# MANUAL DIR: To give access to other people running it from different directories
neural_data_dir = '/om/user/gretatu/neural-nlp/ressources/neural_data/ecog/Fedorenko2016/'

print('--------- Neural data directory: --------- \n', neural_data_dir)

# stim_data_dir = Path(os.getcwd()) / 'ressources' / 'stimuli' / 'sentences_8'
stim_data_dir = '/om/user/gretatu/neural-nlp/ressources/stimuli/sentences_8'

_logger = logging.getLogger(__name__)


@store()
def load_Fedorenko2016():
    # Argument: 
    filepaths_neural = glob(os.path.join(neural_data_dir, '*.mat'))
    print('Filepaths neural: ', filepaths_neural)
    filepaths_stim = glob(os.path.join(stim_data_dir, '*.txt'))
    
    data = None
    
    # ECoG
    for filepath in filepaths_neural:
        # For each file (currently one)
        print(filepath, '...')

        ecog_mat = sio.loadmat(filepath) 
        ecog_mtrix = ecog_mat['ecog']

        ecog_mtrix_T = np.transpose(ecog_mtrix)

        num_words = list(range(np.shape(ecog_mtrix_T)[0]))
        new_sent_idx = num_words[::8]

        # Average across word representations
        sent_avg_ecog = []
        for i in new_sent_idx:
            eight_words = ecog_mtrix_T[i:i+8,:]
            sent_avg = np.mean(eight_words,0)
            sent_avg_ecog.append(sent_avg)
            
    # Stimuli
    for filepath in filepaths_stim:   
        
        with open(filepath, 'r') as file1:
            f1 = file1.readlines()
        
        print(f1)
        
        sentences = []
        words = []
        for sentence in f1:
            sentence = sentence.split(' ')
            sentences.append(sentence)

            for word in sentence:
                words.append(word)

        newline = ['\n']
        sentence_words = [word for word in words if word not in newline]
        print(sentence_words)
        
    # Create sentenceID list
    sentence_lst = list(range(0,52))
    sentenceID = np.repeat(sentence_lst, 8)
    
    # Create a subject ID list corresponding to electrodes
    subject1 = np.repeat(1, 47) # Subject 1 has these language responsive electrodes
    subject2 = np.repeat(2, 9)
    subject3 = np.repeat(3, 9)
    subject4 = np.repeat(4, 15)
    subject5 = np.repeat(5, 18)

    subjectID = np.concatenate([subject1, subject2, subject3, subject4, subject5], axis=0)
    
    # Add a pd df as the stimulus_set
    zipped_lst = list(zip(sentenceID, sentence_words))  
    df_stimulus_set = pd.DataFrame(zipped_lst, columns = ['stimulus_id', 'word'])  
    
   
    # xarray - currently two dims
    assembly = xr.DataArray(ecog_mtrix_T, # Do we want the avg word response too?
                    dims =('timepoint', 'electrode'),
                    coords = {'timepoint1': ('timepoint', list(range(np.shape(ecog_mtrix_T)[0]))),
                             'word': ('timepoint', sentence_words), 
                             'stimulus_id': ('timepoint', sentenceID), 
                             'electrode1': ('electrode', list(range(np.shape(ecog_mtrix_T)[1]))),
                             'subject_UID': ('electrode', subjectID), # Name is subject_UID for consistency
                             })   
                                                     
                              
    
    assembly.attrs['stimulus_set'] = df_stimulus_set # Add the stimulus_set dataframe
    
    # Trying to add the stimulus_UID as an attribute or additional coordinate?
    # assembly.assign_coords['subject_UID'] = subjectID
    
    
    # TO-DO: add avg sentence representations? 
#                              'sentence_number': ('sentence', list(range(int((np.shape(ecog_mtrix_T)[0]/8))))),
#                              'avg_sentence_response': ('sentence', sent_avg_ecog)
    
    data = assembly if data is None else xr.concat(data, assembly)
        
    return data


if __name__ == '__main__':
    data = load()
    pass
