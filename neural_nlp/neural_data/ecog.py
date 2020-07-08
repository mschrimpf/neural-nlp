import os
from glob import glob

import logging
import numpy as np
import scipy.io as sio
import xarray as xr
from brainio_base.assemblies import NeuroidAssembly
from pathlib import Path
from scipy import stats

from neural_nlp.stimuli import StimulusSet

_logger = logging.getLogger(__name__)


# no need to @store()  since it is a very small file
def load_Fedorenko2016(electrodes, version):
    ressources_dir = Path(__file__).parent.parent.parent / 'ressources'
    neural_data_dir = ressources_dir / 'neural_data' / 'ecog-Fedorenko2016/'
    stim_data_dir = ressources_dir / 'stimuli' / 'sentences_8'
    _logger.info(f'Neural data directory: {neural_data_dir}')
    filepaths_stim = glob(os.path.join(stim_data_dir, '*.txt'))

    # ECoG
    data = None

    # For language responsive electrodes:
    if electrodes == 'language':

        # Create a subject ID list corresponding to language electrodes
        subject1 = np.repeat(1, 47)
        subject2 = np.repeat(2, 9)
        subject3 = np.repeat(3, 9)
        subject4 = np.repeat(4, 15)
        subject5 = np.repeat(5, 18)

        if version == 1:
            filepath_neural = glob(os.path.join(neural_data_dir, '*ecog.mat'))

        if version == 2:
            filepath_neural = glob(os.path.join(neural_data_dir, '*metadata_lang.mat'))
            
        if version == 3:
            subject1 = np.repeat(1, 47)
            subject2 = np.repeat(2, 8)
            subject3 = np.repeat(3, 9)
            subject4 = np.repeat(4, 15)
            subject5 = np.repeat(5, 18)
            
            filepath_neural = glob(os.path.join(neural_data_dir, '*g_lang_v3.mat'))
            
        if version == 4:            
            subject1 = np.repeat(1, 49)
            subject2 = np.repeat(2, 8)
            subject3 = np.repeat(3, 10)
            subject4 = np.repeat(4, 16)
            subject5 = np.repeat(5, 19)            
            subject6 = np.repeat(6, 3)
            
            filepath_neural = glob(os.path.join(neural_data_dir, '*g_lang_v4.mat'))

        _logger.debug(f'Running Fedorenko2016 benchmark with language responsive electrodes, data version: {version}')

    # For non-noisy electrodes
    if electrodes == 'all':

        # Create a subject ID list corresponding to all electrodes
        subject1 = np.repeat(1, 70)
        subject2 = np.repeat(2, 35)
        subject3 = np.repeat(3, 20)
        subject4 = np.repeat(4, 29)
        subject5 = np.repeat(5, 26)

        if version == 1:
            filepath_neural = glob(os.path.join(neural_data_dir, '*ecog_all.mat'))

        if version == 2:
            filepath_neural = glob(os.path.join(neural_data_dir, '*metadata_all.mat'))
            
        if version == 3:
            subject1 = np.repeat(1, 67)
            subject2 = np.repeat(2, 35)
            subject3 = np.repeat(3, 20)
            subject4 = np.repeat(4, 29)
            subject5 = np.repeat(5, 26)
            
            filepath_neural = glob(os.path.join(neural_data_dir, '*all_v3.mat'))
            
        if version == 4:
            subject1 = np.repeat(1, 63)
            subject2 = np.repeat(2, 35)
            subject3 = np.repeat(3, 21)
            subject4 = np.repeat(4, 29)
            subject5 = np.repeat(5, 27)
            subject6 = np.repeat(6, 9)
            
            filepath_neural = glob(os.path.join(neural_data_dir, '*all_v4.mat'))

        _logger.debug('Running Fedorenko2016 benchmark with non-noisy electrodes, data version: ', version)

        # For non-noisy electrodes
    if electrodes == 'non-language':
        
        if version == 1 or version == 2:
            filepath_neural = glob(os.path.join(neural_data_dir, '*nonlang.mat'))
    
            # Create a subject ID list corresponding to non-language electrodes
            subject1 = np.repeat(1, 28)
            subject2 = np.repeat(2, 31)
            subject3 = np.repeat(3, 14)
            subject4 = np.repeat(4, 19)
            subject5 = np.repeat(5, 16)
        
        if version == 3:
            filepath_neural = glob(os.path.join(neural_data_dir, '*nonlang_v3.mat'))
    
            # Create a subject ID list corresponding to non-language electrodes
            subject1 = np.repeat(1, 25) # 47 lang selective,
            subject2 = np.repeat(2, 31)
            subject3 = np.repeat(3, 14)
            subject4 = np.repeat(4, 19)
            subject5 = np.repeat(5, 16) # 10 lang electrodes in the non-noisy
            
        if version == 4:
            filepath_neural = glob(os.path.join(neural_data_dir, '*nonlang_v4.mat'))
    
            # Create a subject ID list corresponding to non-language electrodes
            subject1 = np.repeat(1, 22) 
            subject2 = np.repeat(2, 31)
            subject3 = np.repeat(3, 15)
            subject4 = np.repeat(4, 19)
            subject5 = np.repeat(5, 18) 
            subject6 = np.repeat(6, 6) 


        _logger.debug(f'Running Fedorenko2016 benchmark with non-language electrodes, data version: {version}')

    ecog_mat = sio.loadmat(filepath_neural[0])
    ecog_mtrix = ecog_mat['ecog']

    if version == 1:  # Manually z-score the version 1 data
        ecog_z = stats.zscore(ecog_mtrix, 1)
    if version == 2 or version == 3 or version == 4:
        ecog_z = ecog_mtrix

    ecog_mtrix_T = np.transpose(ecog_z)

    num_words = list(range(np.shape(ecog_mtrix_T)[0]))
    new_sent_idx = num_words[::8]

    # Average across word representations
    sent_avg_ecog = []
    for i in new_sent_idx:
        eight_words = ecog_mtrix_T[i:i + 8, :]
        sent_avg = np.mean(eight_words, 0)
        sent_avg_ecog.append(sent_avg)

    # Stimuli
    for filepath in filepaths_stim:
        with open(filepath, 'r') as file1:
            f1 = file1.readlines()

        _logger.debug(f1)

        sentences = []
        sentence_words, word_nums = [], []
        for sentence in f1:
            sentence = sentence.split(' ')
            sentences.append(sentence)
            word_counter = 0

            for word in sentence:
                if word == '\n':
                    continue
                word = word.rstrip('\n')
                sentence_words.append(word)
                word_nums.append(word_counter)
                word_counter += 1

        _logger.debug(sentence_words)

    # Create sentenceID list
    sentence_lst = list(range(0, 52))
    sentenceID = np.repeat(sentence_lst, 8)
    
    if version == 1 or version == 2 or version == 3:
        subjectID = np.concatenate([subject1, subject2, subject3, subject4, subject5], axis=0)

    if version == 4:
        subjectID = np.concatenate([subject1, subject2, subject3, subject4, subject5, subject6], axis=0)

    # Create a list for each word number
    word_number = list(range(np.shape(ecog_mtrix_T)[0]))

    # Add a pd df as the stimulus_set
    zipped_lst = list(zip(sentenceID, word_number, sentence_words))
    df_stimulus_set = StimulusSet(zipped_lst, columns=['sentence_id', 'stimulus_id', 'word'])
    df_stimulus_set.name = 'Fedorenko2016.ecog'

    # xarray
    electrode_numbers = list(range(np.shape(ecog_mtrix_T)[1]))
    assembly = xr.DataArray(ecog_mtrix_T,
                            dims=('presentation', 'neuroid'),
                            coords={'stimulus_id': ('presentation', word_number),
                                    'word': ('presentation', sentence_words),
                                    'word_num': ('presentation', word_nums),
                                    'sentence_id': ('presentation', sentenceID),
                                    'electrode': ('neuroid', electrode_numbers),
                                    'neuroid_id': ('neuroid', electrode_numbers),
                                    'subject_UID': ('neuroid', subjectID),  # Name is subject_UID for consistency
                                    })

    assembly.attrs['stimulus_set'] = df_stimulus_set  # Add the stimulus_set dataframe
    data = assembly if data is None else xr.concat(data, assembly)
    return NeuroidAssembly(data)
