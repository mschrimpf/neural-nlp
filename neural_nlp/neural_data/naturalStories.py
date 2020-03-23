import logging
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm

from neural_nlp.stimuli import StimulusSet
from result_caching import store

_logger = logging.getLogger(__name__)


@store()
def load_naturalStories():
    ressources_dir = Path(__file__).parent.parent.parent / 'ressources'
    data_path = ressources_dir / 'neural_data' / 'naturalstories_RTS'
    data_file = data_path / 'processed_RTs.csv'
    _logger.info(f'Data file: {data_file}')

    # get data
    data = pd.read_csv(data_file)

    # get unique word identifier tuples and order in order of stories
    item_ID = np.array(data['item'])
    zone_ID = np.array(data['zone'])
    zpd_lst = list(zip(item_ID, zone_ID))
    unique_zpd_lst = list(set(zpd_lst))
    unique_zpd_lst = sorted(unique_zpd_lst, key=lambda tup: (tup[0], tup[1]))

    # get unique WorkerIds
    subjects = data.WorkerId.unique()

    # ====== create matrix ======
    r_dim = len(unique_zpd_lst)
    c_dim = len(subjects)

    # default value for a subject's not having an RT for a story/word is NaN
    matrix = np.empty((r_dim, c_dim))
    matrix[:] = np.nan

    # set row and column indices for matrix
    r_indices = {unique_zpd_lst[i]: i for i in range(r_dim)}
    c_indices = {subjects[i]: i for i in range(c_dim)}

    # populate meta information dictionary for subjects xarray dimension
    metaInfo_subjects = {}

    for index, d in tqdm(data.iterrows(), total=len(data), desc='indices'):
        r = r_indices[(d['item'], d['zone'])]
        c = c_indices[d['WorkerId']]
        matrix[r][c] = d['RT']
        key = d['WorkerId']
        if key not in metaInfo_subjects:
            metaInfo_subjects[key] = (d['correct'], d['WorkTimeInSeconds'])

    matrix = np.array(matrix)

    # get subjects' metadata
    correct_meta = [v[0] for v in metaInfo_subjects.values()]
    WorkTimeInSeconds_meta = [v[1] for v in metaInfo_subjects.values()]

    # get metadata for presentation dimension
    word_df = pd.read_csv(f'{data_path}/all_stories.tok', sep='\t')
    voc_item_ID = np.array(word_df['item'])
    voc_zone_ID = np.array(word_df['zone'])
    voc_word = np.array(word_df['word'])

    # get sentence_IDs (finds 481 sentences)
    sentence_ID = []
    idx = 1
    for i, elm in enumerate(voc_word):
        sentence_ID.append(idx)
        if elm.endswith((".", "?", "!", ".'", "?'", "!'", ";'")):
            if i + 1 < len(voc_word):
                if not (voc_word[i + 1].islower() or voc_word[i] == "Mr."):
                    idx += 1

    # get IDs of words within a sentence
    word_within_a_sentence_ID = []
    idx = 0
    for i, elm in enumerate(voc_word):
        idx += 1
        word_within_a_sentence_ID.append(idx)
        if elm.endswith((".", "?", "!", ".'", "?'", "!'", ";'")):
            if i + 1 < len(voc_word):
                if not (voc_word[i + 1].islower() or voc_word[i] == "Mr."):
                    idx = 0
            else:
                idx = 0

    # stimulus_ID
    stimulus_ID = list(range(1, len(voc_word) + 1))

    # set df_stimulus_set for attributes
    df_stimulus_set = word_df[['word', 'item', 'zone']]
    df_stimulus_set = StimulusSet(df_stimulus_set)
    df_stimulus_set['story_id'] = df_stimulus_set['item']
    df_stimulus_set['stimulus_id'] = stimulus_ID
    df_stimulus_set['sentence_id'] = sentence_ID
    df_stimulus_set['word_id'] = voc_zone_ID
    df_stimulus_set['word_within_sentence_id'] = word_within_a_sentence_ID
    df_stimulus_set.name = 'naturalStories'

    # build xarray
    # voc_word = word
    # voc_item_ID = index of story (1-10)
    # voc_zone_ID = index of words within a story
    # sentence_ID = index of words within each sentence
    # stimulus_ID = unique index of word across all stories
    # subjects = WorkerIDs
    # correct_meta = number of correct answers in comprehension questions
    assembly = xr.DataArray(matrix,
                            dims=('presentation', 'subjects'),
                            coords={'word': ('presentation', voc_word),
                                    'story_id': ('presentation', voc_item_ID),
                                    'word_id': ('presentation', voc_zone_ID),
                                    'word_within_sentence_id': ('presentation', word_within_a_sentence_ID),
                                    'sentence_id': ('presentation', sentence_ID),
                                    'stimulus_id': ('presentation', stimulus_ID),
                                    'subject_id': ('subjects', subjects),
                                    'correct': ('subjects', correct_meta),
                                    'WorkTimeInSeconds': ('subjects', WorkTimeInSeconds_meta)
                                    })

    assembly.attrs['stimulus_set'] = df_stimulus_set  # Add the stimulus_set dataframe
    return assembly


if __name__ == '__main__':
    data = load_naturalStories()
