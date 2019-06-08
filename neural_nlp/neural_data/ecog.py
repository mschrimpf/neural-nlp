import xarray as xr
import logging
import os
import re
from glob import glob

import numpy as np
import pandas as pd
from result_caching import store

neural_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
                                               'ressources', 'neural_data', 'ecog'))
_logger = logging.getLogger(__name__)


def _combine_columns(recordings, column_name):
    columns = [column for column in recordings.columns if column.startswith(f'{column_name}_')]
    recordings[column_name] = [[getattr(row, field) for field in columns]
                               for _, row in recordings.iterrows()]
    recordings.drop(columns, axis='columns', inplace=True)


@store()
def load():
    filepaths = glob(os.path.join(neural_data_dir, '*.mat'))
    data = None
    for filepath in filepaths:
        print(filepath, '...')
        subject = re.search('all_subj_info_S([0-9]+)\.mat', filepath)
        subject = int(subject.group(1))
        if subject != 6: continue

        electrode_meta = pd.read_csv(filepath + '-electrodes.csv')
        for combine_column in ['native_coords', 'norm_coords']:
            _combine_columns(electrode_meta, combine_column)

        recordings = pd.read_csv(filepath + '-recordings.csv')

        for combine_column in ['HGA_avg_eois', 'HGA_avg_lang', 'WordList']:
            _combine_columns(recordings, combine_column)

        hga_columns = [column for column in recordings.columns if re.match('HGA_[0-9]+', column)]
        # TODO: fix row- vs column-major ordering
        # The HGA field in trial info contains high-gamma data for each trial (e.g. for subject 1 trial 1,
        # the 135x128x8 matrix contains 135 time points for each of 8 words in 128 electrodes.
        num_electrodes = len(electrode_meta)
        num_words = len(next(recordings.iterrows())[1]['WordList'])
        recordings['HGA'] = [np.array([getattr(row, field) for field in hga_columns], order='F')
                                 .reshape((-1, num_electrodes, num_words)) for _, row in recordings.iterrows()]
        recordings.drop(hga_columns, axis='columns', inplace=True)

        recordings['subject'] = subject

        assembly = xr.DataArray(recordings['HGA'],
                                coords={
                                    # timepoint
                                    'timepoint': ('timepoint', list(range(recordings['HGA'].shape[0]))),
                                    # electrode
                                    'electrode_num': ('electrode', electrode_meta['num']),
                                    'electrode_cat': ('electrode', electrode_meta['cat']),
                                    'electrode_native_coords': ('electrode', electrode_meta['native_coords']),
                                    'electrode_norm_coords': ('electrode', electrode_meta['norm_coords']),
                                    # presentation
                                    'trial_num': ('presentation', recordings['TrialNum']),
                                    'subject': ('presentation', [subject] * len(recordings)),
                                    'run_num': ('presentation', recordings['RunNum']),
                                    'item_num': ('presentation', recordings['ItemNum']),
                                    'word_type': ('presentation', recordings['WordType']),
                                    'correct': ('presentation', recordings['Correct']),
                                    'trial_onset': ('presentation', recordings['trial_onset']),
                                    # word
                                    'word_list': ('word', recordings['WordList']),
                                    'avg_eois': ('word', recordings['HGA_avg_eois']),
                                    'avg_lang': ('word', recordings['HGA_avg_lang']),
                                },
                                dims=['timepoint', 'electrode', 'presentation', 'word'])

        data = assembly if data is None else xr.concat(data, assembly)
    return data


if __name__ == '__main__':
    data = load()
    pass
