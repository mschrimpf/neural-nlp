import glob
import logging
import os
import re

import numpy as np
import pandas as pd
import xarray as xr
from mkgu.assemblies import DataAssembly

from neural_nlp.stimuli import NaturalisticStories

neural_data_dir = os.path.join(os.path.dirname(__file__), '..', 'ressources', 'neural_data')
_logger = logging.getLogger(__name__)


def load_rdm_sentences(story='Boar', roi_filter='from90to100', bold_shift_seconds=4):
    timepoint_rdms = load_rdm_timepoints(story, roi_filter)
    meta_data = load_sentences_meta(story)
    del meta_data['fullSentence']
    meta_data.dropna(inplace=True)
    mapping_column = 'shiftBOLD_{}sec'.format(bold_shift_seconds)
    timepoints = meta_data[mapping_column].values.astype(int)
    # filter and annotate
    assert all(timepoint in timepoint_rdms['timepoint'].values for timepoint in timepoints)
    timepoint_rdms = timepoint_rdms.sel(timepoint=timepoints)
    coords = {**dict(timepoint_rdms.coords.items()), **{'stimulus': meta_data['reducedSentence'].values}}
    # for some reason, xarray re-packages timepoint as a MultiIndex if we pass all coords at once.
    # to avoid that, we create the DataArray first and then add the additional coords.
    timepoint_coord = ('stimulus', coords['timepoint'])
    del coords['timepoint']
    dims = [dim if dim != 'timepoint' else 'stimulus' for dim in timepoint_rdms.dims]
    data = DataAssembly(timepoint_rdms.values, coords=coords, dims=dims)
    data['timepoint'] = timepoint_coord
    return data


def load_sentences_meta(story):
    filepath = os.path.join(neural_data_dir, 'meta', 'story{}_{}_sentencesByTR.csv'.format(
        NaturalisticStories.set_mapping[story], story))
    _logger.debug("Loading meta {}".format(filepath))
    meta_data = pd.read_csv(filepath)
    return meta_data


def load_rdm_timepoints(story='Boar', roi_filter='from90to100'):
    data = []
    for filepath in glob.glob(os.path.join(
            neural_data_dir, '{}{}*.csv'.format(story + '_', roi_filter))):
        _logger.debug("Loading file {}".format(filepath))
        attributes = re.match('^.*/(?P<story>.*)_from(?P<roi_low>[0-9]+)to(?P<roi_high>[0-9]+)'
                              '(_(?P<subjects>[0-9]+)Subjects)?\.mat_r(?P<region>[0-9]+).csv', filepath)
        _data = pd.read_csv(filepath, header=None)
        num_stimuli = len(_data.columns)
        assert len(_data) % num_stimuli == 0
        num_subjects = len(_data) // num_stimuli
        _data = np.stack([_data.iloc[(subject * num_stimuli):((subject + 1) * num_stimuli)]
                          for subject in range(num_subjects)])
        _data = xr.DataArray([_data], coords={
            'timepoint_left': list(range(num_stimuli)), 'timepoint_right': list(range(num_stimuli)),
            'region': [int(attributes['region'])],
            'subject': list(range(num_subjects))},
                             dims=['region', 'subject', 'timepoint_left', 'timepoint_right'])
        stimuli_meta = lambda x: (['timepoint_left', 'timepoint_right'],
                                  np.broadcast_to(x, [num_stimuli, num_stimuli]))
        _data['story'] = stimuli_meta(attributes['story'])
        _data['roi_low'] = stimuli_meta(int(attributes['roi_low']))
        _data['roi_high'] = stimuli_meta(int(attributes['roi_high']))
        data.append(_data)
    data = xr.concat(data, 'region')

    # re-format timepoint_{left,right} to single dimension
    timepoint_dims = ['timepoint_left', 'timepoint_right']
    # for some reason, xarray re-packages timepoint as a MultiIndex if we pass all coords at once.
    # to avoid that, we create the DataArray first and then add the additional coords.
    dim_coords = {'timepoint': data['timepoint_left'].values}
    nondim_coords = {}
    for name, value in data.coords.items():
        if name in timepoint_dims:
            continue
        if np.array_equal(value.dims, timepoint_dims):
            unique = np.unique(value.values)
            assert unique.size == 1
            value = 'timepoint', np.broadcast_to(unique, dim_coords['timepoint'].shape).copy()
            # need to copy due to https://github.com/pandas-dev/pandas/issues/15860
        (dim_coords if name in data.dims else nondim_coords)[name] = value
    dims = [dim if dim not in timepoint_dims else 'timepoint' for dim in data.dims]
    data = DataAssembly(data.values, coords=dim_coords, dims=dims)
    for name, value in nondim_coords.items():
        data[name] = value
    return data
