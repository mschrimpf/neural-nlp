import glob
import logging
import os
import warnings

import fire
import numpy as np
import pandas as pd
import re
import scipy.io
import xarray as xr
from brainio_base.assemblies import merge_data_arrays, DataAssembly, gather_indexes, walk_coords
from pathlib import Path
from tqdm import tqdm
from xarray import DataArray

from neural_nlp.stimuli import NaturalisticStories, StimulusSet
from result_caching import cache, store, store_netcdf

neural_data_dir = (Path(os.path.dirname(__file__)) / '..' / '..' / 'ressources' / 'neural_data' / 'fmri').absolute()
_logger = logging.getLogger(__name__)


def load_voxels():
    annotated_data = load_voxel_data()
    annotated_data = DataAssembly(annotated_data)
    stimulus_set = NaturalisticStories()()
    stimulus_set = _align_stimuli_recordings(stimulus_set, annotated_data)
    annotated_data.attrs['stimulus_set'] = stimulus_set
    annotated_data.attrs['stimulus_set_name'] = stimulus_set.name
    return annotated_data


@store_netcdf()
def load_voxel_data(bold_shift_seconds=4):
    data = load_voxel_timepoints()
    annotated_data = _merge_voxel_meta(data, bold_shift_seconds)
    return annotated_data


def _merge_voxel_meta(data, bold_shift_seconds):
    annotated_data_list = []
    for story in set(data['story'].values):
        try:
            meta_data = load_sentences_meta(story)
            del meta_data['fullSentence']
            meta_data.dropna(inplace=True)
            mapping_column = 'shiftBOLD_{}sec'.format(bold_shift_seconds)
            timepoints = meta_data[mapping_column].values.astype(int)
            # filter
            story_data = data.sel(story=story)
            assert all(timepoint in story_data['timepoint_value'].values for timepoint in timepoints)
            story_data = story_data.sel(timepoint_value=timepoints)
            # re-interpret timepoints as stimuli
            coords = {}
            for coord_name, dims, coord_value in walk_coords(story_data):
                dims = [dim if not dim.startswith('timepoint') else 'stimulus' for dim in dims]
                coords[coord_name] = dims, coord_value
            coords = {**coords, **{'stimulus_sentence': ('stimulus', meta_data['reducedSentence'].values),
                                   'story': ('stimulus', [story] * len(meta_data))}}
            dims = [dim if not dim.startswith('timepoint') else 'stimulus' for dim in story_data.dims]
            annotated_data = xr.DataArray(story_data, coords=coords, dims=dims)
            gather_indexes(annotated_data)
            annotated_data_list.append(annotated_data)
        except (FileNotFoundError, KeyError):
            warnings.warn(f"no meta found for story {story}")
    annotated_data = merge_data_arrays(annotated_data_list)
    return annotated_data


def _align_stimuli_recordings(stimulus_set, assembly):
    def compare_ignore(sentence):
        return sentence.replace(',', '').replace('"', '').replace('\'', '')

    aligned_stimulus_set = []

    stories = set(assembly['story'].values)
    for story in tqdm(sorted(stories), desc='align stimuli', total=len(stories)):
        partial_sentences = assembly.sel(story=story)['stimulus_sentence'].values
        partial_sentences = [compare_ignore(sentence) for sentence in partial_sentences]
        sentence_num = 0

        def append_row(row):
            nonlocal sentence_num
            row = row._replace(sentence_num=sentence_num)
            row = row._replace(sentence=row.sentence.strip())
            aligned_stimulus_set.append(row)
            sentence_num += 1

        story_stimuli = stimulus_set[stimulus_set['story'] == story]
        for row in story_stimuli.itertuples(index=False, name='Stimulus'):
            full_sentence = row.sentence
            # TODO: the following entirely discards ",' etc.
            full_sentence = compare_ignore(full_sentence)
            partial_sentence = [partial_sentence for partial_sentence in partial_sentences
                                if partial_sentence in full_sentence]
            if len(partial_sentence) == 0:
                warnings.warn(f"Sentence {full_sentence} not found in partial sentences")
                row = row._replace(sentence=full_sentence)
                append_row(row)
                continue
            assert len(partial_sentence) == 1
            partial_sentence = partial_sentence[0]
            index = full_sentence.find(partial_sentence)
            assert index >= 0

            # before part
            if index > 0:
                row = row._replace(sentence=full_sentence[0, index])
                append_row(row)
            # part itself
            only_missing_punctuation = index + len(partial_sentence) < len(full_sentence) and \
                                       full_sentence[index + len(partial_sentence):] in NaturalisticStories.sentence_end
            if only_missing_punctuation:
                partial_sentence += full_sentence[index + len(partial_sentence):]
            row = row._replace(sentence=partial_sentence)
            append_row(row)
            # after part
            if index + len(partial_sentence) < len(full_sentence) and not only_missing_punctuation:
                row = row._replace(sentence=full_sentence[index + len(partial_sentence):])
                append_row(row)
        # check
        aligned_story = " ".join(row.sentence for row in aligned_stimulus_set if row.story == story)
        stimulus_set_story = " ".join(row.sentence for row in story_stimuli.itertuples())
        assert aligned_story == compare_ignore(stimulus_set_story)
    aligned_stimulus_set = StimulusSet(aligned_stimulus_set)
    aligned_stimulus_set.name = f"{stimulus_set.name}-aligned"
    return aligned_stimulus_set


def load_voxel_timepoints():
    data_dir = neural_data_dir / 'StoriesData_Dec2018' / 'DataAveragedInEachLangROI'
    files = list(data_dir.glob('*.mat'))
    _logger.info(f"Found {len(files)} voxel files")
    data_list = []
    for filepath in tqdm(files, desc='files', position=0):
        f = scipy.io.loadmat(filepath)
        stories_data = f['data']['stories'][0, 0]
        stories = list(stories_data.dtype.fields)
        for story in stories:
            story_data = stories_data[story][0, 0]
            story_data = DataArray(story_data, coords={
                'voxel_num': np.arange(0, story_data.shape[0]),
                'timepoint_value': ('timepoint', np.arange(0, story_data.shape[1])),
                'story': ('timepoint', [story] * story_data.shape[1]),
            }, dims=['voxel_num', 'timepoint'])
            story_data = story_data.expand_dims('identifier')
            story_data['identifier'] = [filepath.stem]
            gather_indexes(story_data)
            data_list.append(story_data)
    data = merge_data_arrays(data_list)
    return data


@store()
def load_rdm_sentences(story='Boar', roi_filter='from90to100', bold_shift_seconds=4):
    timepoint_rdms = load_rdm_timepoints(story, roi_filter)
    meta_data = load_sentences_meta(story)
    del meta_data['fullSentence']
    meta_data.dropna(inplace=True)
    mapping_column = 'shiftBOLD_{}sec'.format(bold_shift_seconds)
    timepoints = meta_data[mapping_column].values.astype(int)
    # filter and annotate
    assert all(timepoint in timepoint_rdms['timepoint_left'].values for timepoint in timepoints)
    timepoint_rdms = timepoint_rdms.sel(timepoint_left=timepoints, timepoint_right=timepoints)
    # re-interpret timepoints as stimuli
    coords = {}
    for coord_name, coord_value in timepoint_rdms.coords.items():
        dims = timepoint_rdms.coords[coord_name].dims
        dims = [dim if not dim.startswith('timepoint') else 'stimulus' for dim in dims]
        coords[coord_name] = dims, coord_value.values
    coords = {**coords, **{'stimulus_sentence': ('stimulus', meta_data['reducedSentence'].values)}}
    dims = [dim if not dim.startswith('timepoint') else 'stimulus' for dim in timepoint_rdms.dims]
    data = DataAssembly(timepoint_rdms, coords=coords, dims=dims)
    return data


def load_sentences_meta(story):
    filepath = neural_data_dir / 'Stories_old' / 'meta' \
               / f'story{NaturalisticStories.story_item_mapping[story]}_{story}_sentencesByTR.csv'
    _logger.debug("Loading meta {}".format(filepath))
    meta_data = pd.read_csv(filepath)
    return meta_data


@cache()
def load_rdm_timepoints(story='Boar', roi_filter='from90to100'):
    data = []
    data_paths = glob.glob(neural_data_dir / 'Stories_old' / f'{story}_{roi_filter}*.csv')
    for i, filepath in enumerate(data_paths):
        _logger.debug("Loading file {} ({}/{})".format(filepath, i, len(data_paths)))
        basename = os.path.basename(filepath)
        attributes = re.match('^(?P<story>.*)_from(?P<roi_low>[0-9]+)to(?P<roi_high>[0-9]+)'
                              '(_(?P<subjects>[0-9]+)Subjects)?\.mat_r(?P<region>[0-9]+).csv', basename)
        assert attributes is not None, f"file {basename} did not match regex"
        # load values (pandas is much faster than np.loadtxt https://stackoverflow.com/a/18260092/2225200)
        region_data = pd.read_csv(filepath, header=None).values
        assert len(region_data.shape) == 2  # (subjects x stimuli) x stimuli
        num_stimuli = region_data.shape[1]
        assert len(region_data) % num_stimuli == 0
        num_subjects = len(region_data) // num_stimuli
        region_data = np.stack([region_data[(subject * num_stimuli):((subject + 1) * num_stimuli)]
                                for subject in range(num_subjects)])  # subjects x time x time
        region_data = [region_data]  # region x subjects x time x time
        region_data = xr.DataArray(region_data, coords={
            'timepoint_left': list(range(num_stimuli)), 'timepoint_right': list(range(num_stimuli)),
            'region': [int(attributes['region'])],
            'subject': list(range(num_subjects))},
                                   dims=['region', 'subject', 'timepoint_left', 'timepoint_right'])
        stimuli_meta = lambda x: (('timepoint_left', 'timepoint_right'), np.broadcast_to(x, [num_stimuli, num_stimuli]))
        region_data['story'] = stimuli_meta(attributes['story'])
        region_data['roi_low'] = stimuli_meta(int(attributes['roi_low']))
        region_data['roi_high'] = stimuli_meta(int(attributes['roi_high']))
        data.append(region_data)
    data = xr.concat(data, 'region')
    return data


if __name__ == '__main__':
    fire.Fire()
