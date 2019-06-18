import glob
import logging
import operator
import os
import warnings
from collections import namedtuple, defaultdict, OrderedDict

import fire
import numpy as np
import pandas as pd
import re
import scipy.io
import xarray as xr
from brainio_base.assemblies import merge_data_arrays, DataAssembly, gather_indexes, walk_coords, array_is_element
from nltk_contrib.textgrid import TextGrid
from pathlib import Path
from tqdm import tqdm
from xarray import DataArray

from neural_nlp.stimuli import NaturalisticStories, StimulusSet
from neural_nlp.utils import ordered_set, is_sorted
from result_caching import cache, store, store_netcdf

neural_data_dir = (Path(os.path.dirname(__file__)) / '..' / '..' / 'ressources' / 'neural_data' / 'fmri').absolute()
_logger = logging.getLogger(__name__)


def load_voxels():
    assembly = load_voxel_data()
    assembly = DataAssembly(assembly)
    stimulus_set = NaturalisticStories()()
    stimulus_set, assembly = _align_stimuli_recordings(stimulus_set, assembly)
    assert set(assembly['stimulus_sentence'].values).issubset(set(stimulus_set['sentence']))
    assembly.attrs['stimulus_set'] = stimulus_set
    assembly.attrs['stimulus_set_name'] = stimulus_set.name
    return assembly


@store_netcdf()
def load_voxel_data(bold_shift_seconds=4):
    data = load_filtered_voxel_timepoints()
    gather_indexes(data)
    meta = load_time_meta()
    annotated_data = _merge_voxel_meta(data, meta, bold_shift_seconds)
    return annotated_data


@store_netcdf()
def load_filtered_voxel_timepoints():
    data = load_voxel_timepoints()
    data = data.sel(threshold='from90to100')
    gather_indexes(data)
    data = data.sel(subject_nStories=8)
    return data


fROIs = {
    'language': [
        '01_LH_PostTemp',
        '02_LH_AntTemp',
        '03_LH_IFG',
        '04_LH_IFGorb',
        '05_LH_MFG',
        '06_LH_AngG',
        '07_RH_PostTemp',
        '08_RH_AntTemp',
        '09_RH_IFG',
        '10_RH_IFGorb',
        '11_RH_MFG',
        '12_RH_AngG',
    ],
    'MD_langloc': [
        '01_LH_postParietal',
        '02_LH_midParietal',
        '03_LH_antParietal',
        '04_LH_supFrontal',
        '05_LH_Precentral_A_PrecG',
        '06_LH_Precental_B_IFGop',
        '07_LH_midFrontal',
        '08_LH_midFrontalOrb',
        '09_LH_insula',
        '10_LH_medialFrontal',
        '11_RH_postParietal',
        '12_RH_midParietal',
        '13_RH_antParietal',
        '14_RH_supFrontal',
        '15_RH_Precentral_A_PrecG',
        '16_RH_Precental_B_IFGop',
        '17_RH_midFrontal',
        '18_RH_midFrontalOrb',
        '19_RH_insula',
        '20_RH_medialFrontal',
    ],
    'DMN_langloc': [
        '01_LH_FrontalMed.img',
        '02_LH_PostCing.img',
        '03_LH_TPJ.img',
        '04_LH_MidCing.img',
        '05_LH_STGorInsula.img',
        '06_LH_AntTemp.img',
        '07_RH_FrontalMed.img',
        '08_RH_PostCing.img',
        '09_RH_TPJ.img',
        '10_RH_MidCing.img',
        '11_RH_STGorInsula.img',
        '12_RH_AntTemp.img',
    ],
    'auditory': [
        '01_LH_TE11.img',
        '02_LH_TE12.img',
        '03_RH_TE11.img',
        '04_RH_TE12.img',
    ]
}
fROIs['MD_spatWM'] = fROIs['MD_langloc']
fROIs['DMN_spatWM'] = fROIs['DMN_langloc']


@store_netcdf()
def load_voxel_timepoints():
    def _dim_coord_values(assembly):
        dim_coord_values = defaultdict(dict)
        for coord, dims, values in walk_coords(assembly):
            assert len(dims) == 1
            dim = dims[0]
            dim_coord_values[dim][coord] = values.tolist()
        return dim_coord_values

    def _dim_index(dim_coord_values):
        dim_values = {}
        for dim, coord_values in dim_coord_values.items():
            values = [dict(zip(coord_values, t)) for t in zip(*coord_values.values())]
            values = ["__".join(str(value) for value in row_dict.values()) for row_dict in values]
            dim_values[dim] = values
        return dim_values

    # 1st pass: find unique coords
    dim_coord_values = defaultdict(lambda: defaultdict(list))
    for story_data in _iterate_voxel_timepoints(desc='pass 1: coords'):
        story_dim_values = _dim_coord_values(story_data)
        for dim, dict_values in story_dim_values.items():
            for coord, values in dict_values.items():
                dim_coord_values[dim][coord] += values

    dim_index = _dim_index(dim_coord_values)
    dim_index = {dim: np.unique(values, return_index=True) for dim, values in dim_index.items()}
    coords = {coord: (dim, np.array(values)[dim_index[dim][1]])
              for dim, coord_values in dim_coord_values.items() for coord, values in coord_values.items()}

    # 2nd pass: fill coords with data
    data = np.empty([len(values) for (values, index) in dim_index.values()])
    data[:] = np.nan
    for story_data in _iterate_voxel_timepoints(desc='pass 2: data'):
        story_dim_index = _dim_index(_dim_coord_values(story_data))
        indices = {dim: np.searchsorted(dim_index[dim][0], story_dim_index[dim]) for dim in dim_index}
        indices = [indices[dim] for dim in story_data.dims]
        data[np.ix_(*indices)] = story_data.values
    data = DataArray(data, coords=coords, dims=['threshold', 'neuroid', 'timepoint'])
    data['neuroid_id'] = 'neuroid', [".".join([str(value) for value in values]) for values in zip(*[
        data[coord].values for coord in ['subject_UID', 'region', 'fROI_area', 'voxel_num']])]
    return data


def _iterate_voxel_timepoints(desc='files'):
    fixation_offset = 8
    data_dir = neural_data_dir / 'StoriesData_Dec2018'
    meta_filepath = data_dir / 'subjectsWithStoryData_andPreprocessedTimeseries_20190118.mat'
    meta = scipy.io.loadmat(meta_filepath)
    meta = meta['ssStruct']
    subject_meta = {key: [value.squeeze().tolist() for value in meta[key][0]] for key in list(meta.dtype.fields)}
    subject_meta['stories'] = [[story.squeeze().tolist() for story in stories] for stories in subject_meta['stories']]

    files = [data_dir / f"{uid}_{session_id}_preprocessed.mat" for uid, session_id in
             zip(subject_meta['UID'], subject_meta['SessionID'])]
    nonexistent_files = [file for file in files if not file.exists()]
    assert not nonexistent_files, f"Files {nonexistent_files} do not exist"
    file_subject_meta = [dict(zip(subject_meta, t)) for t in zip(*subject_meta.values())]
    for subject_meta, filepath in tqdm(zip(file_subject_meta, files), total=len(file_subject_meta), desc=desc):
        f = scipy.io.loadmat(filepath)
        file_data = f['data']
        regions = list(file_data.dtype.fields)
        for region in regions:
            region_data = file_data[region][0, 0][0, 0]
            thresholds = list(region_data.dtype.fields)
            for threshold in thresholds:
                threshold_data = region_data[threshold].squeeze()
                num_fROIs = threshold_data.shape[0]
                for fROI_index in range(num_fROIs):
                    fROI_name = fROIs[region][fROI_index]
                    timeseries = threshold_data[fROI_index]['timeseries'].squeeze()
                    if timeseries.dtype.fields is None:
                        assert np.isnan(timeseries)
                        warnings.warn(f"NaN timeseries: {filepath}, region {region}, threshold {threshold}, "
                                      f"fROI {fROI_name}/{fROI_index}")
                        continue
                    stories = list(timeseries.dtype.fields)
                    for story in stories:
                        story_data = timeseries[story].tolist()
                        subject_story_index = subject_meta['stories'].index(story)
                        num_neuroids, num_timepoints = story_data.shape[0], story_data.shape[1]

                        story_data = DataArray([story_data], coords={**{
                            'threshold': [threshold],
                            'voxel_num': ('neuroid', np.arange(0, num_neuroids)),
                            'region': ('neuroid', [region] * num_neuroids),
                            'fROI_area': ('neuroid', [fROI_name] * num_neuroids),
                            'fROI_index': ('neuroid', [fROI_index] * num_neuroids),
                            'timepoint_value': ('timepoint', np.arange(
                                2 - fixation_offset, 2 + num_timepoints * 2 - fixation_offset, 2)),  # 2s snapshots
                            'story': ('timepoint', [story] * num_timepoints),
                        }, **{
                            f"subject_{key}": ('neuroid', [value] * num_neuroids)
                            for key, value in subject_meta.items()
                            if key not in ['stories', 'storiesComprehensionScores', 'storiesComprehensionUnanswered']
                        },  # **{
                                                                     # f"story_comprehension_score": (('neuroid', 'timepoint'), np.tile(
                                                                     # subject_meta['storiesComprehensionScores'][subject_story_index], story_data.shape)),
                                                                     #         f"story_comprehension_unanswered": (('neuroid', 'timepoint'), np.tile(bool(
                                                                     #             subject_meta['storiesComprehensionUnanswered'][subject_story_index])
                                                                     #                                                           , story_data.shape))
                                                                     #         }
                                                                     }, dims=['threshold', 'neuroid', 'timepoint'])
                        yield story_data


def load_time_meta():
    data_dir = neural_data_dir / 'StoriesData_Dec2018' / 'stories_textgridsbyJeanne'
    files = data_dir.glob("*TextGrid*")
    time_to_words = []
    for file in files:
        textgrid = TextGrid.load(file)
        words = [tier for tier in textgrid.tiers if tier.nameid == 'words'][0]
        rows = defaultdict(list)
        for (time_start, time_end, word) in words.simple_transcript:
            rows['time_start'].append(float(time_start))
            rows['time_end'].append(float(time_end))
            rows['word'].append(word)
        story_index = int(file.stem)
        story = stories_meta.sel(number=story_index).values
        story = next(iter(set(story)))  # Boar was read twice
        rows = DataArray(rows['word'],
                         coords={'filepath': ('time_bin', [file.name] * len(rows['word'])),
                                 'story': ('time_bin', [story] * len(rows['word'])),
                                 'time_start': ('time_bin', rows['time_start']),
                                 'time_end': ('time_bin', rows['time_end']),
                                 },
                         dims=['time_bin'])
        gather_indexes(rows)
        time_to_words.append(rows)
    time_to_words = merge_data_arrays(time_to_words)
    return time_to_words


stories_meta = ['Boar', 'Aqua', 'MatchstickSeller', 'KingOfBirds', 'Elvis', 'MrSticky',
                'HighSchool', 'Roswell', 'Tulips', 'Tourette', 'Boar']
stories_meta = DataArray(stories_meta, coords={
    'story_name': ('story', stories_meta),
    'number': ('story', list(range(1, 11)) + [1]),
    'reader': ('story', ['Ted', 'Ted', 'Nancy', 'Nancy', 'Ted', 'Nancy', 'Nancy', 'Ted', 'Ted', 'Nancy', 'Terri']),
    'time_with_fixation': ('story', [338, 318, 394, 396, 302, 410, 348, 394, 388, 422, 338]),
    'time_without_fixation': ('story', [5 * 60 + 6, 4 * 60 + 46, 6 * 60 + 2, 6 * 60 + 4, 4 * 60 + 30,
                                        6 * 60 + 18, 5 * 60 + 16, 6 * 60 + 2, 5 * 60 + 56, 6 * 60 + 30, 5 * 60 + 6]),
    'recording_timepoints': ('story', [169, 159, 197, 198, 151, 205, 174, 197, 194, 211, 169])
}, dims=['story'])
stories_meta['story_index'] = 'story', [".".join([str(value) for value in values]) for values in zip(*[
    stories_meta[coord].values for coord in ['story_name', 'reader']])]
gather_indexes(stories_meta)


def _merge_voxel_meta(data, meta, bold_shift_seconds):
    data_missing = set(meta['story'].values) - set(data['story'].values)
    if data_missing:
        warnings.warn(f"Stories missing from the data: {data_missing}")
    meta_missing = set(data['story'].values) - set(meta['story'].values)
    if meta_missing:
        warnings.warn(f"Stories missing from the meta: {meta_missing}")

    ignored_words = [None, '', '<s>', '</s>']
    annotated_data = []
    for story in tqdm(ordered_set(data['story'].values), desc='merge meta'):
        if story not in meta['story'].values:
            continue
        story_meta = meta.sel(story=story)
        story_meta = story_meta.sortby('time_end')

        story_data = data.sel(story=story).stack(timepoint=['timepoint_value'])
        story_data = story_data.sortby('timepoint_value')
        timepoints = story_data['timepoint_value'].values.tolist()
        assert is_sorted(timepoints)
        timepoints = [timepoint + bold_shift_seconds for timepoint in timepoints]
        sentences = []
        last_timepoint = -np.inf
        for timepoint in timepoints:
            if last_timepoint >= max(story_meta['time_end'].values):
                break
            if timepoint <= 0:
                sentences.append(None)
                continue  # ignore fixation period
            timebin_meta = [last_timepoint < end <= timepoint for end in story_meta['time_end'].values]
            timebin_meta = story_meta[{'time_bin': timebin_meta}]
            sentence = ' '.join(word.strip() for word in timebin_meta.values if word not in ignored_words)
            sentence = sentence.lower().strip()
            sentences.append(sentence)
            last_timepoint = timebin_meta['time_end'].values[-1]
        sentence_index = [i for i, sentence in enumerate(sentences) if sentence]
        sentences = np.array(sentences)[sentence_index]
        annotated_sentence = ' '.join(sentence for sentence in sentences)
        meta_sentence = ' '.join(
            word.strip() for word in story_meta.values if word not in ignored_words).lower().strip()
        assert annotated_sentence == meta_sentence
        # re-interpret timepoints as stimuli
        coords = {}
        for coord_name, dims, coord_value in walk_coords(story_data):
            dims = [dim if not dim.startswith('timepoint') else 'presentation' for dim in dims]
            # discard the timepoints for which the stimulus did not change (empty word)
            coord_value = coord_value if not array_is_element(dims, 'presentation') else coord_value[sentence_index]
            coords[coord_name] = dims, coord_value
        coords = {**coords, **{'stimulus_sentence': ('presentation', sentences)}}
        story_data = story_data[{dim: slice(None) if dim != 'timepoint' else sentence_index
                                 for dim in story_data.dims}]
        dims = [dim if not dim.startswith('timepoint') else 'presentation' for dim in story_data.dims]
        story_data = xr.DataArray(story_data.values, coords=coords, dims=dims)
        story_data['story'] = 'presentation', [story] * len(story_data['presentation'])
        gather_indexes(story_data)
        annotated_data.append(story_data)
    annotated_data = merge_data_arrays(annotated_data)
    return annotated_data


def compare_ignore(sentence):
    return sentence.replace(',', '').replace('"', '').replace('\'', '') \
        .replace('.', '').replace('!', '').replace('?', '').replace('-', ' ') \
        .lower()


def _align_stimuli_recordings(stimulus_set, assembly):
    aligned_stimulus_set = []
    partial_sentences = assembly['stimulus_sentence'].values
    partial_sentences = [compare_ignore(sentence) for sentence in partial_sentences]
    assembly_stimset = {}
    stimuli_idx = 0

    stories = ordered_set(assembly['story'].values.tolist())
    for story in tqdm(sorted(stories), desc='align stimuli', total=len(stories)):
        story_partial_sentences = OrderedDict((sentence, i) for i, (sentence, sentence_story) in enumerate(zip(
            partial_sentences, assembly['story'].values)) if sentence_story == story)
        sentence_part = None

        def append_row(row):
            nonlocal sentence_part, stimuli_idx
            row = row._replace(sentence=row.sentence.strip())
            row_ctr = namedtuple(type(row).__name__, row._fields + ('sentence_part',))
            row = row_ctr(**{**row._asdict(), **dict(sentence_part=sentence_part)})
            aligned_stimulus_set.append(row)
            sentence_part += 1
            stimuli_idx += 1

        story_stimuli = stimulus_set[stimulus_set['story'] == story]
        assert ' '.join(story_partial_sentences) == compare_ignore(' '.join(story_stimuli['sentence']))
        for row in story_stimuli.itertuples(index=False, name='Stimulus'):
            sentence_part = 0
            full_sentence = row.sentence
            # TODO: the following entirely discards ",' etc.
            full_sentence = compare_ignore(full_sentence)
            partial_sentence_idx = [index for partial_sentence, index in story_partial_sentences.items()
                                    if partial_sentence in full_sentence]
            if len(partial_sentence_idx) == 0:
                warnings.warn(f"Sentence {full_sentence} not found in partial sentences")
                row = row._replace(sentence=full_sentence)
                append_row(row)
                continue
            assert len(partial_sentence_idx) == 1
            partial_sentence_idx = partial_sentence_idx[0]
            assert partial_sentence_idx not in assembly_stimset
            assembly_stimset[partial_sentence_idx] = stimuli_idx
            partial_sentence = partial_sentences[partial_sentence_idx]
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
    # build StimulusSet
    aligned_stimulus_set = StimulusSet(aligned_stimulus_set)
    aligned_stimulus_set['stimulus_id'] = [".".join([str(value) for value in values]) for values in zip(*[
        aligned_stimulus_set[coord].values for coord in ['story', 'sentence_num', 'sentence_part']])]
    aligned_stimulus_set.name = f"{stimulus_set.name}-aligned"

    # align assembly
    alignment = [stimset_idx for assembly_idx, stimset_idx in (
        sorted(assembly_stimset.items(), key=operator.itemgetter(0)))]
    assembly_coords = {**{coord: (dims, values) for coord, dims, values in walk_coords(assembly)},
                       **{'stimulus_id': ('presentation', aligned_stimulus_set['stimulus_id'].values[alignment]),
                          'meta_sentence': ('presentation', assembly['stimulus_sentence'].values),
                          'stimulus_sentence': ('presentation', aligned_stimulus_set['sentence'].values[alignment])}}
    assembly = type(assembly)(assembly.values, coords=assembly_coords, dims=assembly.dims)

    return aligned_stimulus_set, assembly


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
