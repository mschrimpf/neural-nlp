import operator
import os
import warnings
from collections import namedtuple, defaultdict

import fire
import logging
import numpy as np
import pandas as pd
import re
import scipy.io
import xarray as xr
from brainio_base.assemblies import merge_data_arrays, DataAssembly, gather_indexes, walk_coords, array_is_element, \
    NeuroidAssembly
from nltk_contrib.textgrid import TextGrid
from pathlib import Path
from tqdm import tqdm
from xarray import DataArray

from brainscore.metrics import Score
from brainscore.metrics.regression import CrossRegressedCorrelation, linear_regression
from neural_nlp.stimuli import NaturalisticStories, StimulusSet
from neural_nlp.utils import ordered_set, is_sorted
from result_caching import cache, store, store_netcdf

neural_data_dir = (Path(os.path.dirname(__file__)) / '..' / '..' / 'ressources' / 'neural_data' / 'fmri').resolve()
_logger = logging.getLogger(__name__)


@store()
def load_Pereira2018_Blank_languageresiduals():
    # hijack the corresponding encoding benchmark to regress, but then store residuals instead of correlate
    from neural_nlp.benchmarks.neural import PereiraEncoding
    benchmark = PereiraEncoding()
    assembly, cross = benchmark._target_assembly, benchmark._cross
    residuals = []

    def store_residuals(nonlanguage_prediction, language_target):
        residual = language_target - nonlanguage_prediction
        residuals.append(residual)
        return Score([0], coords={'neuroid_id': ('neuroid', [0])}, dims=['neuroid'])  # dummy score

    pseudo_metric = CrossRegressedCorrelation(
        regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
        correlation=store_residuals,
        crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))

    # separate language from non-language networks
    language_assembly = assembly[{'neuroid': [atlas in ['DMN', 'MD', 'language']
                                              for atlas in assembly['atlas'].values]}]
    nonlanguage_assembly = assembly[{'neuroid': [atlas in ['visual', 'auditory']
                                                 for atlas in assembly['atlas'].values]}]

    # run
    def apply_cross(source_assembly, target_assembly):
        # filter experiment
        source_assembly = source_assembly[{'presentation': [stimulus_id in target_assembly['stimulus_id'].values
                                                            for stimulus_id in source_assembly['stimulus_id'].values]}]
        assert all(source_assembly['stimulus_id'].values == target_assembly['stimulus_id'].values)
        # filter subjects that have not done this experiment
        source_assembly = source_assembly.dropna('neuroid')
        # for the target assembly, it's going to become awkward if we just drop those neuroids.
        # instead, we set them to zero which makes for simple zero regression weights.
        target_assembly = target_assembly.fillna(0)
        # this will regress from joint visual+auditory neural space to one of the language networks
        return pseudo_metric(source_assembly, target_assembly)

    cross(language_assembly, apply=lambda cross_assembly: apply_cross(nonlanguage_assembly, cross_assembly))

    # combine residuals
    assert len(residuals) == 5 * 2 * 3  # 5-fold CV, 2 experiments, 3 language brain networks
    # ensure uniqueness
    neuroid_ids, stimulus_ids = [], []
    for residual in residuals:
        neuroid_ids += residual['neuroid_id'].values.tolist()
        stimulus_ids += residual['stimulus_id'].values.tolist()
    assert len(neuroid_ids) == len(language_assembly['neuroid']) * 5 * 2
    assert len(set(neuroid_ids)) == len(set(language_assembly['neuroid_id'].values))
    assert len(stimulus_ids) == len(language_assembly['presentation']) * 3
    assert len(set(stimulus_ids)) == len(set(language_assembly['stimulus_id'].values))
    residuals = merge_data_arrays(residuals)
    residuals = type(language_assembly)(residuals)
    residuals.attrs['stimulus_set'] = assembly.stimulus_set
    return residuals


@store()
def load_Pereira2018_Blank(version='base'):
    reference_data = load_Pereira2018()

    data_dir = neural_data_dir / ("Pereira2018_Blank" + ("_langonly" if version != 'base' else ""))
    experiments = {'n72': "243sentences", 'n96': "384sentences"}
    assemblies = []
    subjects = ['018', '199', '215', '288', '289', '296', '343', '366', '407', '426']
    for subject in tqdm(subjects, desc="subjects"):
        subject_assemblies = []
        for experiment_filepart, experiment_name in experiments.items():
            filepath = data_dir / f"{'ICA_' if version != 'base' else ''}" \
                                  f"{subject}_complang_passages_{experiment_filepart}_persent.mat"
            if not filepath.is_file():
                _logger.debug(f"Subject {subject} did not run {experiment_name}: {filepath} does not exist")
                continue
            data = scipy.io.loadmat(str(filepath))
            if version != 'base':
                data = data['x'][0, 0]

            # construct assembly
            assembly = data['data' if version == 'base' else f'data{version}']
            neuroid_meta = data['meta']

            expanded_assembly = []
            voxel_nums, atlases, filter_strategies, atlas_selections, atlas_filter_lower, rois = [], [], [], [], [], []
            for voxel_num in range(assembly.shape[1]):
                for atlas_iter, atlas_selection in enumerate(neuroid_meta['atlases'][0, 0][:, 0]):
                    multimask = neuroid_meta['roiMultimask'][0, 0][atlas_iter, 0][voxel_num, 0]
                    if np.isnan(multimask):
                        continue
                    atlas_selection = atlas_selection[0].split('_')
                    filter_strategy = None if len(atlas_selection) != 3 else atlas_selection[1]
                    filter_lower = re.match(r'from([0-9]{2})to100prcnt', atlas_selection[-1])
                    atlas_filter_lower.append(int(filter_lower.group(1)))
                    atlas, selection = atlas_selection[0], atlas_selection[-1]
                    atlases.append(atlas)
                    filter_strategies.append(filter_strategy)
                    atlas_selections.append(selection)
                    multimask = int(multimask) - 1  # Matlab 1-based to Python 0-based indexing
                    rois.append(neuroid_meta['rois'][0, 0][atlas_iter, 0][multimask, 0][0])
                    voxel_nums.append(voxel_num)
                    expanded_assembly.append(assembly[:, voxel_num])
            # ensure all are mapped
            assert set(voxel_nums) == set(range(assembly.shape[1])), "not all voxels mapped"
            # add indices
            assembly = np.stack(expanded_assembly).T
            assert assembly.shape[1] == len(atlases) == len(atlas_selections) == len(rois)
            indices_in_3d = neuroid_meta['indicesIn3D'][0, 0][:, 0]
            indices_in_3d = [indices_in_3d[voxel_num] for voxel_num in voxel_nums]
            # add coords
            col_to_coords = np.array([neuroid_meta['colToCoord'][0, 0][voxel_num] for voxel_num in voxel_nums])

            # put it all together
            assembly = NeuroidAssembly(assembly, coords={
                **{coord: (dims, value) for coord, dims, value in walk_coords(
                    reference_data.sel(experiment=experiment_name)['presentation'])},
                **{'experiment': ('presentation', [experiment_name] * assembly.shape[0]),
                   'subject': ('neuroid', [subject] * assembly.shape[1]),
                   'voxel_num': ('neuroid', voxel_nums),
                   'atlas': ('neuroid', atlases),
                   'filter_strategy': ('neuroid', filter_strategies),
                   'atlas_selection': ('neuroid', atlas_selections),
                   'atlas_selection_lower': ('neuroid', atlas_filter_lower),
                   'roi': ('neuroid', rois),
                   'indices_in_3d': ('neuroid', indices_in_3d),
                   'col_to_coord_1': ('neuroid', col_to_coords[:, 0]),
                   'col_to_coord_2': ('neuroid', col_to_coords[:, 1]),
                   'col_to_coord_3': ('neuroid', col_to_coords[:, 2]),
                   }}, dims=['presentation', 'neuroid'])
            assembly['neuroid_id'] = 'neuroid', _build_id(assembly, ['subject', 'voxel_num'])
            subject_assemblies.append(assembly)
        assembly = merge_data_arrays(subject_assemblies)
        assemblies.append(assembly)

    _logger.debug(f"Merging {len(assemblies)} assemblies")
    assembly = merge_data_arrays(assemblies)
    assembly.attrs['version'] = version

    _logger.debug("Creating StimulusSet")
    assembly.attrs['stimulus_set'] = reference_data.stimulus_set
    return assembly


@store()
def load_Pereira2018():
    data_dir = neural_data_dir / "Pereira2018"
    experiment2, experiment3 = "243sentences.mat", "384sentences.mat"
    stimuli = {}  # experiment -> stimuli
    assemblies = []
    subject_directories = [d for d in data_dir.iterdir() if d.is_dir()]
    for subject_directory in tqdm(subject_directories, desc="subjects"):
        for experiment_filename in [experiment2, experiment3]:
            data_file = subject_directory / f"examples_{experiment_filename}"
            if not data_file.is_file():
                _logger.debug(f"{subject_directory} does not contain {experiment_filename}")
                continue
            data = scipy.io.loadmat(str(data_file))

            # assembly
            assembly = data['examples']
            meta = data['meta']
            assembly = NeuroidAssembly(assembly, coords={
                'experiment': ('presentation', [os.path.splitext(experiment_filename)[0]] * assembly.shape[0]),
                'stimulus_num': ('presentation', list(range(assembly.shape[0]))),
                'passage_index': ('presentation', data['labelsPassageForEachSentence'][:, 0]),
                'passage_label': ('presentation', [data['keyPassages'][index - 1, 0][0]
                                                   for index in data['labelsPassageForEachSentence'][:, 0]]),
                'passage_category': ('presentation', [
                    data['keyPassageCategory'][0, data['labelsPassageCategory'][index - 1, 0] - 1][0][0]
                    for index in data['labelsPassageForEachSentence']]),

                'subject': ('neuroid', [subject_directory.name] * assembly.shape[1]),
                'voxel_num': ('neuroid', list(range(assembly.shape[1]))),
                'AAL_roi_index': ('neuroid', meta[0][0]['roiMultimaskAAL'][:, 0]),
            }, dims=['presentation', 'neuroid'])
            stimulus_id = _build_id(assembly, ['experiment', 'stimulus_num'])
            assembly['stimulus_id'] = 'presentation', stimulus_id
            # set story for compatibility
            assembly['story'] = 'presentation', _build_id(assembly, ['experiment', 'passage_category'])
            assembly['neuroid_id'] = 'neuroid', _build_id(assembly, ['subject', 'voxel_num'])
            assemblies.append(assembly)

            # stimuli
            if experiment_filename not in stimuli:
                sentences = data['keySentences']
                sentences = [sentence[0] for sentence in sentences[:, 0]]
                stimuli[experiment_filename] = {
                    'sentence': sentences,
                    'sentence_num': list(range(len(sentences))),
                    'stimulus_id': stimulus_id,
                    'experiment': assembly['experiment'].values,
                    'story': assembly['story'].values,
                }
                for copy_coord in ['experiment', 'story', 'passage_index', 'passage_label', 'passage_category']:
                    stimuli[experiment_filename][copy_coord] = assembly[copy_coord].values

    _logger.debug(f"Merging {len(assemblies)} assemblies")
    assembly = merge_data_arrays(assemblies)

    _logger.debug("Creating StimulusSet")
    combined_stimuli = {}
    for key in stimuli[experiment2]:
        combined_stimuli[key] = np.concatenate((stimuli[experiment2][key], stimuli[experiment3][key]))
    stimuli = StimulusSet(combined_stimuli)
    stimuli.name = "Pereira2018"
    assembly.attrs['stimulus_set'] = stimuli
    return assembly


def _build_id(assembly, coords):
    return [".".join([f"{value}" for value in values]) for values in zip(*[assembly[coord].values for coord in coords])]


def load_voxels(bold_shift_seconds=4):
    assembly = load_voxel_data(bold_shift_seconds=bold_shift_seconds)
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
        if subject_meta['UID'] not in ['088', '085', '098', '061', '090']:
            continue
        f = scipy.io.loadmat(filepath)
        file_data = f['data']
        regions = list(file_data.dtype.fields)
        for region in regions:
            if region not in ['language']:
                continue
            region_data = file_data[region][0, 0][0, 0]
            thresholds = list(region_data.dtype.fields)
            for threshold in thresholds:
                if threshold not in ['from90to100']:
                    continue
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

    ignored_words = [None, '', '<s>', '</s>', '<s']
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
        timepoints = [timepoint - bold_shift_seconds for timepoint in timepoints]
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
            # quick-fixes
            if story == 'Boar' and sentence == 'interactions the the':  # Boar duplicate
                sentence = 'interactions the'
            if story == 'KingOfBirds' and sentence == 'the fact that the larger':  # missing word in TextGrid
                sentence = 'earth ' + sentence
            if story == 'MrSticky' and sentence == 'worry don\'t worry i went extra slowly since it\'s':
                sentence = 'don\'t worry i went extra slowly since it\'s'
            sentences.append(sentence)
            last_timepoint = timebin_meta['time_end'].values[-1]
        sentence_index = [i for i, sentence in enumerate(sentences) if sentence]
        sentences = np.array(sentences)[sentence_index]
        if story not in ['Boar', 'KingOfBirds', 'MrSticky']:  # ignore quick-fixes
            annotated_sentence = ' '.join(sentences)
            meta_sentence = ' '.join(word.strip() for word in story_meta.values if word not in ignored_words) \
                .lower().strip()
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


compare_characters = [',', '"', '\'', ':', '.', '!', '?', '(', ')']


def compare_ignore(sentence):
    for compare_character in compare_characters:
        sentence = sentence.replace(compare_character, '')
    sentence = sentence.replace('-', ' ')
    sentence = sentence.lower()
    return sentence


def _align_stimuli_recordings(stimulus_set, assembly):
    aligned_stimulus_set = []
    partial_sentences = assembly['stimulus_sentence'].values
    partial_sentences = [compare_ignore(sentence) for sentence in partial_sentences]
    assembly_stimset = {}
    stimulus_set_index = 0

    stories = ordered_set(assembly['story'].values.tolist())
    for story in tqdm(sorted(stories), desc='align stimuli', total=len(stories)):
        story_partial_sentences = [(sentence, i) for i, (sentence, sentence_story) in enumerate(zip(
            partial_sentences, assembly['story'].values)) if sentence_story == story]

        story_stimuli = stimulus_set[stimulus_set['story'] == story]
        stimuli_story = ' '.join(story_stimuli['sentence'])
        stimuli_story_sentence_starts = [0] + [len(sentence) + 1 for sentence in story_stimuli['sentence']]
        stimuli_story_sentence_starts = np.cumsum(stimuli_story_sentence_starts)
        assert ' '.join(s for s, i in story_partial_sentences) == compare_ignore(stimuli_story)
        stimulus_index = 0
        Stimulus = namedtuple('Stimulus', ['story', 'sentence', 'sentence_num', 'sentence_part'])
        sentence_parts = defaultdict(lambda: 0)
        for partial_sentence, assembly_index in story_partial_sentences:
            full_partial_sentence = ''
            partial_sentence_index = 0
            while partial_sentence_index < len(partial_sentence) \
                    or stimulus_index < len(stimuli_story) \
                    and stimuli_story[stimulus_index] in compare_characters + [' ']:
                if partial_sentence_index < len(partial_sentence) \
                        and partial_sentence[partial_sentence_index].lower() \
                        == stimuli_story[stimulus_index].lower():
                    full_partial_sentence += stimuli_story[stimulus_index]
                    stimulus_index += 1
                    partial_sentence_index += 1
                elif stimuli_story[stimulus_index] in compare_characters + [' ']:
                    # this case leads to a potential issue: Beginning quotations ' are always appended to
                    # the current instead of the next sentence. For now, I'm hoping this won't lead to issues.
                    full_partial_sentence += stimuli_story[stimulus_index]
                    stimulus_index += 1
                elif stimuli_story[stimulus_index] == '-':
                    full_partial_sentence += '-'
                    stimulus_index += 1
                    if partial_sentence[partial_sentence_index] == ' ':
                        partial_sentence_index += 1
                else:
                    raise NotImplementedError()
            sentence_num = next(index for index, start in enumerate(stimuli_story_sentence_starts)
                                if start >= stimulus_index) - 1
            sentence_part = sentence_parts[sentence_num]
            sentence_parts[sentence_num] += 1
            row = Stimulus(sentence=full_partial_sentence, story=story,
                           sentence_num=sentence_num, sentence_part=sentence_part)
            aligned_stimulus_set.append(row)
            assembly_stimset[assembly_index] = stimulus_set_index
            stimulus_set_index += 1
        # check
        aligned_story = "".join(row.sentence for row in aligned_stimulus_set if row.story == story)
        assert aligned_story == stimuli_story
    # build StimulusSet
    aligned_stimulus_set = StimulusSet(aligned_stimulus_set)
    aligned_stimulus_set['stimulus_id'] = [".".join([str(value) for value in values]) for values in zip(*[
        aligned_stimulus_set[coord].values for coord in ['story', 'sentence_num', 'sentence_part']])]
    aligned_stimulus_set.name = f"{stimulus_set.name}-aligned"

    # align assembly
    alignment = [stimset_idx for assembly_idx, stimset_idx in
                 sorted(assembly_stimset.items(), key=operator.itemgetter(0))]
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
        dims = [dim if not dim.startswith('timepoint') else 'presentation' for dim in dims]
        coords[coord_name] = dims, coord_value.values
    coords = {**coords, **{'stimulus_sentence': ('presentation', meta_data['reducedSentence'].values)}}
    dims = [dim if not dim.startswith('timepoint') else 'presentation' for dim in timepoint_rdms.dims]
    data = DataAssembly(timepoint_rdms, coords=coords, dims=dims)
    return data


def load_sentences_meta(story):
    filepath = neural_data_dir / 'Stories_RDMs' / 'meta' \
               / f'story{NaturalisticStories.story_item_mapping[story]}_{story}_sentencesByTR.csv'
    _logger.debug("Loading meta {}".format(filepath))
    meta_data = pd.read_csv(filepath)
    return meta_data


@cache()
def load_rdm_timepoints(story='Boar', roi_filter='from90to100'):
    data = []
    data_paths = list((neural_data_dir / 'Stories_RDMs').glob(f'{story}_{roi_filter}*.csv'))
    for i, filepath in enumerate(data_paths):
        _logger.debug(f"Loading file {filepath} ({i}/{len(data_paths)})")
        basename = os.path.basename(filepath)
        attributes = re.match('^(?P<story>.*)_from(?P<roi_low>[0-9]+)to(?P<roi_high>[0-9]+)'
                              '(_(?P<subjects>[0-9]+)Subjects)?\.mat_r(?P<region>[0-9]+).csv', basename)
        if attributes is None:
            warnings.warn(f"file {basename} did not match regex -- ignoring")
            continue
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
