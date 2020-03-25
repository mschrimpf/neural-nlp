import numpy as np
import pytest
from pytest import approx

from neural_nlp import neural_data
from neural_nlp.neural_data.ecog import load_Fedorenko2016
from neural_nlp.neural_data.fmri import load_Pereira2018, load_Pereira2018_Blank, \
    load_Pereira2018_Blank_languageresiduals
from neural_nlp.neural_data.naturalStories import load_naturalStories


class TestPereiraLanguageResiduals:
    def test(self):
        assembly = load_Pereira2018_Blank_languageresiduals()
        assert set(assembly['atlas'].values) == {'DMN', 'MD', 'language'}
        assert set(assembly['experiment'].values) == {'384sentences', '243sentences'}
        assert len(assembly['presentation']) == 627
        assert len(set(assembly['subject'].values)) == 10


class TestLoadRdmSentences(object):
    def test_boar(self):
        self._test_story('Boar', num_sentences=23, num_subjects=39)

    def test_matchstick_seller(self):
        self._test_story('MatchstickSeller', num_sentences=41, num_subjects=15)

    def test_king_of_birds(self):
        self._test_story('KingOfBirds', num_sentences=29, num_subjects=35)

    def test_elvis(self):
        self._test_story('Elvis', num_sentences=16, num_subjects=36)

    def test_highschool(self):
        self._test_story('HighSchool', num_sentences=27, num_subjects=15)

    def _test_story(self, story, num_sentences, num_subjects, num_regions=44):
        data = neural_data.load_rdm_sentences(story)
        assert len(data['stimulus']) == num_sentences
        assert len(data['subject']) == num_subjects
        assert len(data['region']) == num_regions


class TestLoadRdmTimepoints:
    def test_boar(self):
        self._test_story('Boar', num_timepoints=169, num_subjects=39)

    def test_elvis(self):
        self._test_story('Elvis', num_timepoints=151, num_subjects=36)

    def test_highschool(self):
        self._test_story('HighSchool', num_timepoints=174, num_subjects=15)

    def test_king_of_birds(self):
        self._test_story('KingOfBirds', num_timepoints=198, num_subjects=35)

    def test_matchstick_seller(self):
        self._test_story('MatchstickSeller', num_timepoints=197, num_subjects=15)

    def _test_story(self, story, num_timepoints, num_subjects, num_regions=44):
        data = neural_data.load_rdm_timepoints(story)
        assert len(data['subject']) == num_subjects
        assert len(data['timepoint']) == num_timepoints
        np.testing.assert_array_equal(data['timepoint'], list(range(num_timepoints)))
        assert len(data['region']) == num_regions


def test_Pereira():
    assembly = load_Pereira2018()
    assert set(assembly['experiment'].values) == {'243sentences', '384sentences'}
    assert len(assembly['presentation']) == 243 + 384
    assert len(assembly['neuroid']) == 1592159
    assert len(set(assembly['subject'].values)) == 9
    subject_assembly = assembly.sel(subject='P01')
    assert not np.isnan(subject_assembly).any()
    assert subject_assembly.values.sum() == approx(101003052.8293553)


@pytest.mark.parametrize(('version', 'values_sum'), [
    ('base', -38516843.18636856),
    ('ICA', 52337776.49628149),
    ('Demean', -10854974.038396357),
    ('NoVisAud', -12719536.616299922),
])
def test_PereiraBlank(version, values_sum):
    assembly = load_Pereira2018_Blank(version=version)
    assert set(assembly['experiment'].values) == {'243sentences', '384sentences'}
    assert len(assembly['presentation']) == 243 + 384
    assert len(set(assembly['subject'].values)) == 10
    assert len(set(assembly['neuroid_id'].values)) == 310125
    assert set(assembly['atlas'].values) == {'language', 'MD', 'DMN', 'auditory', 'visual'}
    assert set(assembly['filter_strategy'].values) == {np.nan, 'NminusS', 'HminusE', 'FIXminusN', 'FIXminusH'}
    subject_assembly = assembly.sel(subject='018', atlas='auditory', atlas_selection_lower=90)
    assert not np.isnan(subject_assembly).any()  # note though that other atlases have nan values from the data itself
    assert np.nansum(assembly.values) == approx(values_sum)
    voxel = assembly.sel(subject='018', voxel_num=15 - 1)  # -1 to go from matlab 1-based indexing to python 0-based
    assert _single_element(voxel['indices_in_3d']) == 65158
    cols = [_single_element(voxel[f'col_to_coord_{i + 1}']) for i in range(3)]
    np.testing.assert_array_equal(cols, [62, 65, 9])


def _single_element(array):
    unique_values = np.unique(array)
    assert len(unique_values) == 1
    return unique_values[0]


@pytest.mark.parametrize('version, electrodes, expected_neuroids', [
    (2, 'language', 98),
    (2, 'all', 180),
    (3, 'language', 97),
    (3, 'all', 177),
    (3, 'non-language', 105),
])
def test_Fedorenko2016(version, electrodes, expected_neuroids):
    assembly = load_Fedorenko2016(version=version, electrodes=electrodes)
    assert len(assembly['presentation']) == 416
    assert len(assembly['neuroid']) == expected_neuroids
    assert len(np.unique(assembly['subject_UID'])) == 5


def test_natural_stories():
    assembly = load_naturalStories()
    assert len(assembly['word']) == 10256
    assert len(set(assembly['stimulus_id'].values)) == len(assembly['presentation'])
    assert len(set(assembly['story_id'].values)) == 10
    assert len(set(assembly['sentence_id'].values)) == 481
    assert len(set(assembly['subject_id'].values)) == 180

    stimulus_set = assembly.stimulus_set
    assert set(stimulus_set['stimulus_id'].values) == set(assembly['stimulus_id'].values)
    assert len(set(stimulus_set['story_id'].values)) == len(set(assembly['story_id'].values))
    assert len(set(stimulus_set['sentence_id'].values)) == len(set(assembly['sentence_id'].values))
    assert ' '.join(stimulus_set['word']) == ' '.join(assembly['word'].values)
    assert stimulus_set.name == 'naturalStories'

    mean_assembly = assembly.mean('subjects')
    assert not np.isnan(mean_assembly).any()
