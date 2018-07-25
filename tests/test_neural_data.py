from neural_nlp import neural_data
import numpy as np


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
