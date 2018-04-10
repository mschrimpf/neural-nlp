from neural_nlp import neural_data
import numpy as np


class TestLoadRdms():
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
        data = neural_data.load_rdms(story)
        assert len(data['subject']) == num_subjects
        assert len(data['timepoint']) == num_timepoints
        np.testing.assert_array_equal(data['timepoint'], list(range(num_timepoints)))
        assert len(data['region']) == num_regions
