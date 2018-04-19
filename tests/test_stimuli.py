from neural_nlp.stimuli import load_stimuli


class TestDiverseSentences:
    def test_1(self):
        data = load_stimuli('diverse.1')
        assert len(data) == 384
        assert data[0] == 'An accordion is a portable musical instrument with two keyboards.'
        assert data[-1] == 'A woman has different reproductive organs than a man.'

    def test_2(self):
        data = load_stimuli('diverse.2')
        assert len(data) == 243
        assert data[0] == 'Beekeeping encourages the conservation of local habitats.'
        assert data[-1] == "This has led to more injuries, particularly to ligaments in the skier's knee."


class TestNaturalisticStories:
    def test_boar(self):
        data = load_stimuli('naturalistic.Boar')
        assert len(data) == 47
        assert data[0] == 'If you were to journey to the North of England, ' \
                          'you would come to a valley that is surrounded by moors as high as mountains.'
        assert data[-1] == "His fame was indeed assured, but it was not nearly as lasting " \
                           "as that of the fearsome Bradford Boar."


class TestNaturalisticStoriesNeural:
    def test_boar_full(self):
        data = load_stimuli('naturalistic-neural-full.Boar')
        assert len(data) == 23
        assert data[0] == 'That all mill owners were generally busy as beavers and quite pleased with themselves for ' \
                          'being so successful and well off was known to the residents of Bradford and if you were to ' \
                          'go into the city to visit the stately City Hall you would see there the Crest of the City ' \
                          'of Bradford which those same mill owners created to celebrate their achievements.'
        assert data[-1] == "His fame was indeed assured but it was not nearly as lasting " \
                           "as that of the fearsome Bradford Boar."

    def test_boar_reduced(self):
        data = load_stimuli('naturalistic-neural-reduced.Boar')
        assert len(data) == 47
        assert data[0] == 'If you were to journey to the North of England you would ' \
                          'come to a valley that is surrounded by moors as high as'
        assert data[-1] == "His fame was indeed assured but it was not nearly as lasting " \
                           "as that of the fearsome Bradford Boar."

    def test_load_all_full(self):
        self._test_load_all(reduced=False)

    def test_load_all_reduced(self):
        self._test_load_all(reduced=True)

    def _test_load_all(self, reduced=False):
        for story in ['Boar', 'MatchstickSeller', 'KingOfBirds', 'Elvis', 'HighSchool']:
            data = load_stimuli('naturalistic-neural-{}.{}'.format('reduced' if reduced else 'full', story))
            assert data is not None
            assert len(data) > 0
