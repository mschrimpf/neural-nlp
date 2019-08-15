import logging

import numpy as np

from neural_nlp.analyze.architecture_sampling.neural_scores import build_activations_model

_logger = logging.getLogger(__name__)


def test_sentence():
    _logger.debug("Building model")
    model = build_activations_model('/braintree/data2/active/users/msch/zoo.wmt17/'
                                    '696bc5a3b245ef2d61cf89cf77ca988b4d0e5f54', load_weights=False)
    _logger.debug("Running model")
    activations = model(['the brown fox'])
    assert not np.isnan(activations).any()
    assert not (activations.values == 0).all()
