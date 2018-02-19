import logging
import os

from neural_metrics import models
from neural_metrics.metrics import score_model_activations
from neural_metrics.metrics.physiology import metrics_for_activations
from neural_metrics.models import activations_for_model
from neural_metrics.plot import plot_layer_correlations, plot_scores, results_dir

logger = logging.getLogger(__name__)


class _Defaults(object):
    regions = ('V4', 'IT')


def run(model, layers, regions=_Defaults.regions,
        model_weights=models._Defaults.model_weights, save_plot=False):
    logger.info('Computing activations')
    activations_savepath = activations_for_model(model=model, model_weights=model_weights,
                                                 layers=layers, use_cached=True)
    logger.info('Computing scores')
    scores = score_model_activations(activations_savepath, regions, use_cached=True)
    logger.info('Plotting')
    file_name = os.path.splitext(os.path.basename(activations_savepath))[0]
    output_filepath = os.path.join(results_dir, '{}-scores-regions_{}.{}'.format(file_name, ''.join(regions), 'svg'))
    plot_scores(scores, output_filepath=output_filepath if save_plot else None)
    if save_plot:
        logger.info('Plot saved to {}'.format(output_filepath))
