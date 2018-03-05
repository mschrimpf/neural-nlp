import logging
import os

from neural_metrics import models
logger = logging.getLogger(__name__)


class _Defaults(object):
    regions = ('V4', 'IT')


def run(model, layers, regions=_Defaults.regions,
        model_weights=models._Defaults.model_weights, save_plot=False):
    from neural_metrics.metrics import score_model_activations
    from neural_metrics.metrics.physiology import metrics_for_activations
    from neural_metrics.models import activations_for_model
    from neural_metrics.plot import plot_layer_correlations, plot_scores, results_dir

    logger.info('Computing activations')
    activations = activations_for_model(model=model, layers=layers, model_weights=model_weights)
    activations_filepath = models.get_savepath(model, model_weights)
    logger.info('Computing scores')
    scores = score_model_activations(activations, regions, basepath=activations_filepath)
    logger.info('Plotting')
    output_filepath = os.path.join(results_dir, '{}-scores-regions_{}.{}'.format(
        os.path.basename(os.path.splitext(activations_filepath)[0]), ''.join(regions), 'svg'))
    plot_scores(scores, output_filepath=output_filepath if save_plot else None)
    if save_plot:
        logger.info('Plot saved to {}'.format(output_filepath))
    return scores
