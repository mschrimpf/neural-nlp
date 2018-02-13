import logging
import os

from neural_metrics import models
from neural_metrics.compare import metrics_for_activations
from neural_metrics.models import activations_for_model
from neural_metrics.plot import plot_layer_correlations

logger = logging.getLogger(__name__)


def run(model, layers, regions=('V4', 'IT'),
        model_weights=models._Defaults.model_weights, concat_up_to_n_layers=1,
        save_plot=False):
    try:
        logger.debug("Attempt bypassing activations if metrics files exist")
        activations_savepath = models.get_savepath(model, model_weights)
        metrics_savepaths = [metrics_for_activations(activations_savepath, region=region, use_cached=True)
                             for region in regions]
        logger.info("Bypassed activations computation by using existing metrics files")
    except FileNotFoundError:
        logger.info('Computing activations')
        activations_savepath = activations_for_model(model=model, model_weights=model_weights,
                                                     layers=layers, use_cached=True)
        logger.info('Computing metrics')
        metrics_savepaths = [metrics_for_activations(activations_savepath, region=region,
                                                     concat_up_to_n_layers=concat_up_to_n_layers, use_cached=True)
                             for region in regions]
    logger.info('Plotting')
    file_name = os.path.splitext(os.path.basename(activations_savepath))[0]
    output_filepath = os.path.join(os.path.dirname(__file__), '..', 'results',
                                   '{}-regions_{}{}'.format(file_name, ''.join(regions), '.svg'))
    plot_layer_correlations(metrics_savepaths, labels=regions,
                            output_filepath=output_filepath if save_plot else None)
