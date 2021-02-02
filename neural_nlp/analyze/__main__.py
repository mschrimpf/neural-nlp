import logging
import matplotlib
import seaborn
import sys

from neural_nlp.analyze import scores, stats, data
from neural_nlp.analyze.data import ceiling
from neural_nlp.analyze.scores import bars, layers

_logger = logging.getLogger(__name__)


def paper_figures():
    seaborn.set(context='talk')
    seaborn.set_style("whitegrid", {'axes.grid': False})
    matplotlib.rc('axes', edgecolor='black')
    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.spines.top'] = False
    matplotlib.rcParams['svg.fonttype'] = 'none'  # export SVG as text, not path

    annotated_models = ['ETM', 'glove', 'skip-thoughts', 'lm_1b', 'transformer',
                        'bert-base-uncased', 'bert-large-uncased', 'roberta-large',
                        'xlm-mlm-en-2048', 'xlnet-large-cased',
                        't5-small', 'albert-xxlarge-v2', 'ctrl',
                        'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']

    neural_data_identifiers = ['Pereira2018', 'Fedorenko2016v3', 'Blank2014fROI']
    neural_benchmarks = [f'{prefix}-encoding' for prefix in neural_data_identifiers]
    brain_data_identifiers = neural_data_identifiers + ['Futrell2018']
    brain_benchmarks = [f'{prefix}-encoding' for prefix in brain_data_identifiers]
    wiki_color = '#0035ff'

    # 2a: per benchmark scores -- we can predict
    _logger.info("Figures 2a")
    for i, benchmark in enumerate(neural_benchmarks):
        bars.whole_best(benchmark=benchmark, annotate=i == 0)
    # 2b: scores are correlated
    _logger.info("Figures 2b")
    settings = dict(best_layer=True, plot_correlation=False, plot_significance_stars=False)
    scores.Pereira2018_experiment_correlations(**settings)  # within Pereira
    for comparison_benchmark in neural_benchmarks[1:]:
        scores.compare(benchmark1=neural_benchmarks[0], benchmark2=comparison_benchmark,
                       plot_ceiling=False, identity_line=True, **settings)
    # 3: LM predicts brain, but other tasks don't
    _logger.info("Figures 3")
    settings = dict(best_layer=True, plot_ceiling=False, plot_significance_stars=False, identity_line=False)
    scores.compare(benchmark1='wikitext-2', benchmark2='overall_neural-encoding', **settings,
                   annotate=annotated_models)
    bars.predictor('wikitext-2', neural_benchmarks, ylim=[0, .65], color=wiki_color)
    scores.compare(benchmark1='overall_glue', benchmark2='overall_neural-encoding', **settings,
                   xlim=[.25, .7], xtick_locator_base=0.1, annotate=False)
    bars.predictor('overall_glue', neural_benchmarks, ylim=[0, .65])
    bars.task_predictors('overall_neural-encoding')
    scores.compare(benchmark1='wikitext-2', benchmark2='overall_neural-encoding', **settings)  # text only
    # 4: neural/LM predicts behavior
    bars.whole_best(benchmark='Futrell2018-encoding', annotate=True)
    behavior_ylim = 1.335
    settings = dict(benchmark2='Futrell2018-encoding', best_layer=True, annotate=annotated_models, plot_ceiling=False,
                    plot_significance_stars=False, ylim=[0, behavior_ylim])
    scores.compare(benchmark1='overall_neural-encoding', **settings, xlim=[0, 1])
    bars.predictor('Futrell2018-encoding', neural_benchmarks, ylim=[0, .65])
    scores.compare(benchmark1='wikitext-2', **settings, identity_line=False)
    # 5: untrained predicts trained
    _logger.info("Figures 5")
    scores.untrained_vs_trained(benchmark='overall_neural-encoding')
    bars.untrained_predictor(benchmarks=neural_benchmarks)
    # 6: overview table
    scores.compare(benchmark1='wikitext-2', benchmark2='overall_neural-encoding',
                   plot_significance_stars=False, identity_line=False)
    scores.compare(benchmark1='overall_neural-encoding', benchmark2='Futrell2018-encoding',
                   plot_significance_stars=False)
    scores.untrained_vs_trained(benchmark='overall_neural-encoding')
    bars.whole_best(benchmark='overall-encoding', annotate=True)

    # fig6 caption: untrained/trained diff
    scores.untrained_vs_trained(benchmark='overall-encoding', analyze_only=True)
    # text: untrained/trained gpt2-xl
    stats.model_training_diff(model='gpt2-xl', benchmark='overall-encoding')
    # text: untrained/trained AlBERTs Fedorenko2016
    scores.untrained_vs_trained(
        benchmark='Fedorenko2016v3-encoding', analyze_only=True,
        model_selection=[identifier for identifier in scores.models if identifier.startswith('albert')])

    # S1: ceiling extrapolation
    _logger.info("Figures S1")
    tick_formatting = {benchmark: formatting for benchmark, formatting in zip(brain_benchmarks, [1, 2, 2, 1])}
    for benchmark in brain_benchmarks:
        ceiling.plot_extrapolation_ceiling(benchmark=benchmark, ytick_formatting_frequency=tick_formatting[benchmark])
    # S2: cross-metrics
    _logger.info("Figures S2")
    scores.metric_generalizations()
    # S4b: non language signal
    _logger.info("Figures S4a")
    scores.compare(benchmark1='Fedorenko2016v3-encoding', benchmark2='Fedorenko2016v3nonlang-encoding',
                   identity_line=True, plot_ceiling=False, plot_significance_stars=False)
    # S5: predictors (individual wikitext + GLUE)
    _logger.info("Figures S5")
    settings = dict(best_layer=True, plot_ceiling=False, plot_significance_stars=False, identity_line=False)
    for neural_benchmark in neural_benchmarks:
        scores.compare(benchmark1='wikitext-2', benchmark2=neural_benchmark, **settings,
                       tick_locator_base=0.1 if neural_benchmark.startswith('Blank') else 0.2,
                       ylim=[-0.055, 0.5] if neural_benchmark.startswith('Blank') else None, annotate=None)
        scores.compare_glue(benchmark2=neural_benchmark)
    # S6: individual benchmarks predict behavior
    _logger.info("Figures S6")
    for benchmark in neural_benchmarks:
        scores.compare(benchmark1=benchmark, benchmark2='Futrell2018-encoding', best_layer=True, annotate=False,
                       plot_ceiling=False, plot_significance_stars=False, ylim=[0, behavior_ylim])
    # S7: untrained/trained per dataset
    for benchmark in brain_benchmarks:
        scores.untrained_vs_trained(benchmark=benchmark)
    # S8: untrained controls
    _logger.info("Figures S7: a) wikitext untrained; b) random embedding")
    scores.untrained_vs_trained(benchmark='wikitext-2', identity_line=False, loss_xaxis=True)
    bars.random_embedding()
    layers.layer_preference_single()
    for benchmark in neural_benchmarks:
        layers.layer_preference(benchmark=benchmark)

    # additional controls
    for benchmark in neural_benchmarks:
        data.train_test_overlap(benchmark)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    paper_figures()
