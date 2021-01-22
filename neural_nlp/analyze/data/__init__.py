import numpy as np
import pandas as pd
from brainio_base.assemblies import walk_coords
from pathlib import Path

from brainscore.metrics import Score
from neural_nlp import benchmark_pool, model_pool, FixedLayer
from neural_nlp.models.implementations import ETM


class prefixdict(dict):
    def __init__(self, default=None, **kwargs):
        super(prefixdict, self).__init__(**kwargs)
        self._default = default

    def __getitem__(self, item):
        subitem = item
        while len(subitem) > 1:
            try:
                return super(prefixdict, self).__getitem__(subitem)
            except KeyError:
                subitem = subitem[:-1]
        return self._default


subject_columns = prefixdict(default='subject_id',
                             Fedorenko='subject_UID',
                             Pereira='subject',
                             Blank='subject_UID')


def print_train_test(benchmark_identifier):
    benchmark = benchmark_pool[benchmark_identifier]
    stimulus_set = benchmark._target_assembly.stimulus_set

    # idea: change the metric's apply function to output train/test
    result = []

    def output_apply(*args):
        for stimulus_type, assembly in zip(["source_train", "target_train", "source_test", "target_test"], args):
            if stimulus_type.startswith("target_"):
                # ignore duplicates from "right" side of split
                continue
            presentation_values = {
                **{"type": stimulus_type},
                **{key: value for key, dims, value in walk_coords(assembly['presentation'])},
                "stimulus_lookup": [
                    stimulus_set['sentence' if hasattr(stimulus_set, 'sentence') else 'word']
                    [stimulus_set['stimulus_id'] == stimulus_id].values
                    for stimulus_id in assembly['stimulus_id'].values]}
            if benchmark_identifier.startswith('Blank2014'):
                presentation_values["stimulus_lookup"] = [line for arr in presentation_values["stimulus_lookup"]
                                                          for line in arr]
                presentation_values = {key: str(value) for key, value in presentation_values.items()}

            result.append(presentation_values)
        return Score([np.nan],
                     coords={**{"neuroid_id": ("neuroid", [0])},
                             **{subject_column: ("neuroid", [0]) for subject_column in set(subject_columns.values())}},
                     dims=["neuroid"])

    if hasattr(benchmark, "_single_metric"):
        benchmark._single_metric.apply = output_apply
    else:
        benchmark._metric.apply = output_apply

    # run with dummy model to invoke metric
    dummy_model = model_pool['ETM']
    candidate = FixedLayer(dummy_model, ETM.available_layers[0])
    benchmark(candidate)

    # aggregate and print
    result = pd.DataFrame(result)
    # excel cannot open csvs with cells >32k
    result = result.apply(lambda x: x.str.slice(0, 32_000).replace('\n', ''))
    print(result)
    result.to_csv(Path(__file__).parent / f'splits-{benchmark_identifier}.csv')
