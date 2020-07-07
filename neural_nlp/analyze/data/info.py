from neural_nlp import benchmark_pool, score
from neural_nlp.analyze.data import subject_columns


def _print_score_info(benchmark_identifier, neuroid_dim='neuroid'):
    score_object = score(model='bert-base-uncased', benchmark=benchmark_identifier)
    print(
        f"  score\n"
        f"    - {len(score_object.raw[neuroid_dim])} neuroids"
    )


def _print_assembly_info(benchmark_identifier):
    benchmark = benchmark_pool[benchmark_identifier]
    assembly = benchmark._target_assembly
    subject_coord = subject_columns[benchmark_identifier]
    print(
        f"  assembly\n"
        f"    - {len(assembly['presentation'])} stimuli x {len(assembly['neuroid'])} neuroids"
        f" ({len(set(assembly['neuroid_id'].values))} unique, {len(set(assembly[subject_coord].values))} subjects)"
    )


def _print_stimulus_info(benchmark_identifier, group_column='story', group_value=None, stimulus_column='sentence'):
    benchmark = benchmark_pool[benchmark_identifier]
    stimulus_set = benchmark._target_assembly.stimulus_set
    group_value = group_value or stimulus_set[group_column].values[0]
    stimulus = stimulus_set[stimulus_set[group_column] == group_value]
    stimulus = " | ".join(stimulus[stimulus_column])
    print(
        f"    - sample stimulus: {stimulus}\n"
    )


def print_Pereira2018():
    benchmark_identifier = 'Pereira2018-encoding'
    score_object = score(model='bert-base-uncased', benchmark=benchmark_identifier)
    score_object.attrs['raw'] = score_object.raw.raw  # language only
    print(f"## {benchmark_identifier} ##")
    print(
        f"  score\n"
        f"    - {len(score_object.raw['neuroid'])} neuroids"
    )
    _print_assembly_info(benchmark_identifier=benchmark_identifier)
    _print_stimulus_info(benchmark_identifier=benchmark_identifier)

    # num voxels
    for atlas in ['MD', 'DMN']:
        benchmark = benchmark_pool[benchmark_identifier]
        assembly = benchmark._target_assembly
        atlas_assembly = assembly.sel(atlas=atlas)
        # some subjects have nans for either experiment. The sum gets rid of those
        subject_assembly = atlas_assembly.sum('presentation')
        subject_assembly = subject_assembly.groupby('subject').count('neuroid')
        mean, std = subject_assembly.mean().values, subject_assembly.std().values
        print(
            f"  {atlas}: {len(atlas_assembly['neuroid'])} voxels ({mean:.0f}+-{std:.1f})"
        )


def print_Fedorenko2016():
    benchmark_identifier = 'Fedorenko2016v3-encoding'
    print(f"## {benchmark_identifier} ##")
    _print_score_info(benchmark_identifier)
    _print_assembly_info(benchmark_identifier=benchmark_identifier)
    _print_stimulus_info(benchmark_identifier=benchmark_identifier, group_column='sentence_id', stimulus_column='word')


def print_Blank2014():
    benchmark_identifier = 'Blank2014fROI-encoding'
    print(f"## {benchmark_identifier} ##")
    _print_score_info(benchmark_identifier)
    _print_assembly_info(benchmark_identifier=benchmark_identifier)
    _print_stimulus_info(benchmark_identifier=benchmark_identifier, group_value='Boar')


def print_Futrell2018():
    benchmark_identifier = 'Futrell2018-encoding'
    print(f"## {benchmark_identifier} ##")
    _print_score_info(benchmark_identifier, neuroid_dim='subject_id')
    _print_assembly_info(benchmark_identifier=benchmark_identifier)
    _print_stimulus_info(benchmark_identifier=benchmark_identifier, stimulus_column='word',
                         group_column='story_id', group_value=1)  # use Boar


def print_all():
    print_Pereira2018()
    print_Fedorenko2016()
    print_Blank2014()
    print_Futrell2018()


if __name__ == '__main__':
    import warnings

    warnings.simplefilter(action='ignore')
    print_all()
