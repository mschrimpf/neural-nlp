import numpy as np
from brainio_base.assemblies import walk_coords
from tqdm import tqdm
from xarray import DataArray

from brainscore.metrics import Score
from brainscore.metrics.regression import pearsonr_correlation
from brainscore.metrics.transformations import apply_aggregate, subset
from neural_nlp.neural_data.fmri import load_voxels
from result_caching import store


@store()
def fROI_correlation():
    assembly = load_voxels()

    subjects = list(sorted(set(assembly['subject_UID'].values)))
    split_scores = []
    correlate = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id', neuroid_coord='fROI_area'))
    for heldout_subject in tqdm(subjects, desc='subject holdout'):
        subject_pool = list(sorted(set(subjects) - {heldout_subject}))
        indexer_pool = DataArray(np.zeros(len(subject_pool)), coords={'subject_UID': subject_pool},
                                 dims=['subject_UID']).stack(neuroid=['subject_UID'])
        heldout_indexer = DataArray(np.zeros(1), coords={'subject_UID': [heldout_subject]},
                                    dims=['subject_UID']).stack(neuroid=['subject_UID'])
        subject_pool = subset(assembly, indexer_pool, dims_must_match=False)
        subject_pool = average_subregions(subject_pool)
        heldout = subset(assembly, heldout_indexer, dims_must_match=False)
        heldout = average_subregions(heldout)
        split_score = correlate(subject_pool, heldout)
        split_score = type(split_score)(split_score.values, coords={
            coord: (dims, values) for coord, dims, values in walk_coords(split_score)
            if not coord.startswith('subject_') and coord != 'neuroid_id'}, dims=split_score.dims)

        split_score = split_score.expand_dims('heldout_subject')
        split_score['heldout_subject'] = [heldout_subject]
        split_scores.append(split_score)
    correlation = Score.merge(*split_scores)

    correlation = apply_aggregate(lambda scores: scores.median('neuroid'), correlation)
    center = correlation.mean('heldout_subject')
    error = correlation.std('heldout_subject')
    score = Score([center, error], coords={**{'aggregation': ['center', 'error']},
                                           **{coord: (dims, values) for coord, dims, values in walk_coords(center)}},
                  dims=('aggregation',) + center.dims)
    score.attrs[Score.RAW_VALUES_KEY] = correlation.attrs[Score.RAW_VALUES_KEY]
    return score


def average_subregions(assembly):
    del assembly['threshold']
    assembly = assembly.multi_dim_apply(['stimulus_id', 'fROI_area'], lambda group, **_: group.mean())
    _, index = np.unique(assembly['fROI_area'], return_index=True)
    assembly = assembly.isel(neuroid=index)
    return assembly


if __name__ == '__main__':
    r = fROI_correlation()
    print(r)
