import numpy as np
import os

import seaborn
from matplotlib import pyplot

from neural_nlp.benchmarks.neural import StoriesVoxelEncoding, PereiraEncoding
import fire

seaborn.set()


def subject_stimuli():
    benchmark = PereiraEncoding()
    data = benchmark._target_assembly

    def apply(group, axis=1):
        return ~np.isnan(group).any(axis=axis)

    data = data.groupby('subject').reduce(apply, dim='neuroid')
    data = data.transpose('presentation', 'subject')
    fig, ax = pyplot.subplots(figsize=(3, 21))
    ax.imshow(data)
    ax.set_xlabel('subject')
    ax.set_ylabel('presentation')
    savepath = os.path.join(os.path.dirname(__file__), 'subject_stimuli.png')
    pyplot.savefig(savepath)
    print(f"Saved to {savepath}")


def subject_voxels():
    benchmark = StoriesVoxelEncoding()
    data = benchmark._target_assembly
    subject_voxels = {}
    for subject in set(data['subject_UID'].values):
        subject_assembly = data.sel(subject_UID=subject)
        subject_voxels[subject] = len(subject_assembly['neuroid'])
    fig, ax = pyplot.subplots()
    ax.bar(subject_voxels.keys(), subject_voxels.values())
    ax.set_xlabel('subject')
    ax.set_ylabel('number of used neurons')
    savepath = os.path.join(os.path.dirname(__file__), 'subject_voxels.png')
    pyplot.savefig(savepath)
    print(f"Saved to {savepath}")


if __name__ == '__main__':
    fire.Fire()
