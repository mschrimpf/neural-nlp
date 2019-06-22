import os

import seaborn
from matplotlib import pyplot

from neural_nlp import VoxelBenchmark

seaborn.set()


def main():
    benchmark = VoxelBenchmark()
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
    main()
