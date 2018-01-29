import xarray as xr
import pandas as pd
import pickle
import argparse
import os
from neural_metrics.utils import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activations_filepath', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'images', 'sorted',
                                             'squeezenet-activations.pkl'))
    parser.add_argument('--pcs', type=int, default=200)
    args = parser.parse_args()
    print("Running with args", args)

    model_data = load_image_activations(args.activations_filepath)
    array_list = []
    img_index = pd.Index(model_data.keys(), name='img_id')

    for image_data in model_data:
        data_dict = model_data[image_data]
        data = list(data_dict.values())
        layers = list(data_dict.keys())
        princomps = range(1, args.pcs + 1)
        x_xrarray = xr.DataArray(data, coords=[layers, princomps],
                                 dims=['layer', 'princomp'])
        array_list.append(x_xrarray)

    merged = xr.concat(array_list, dim=img_index)

    save(merged, args.activations_filepath.replace('.pkl', '-xr.pkl'))


def load_image_activations(activations_filepath):
    with open(activations_filepath, 'rb') as file:
        image_activations = pickle.load(file)
    return image_activations

if __name__ == '__main__':
    main()
