import logging
import os
import sys
import xarray as xr

from brainscore.utils import LazyLoad
from neural_nlp import benchmark_pool

_logger = logging.getLogger(__name__)


def main():
    for assembly_identifier in ['Pereira2018', 'Fedorenko2016v3', 'Fedorenko2016v3nonlang', 'Blank2014fROI']:
        for i, metric_identifier in enumerate(['encoding', 'rdm']):
            if assembly_identifier == 'Fedorenko2016v3nonlang' and metric_identifier == 'rdm':  # never used
                continue
            identifier = f"{assembly_identifier}-{metric_identifier}"
            benchmark = benchmark_pool[identifier]
            if i == 0:
                assembly = benchmark._target_assembly
                assembly._ensure_loaded()  # force LazyLoad to retrieve the content
                assembly = assembly.content
                _store_s3(assembly, key=assembly_identifier)
            ceiling = benchmark.ceiling
            _store_s3(ceiling, key=identifier + '-ceiling')


def _store_s3(assembly, key):
    _logger.info(f"Storing into {key}: {type(assembly)}")
    # we'll actually just create a local file and then upload by hand
    basename = os.path.join(os.path.dirname(__file__), key)
    # make sure we can reconstruct the class
    while isinstance(assembly, LazyLoad):
        assembly._ensure_loaded()  # force LazyLoad to retrieve the content
        assembly = assembly.content
    assembly.attrs['class_module'] = assembly.__module__
    assembly.attrs['class_name'] = type(assembly).__name__
    # deal with stimulus_set: upload separately, replace the attribute with s3 link, add name to assembly attributes
    if 'stimulus_set' in assembly.attrs:
        assembly.attrs['stimulus_set_name'] = assembly.attrs['stimulus_set'].name
        stimulus_set_key = key + '-stimulus_set.csv'
        assembly.attrs['stimulus_set'].to_csv(os.path.join(os.path.dirname(__file__), stimulus_set_key), index=False)
        assembly.attrs['stimulus_set'] = 's3:' + stimulus_set_key
    # deal with nested xarrays
    for attr, value in assembly.attrs.items():
        if not isinstance(value, xr.DataArray):
            continue
        # single-scalar e.g. endpoint_x: just remove xarray packaging
        if value.ndim == 0:
            assembly.attrs[attr] = value.values.tolist()
        # actual xarrays e.g. bootstrapped_params and raw: store as their own xarrays with s3 link
        else:
            attr_key = key + '-' + attr
            _store_s3(value, key=attr_key)
            assembly.attrs[attr] = 's3:' + attr_key + '.nc'
    # write to netcdf
    assembly = xr.DataArray(assembly)
    for index in assembly.indexes.keys():
        assembly = assembly.reset_index(index)
    assembly.to_netcdf(basename + '.nc')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main()
