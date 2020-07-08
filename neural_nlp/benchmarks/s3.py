import importlib

import boto3
import os
import pandas as pd
import sys
import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

from neural_nlp.stimuli import StimulusSet
from result_caching import _Storage


class _S3Storage(_Storage):
    """
    load pre-computed xarray assemblies from S3
    """

    _NO_SIGNATURE = Config(signature_version=UNSIGNED)

    def __init__(self, *args, key, bucket='brainio-language', region='us-east-1', **kwargs):
        super(_S3Storage, self).__init__(*args, **kwargs)
        self._key = key
        self._bucket = bucket
        self._region = region
        self._local_root_dir = os.path.expanduser('~/.neural_nlp/')
        os.makedirs(self._local_root_dir, exist_ok=True)

    def is_stored(self, function_identifier):
        return True  # everything is on S3

    def load(self, function_identifier):
        assembly = self._retrieve(self._key + '.nc', self._local_root_dir)
        return assembly

    def _retrieve(self, key, dir):
        local_path = os.path.join(dir, key)
        if not os.path.isfile(local_path):
            self._download_file(key, local_path)
        if key.endswith('.csv'):  # stimulus_set is no xarray and does not need attrs dealt with
            return StimulusSet(pd.read_csv(local_path))
        assembly = xr.open_dataarray(local_path)
        # deal with nested assemblies
        for attr, value in assembly.attrs.items():
            if isinstance(value, str) and value.startswith('s3:'):
                _, attr_key = value.split('s3:')
                value = self._retrieve(attr_key, dir)
                assembly.attrs[attr] = value
                if attr == 'stimulus_set':
                    value.name = assembly.attrs['stimulus_set_name']
        # put into correct class
        cls_module = importlib.import_module(assembly.attrs['class_module'])
        cls = getattr(cls_module, assembly.attrs['class_name'])
        assembly = cls(assembly)
        return assembly

    def _download_file(self, key, local_path):
        self._logger.debug(f"Downloading {key} to {local_path}")
        s3 = boto3.resource('s3', region_name=self._region, config=self._NO_SIGNATURE)
        obj = s3.Object(self._bucket, key)
        # show progress. see https://gist.github.com/wy193777/e7607d12fad13459e8992d4f69b53586
        with tqdm(total=obj.content_length, unit='B', unit_scale=True, desc=key, file=sys.stdout) as progress_bar:
            def progress_hook(bytes_amount):
                progress_bar.update(bytes_amount)

            obj.download_file(local_path, Callback=progress_hook)

    def save(self, result, function_identifier):
        raise NotImplementedError("can only load from S3, but not save")


load_s3 = _S3Storage
