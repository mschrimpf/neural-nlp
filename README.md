

## Installation
mkgu
1. install mkgu: `mkgu$ python setup.py install` (might have to install dependencies by hand using conda/pip).
2. as a workaround to https://github.com/dicarlolab/mkgu/issues/16, copy `mkgu/mkgu/lookup.db` to `site-packages/mkgu-0.1.0-py3.6.egg/mkgu`.
3. configure AWS credentials: using awscli `aws configure` (options e.g. region `us-east-1`, format `json`)
