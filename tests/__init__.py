import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)-15s %(levelname)s:%(name)s:%(message)s')

for ignore_logger in ['botocore', 'boto3', 'urllib3', 's3transfer']:
    logging.getLogger(ignore_logger).setLevel(logging.INFO)
