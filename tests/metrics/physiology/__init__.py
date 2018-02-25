import argparse
import logging
import sys

_logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    _logger.info("Running with args %s", vars(args))


if __name__ == '__main__':
    main()
