import logging
import sys


def init_logger():
  # Logging Configuration
  formatter = logging.Formatter(
    fmt='%(asctime)s [%(levelname)s] : %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')

  logging.getLogger().setLevel(logging.DEBUG)

  sh = logging.StreamHandler(sys.stdout)
  sh.setLevel(logging.DEBUG)
  sh.setFormatter(formatter)
  logging.getLogger().addHandler(sh)
