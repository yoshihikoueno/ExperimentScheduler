import logging
import sys
import datetime
import os


def init_logger(folder):
  # Logging Configuration
  formatter = logging.Formatter(
    fmt='%(asctime)s [%(levelname)s] : %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')

  logging.getLogger().setLevel(logging.DEBUG)

  sh = logging.StreamHandler(sys.stdout)
  sh.setLevel(logging.DEBUG)
 # sh.setFormatter(formatter)
  logging.getLogger().addHandler(sh)

  fh = logging.FileHandler(os.path.join(folder, 'scheduler_log_{}'.format(
    datetime.datetime.now())), mode='w')
  fh.setLevel(logging.DEBUG)
  #fh.setFormatter(formatter)
  logging.getLogger().addHandler(fh)
