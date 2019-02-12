import argparse
import grp
import threading
import logging
import time
import os

from google.protobuf import text_format

import scheduler
import worker_interface
from utils import logger
from web import web_interface as wi
from protos import scheduler_config_pb2

# Parse CL arguments
parser = argparse.ArgumentParser()
parser.add_argument('logdir',
                    help="Directory where all log files should be stored")
parser.add_argument('config',
                    help="Config file for the scheduler")
parser.add_argument('--public', action='store_true',
                    help='Whether to publish web server to the local network')

args = parser.parse_args()


def load_config(config_path):
  scheduler_config = scheduler_config_pb2.SchedulerConfig()
  with open(config_path, 'r') as f:
    text_format.Merge(f.read(), scheduler_config)

  return scheduler_config


def run():
  if not os.path.exists(args.logdir):
    raise ValueError("Invalid logdir.")
  if not os.path.exists(args.config):
    raise ValueError("Invalid config file.")

  logger.init_logger(args.logdir)

  config = load_config(args.config)

  num_devices_per_worker = config.num_devices_per_worker
  # In hours, 0 for no limit
  experiment_time_limit = config.experiment_time_limit
  initial_tf_port = config.initial_tf_port
  # We could run up to one tf server per device on one worker, so we need to
  # have that many ports
  tf_ports = list(range(initial_tf_port,
                        initial_tf_port + num_devices_per_worker))
  hosts = config.host_addresses

  user_name_list = []
  group_database = grp.getgrnam('researchers')

  for user in group_database.gr_mem:
    user_name_list.append(user)

  workers = dict()
  for host in hosts:
    workers[host] = worker_interface.WorkerInterface(
      host=host, tf_ports=tf_ports, num_devices=num_devices_per_worker,
      logdir=args.logdir)

  experiment_scheduler = scheduler.Scheduler(
    workers=workers, user_name_list=user_name_list,
    logdir=args.logdir, experiment_time_limit=experiment_time_limit,
    reorganize_experiments_interval=config.reorganize_experiments_interval)

  web_interface = wi.WebInterface(scheduler_ref=experiment_scheduler)

  public = args.public
  # Start web server thread
  web_thread = threading.Thread(target=web_interface.run, args=(public,))
  web_thread.daemon = True
  web_thread.start()
  logging.info('Web Interface Thread started.')

  # Specify updates per second
  ups = config.ups
  frame_time = 1.0 / ups
  t0 = time.time()
  t_accumulated = 0
  while True:
    t1 = time.time()
    t_accumulated += (t1 - t0)
    t0 = t1

    while t_accumulated > frame_time:
      t_accumulated -= frame_time
      experiment_scheduler.update()

    time.sleep(frame_time - t_accumulated)


if __name__ == '__main__':
  run()
