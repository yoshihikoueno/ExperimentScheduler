import argparse
import grp
import threading
import logging
import time
import os

import scheduler
import worker_interface
from utils import logger
from web import web_interface as wi

# Parse CL arguments
parser = argparse.ArgumentParser()
parser.add_argument('logdir',
                    help="Directory where all log files should be stored")
parser.add_argument('--public', action='store_true',
                    help='Whether to publish web server to the local network')

args = parser.parse_args()


def run():
  if not os.path.exists(args.logdir):
    raise ValueError("Invalid logdir.")

  logger.init_logger(args.logdir)

  num_devices_per_worker = 4
  # In hours, 0 for no limit
  experiment_time_limit = 12
  initial_tf_port = 2222
  # We could run up to one tf server per device on one worker, so we need to
  # have that many ports
  tf_ports = list(range(initial_tf_port,
                        initial_tf_port + num_devices_per_worker))
  hosts = ['127.0.0.1', '127.0.0.2', '127.0.0.3', '127.0.0.4']

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
    logdir=args.logdir, experiment_time_limit=experiment_time_limit)

  web_interface = wi.WebInterface(
    scheduler_ref=experiment_scheduler,
    num_devices_per_worker=num_devices_per_worker)

  public = args.public
  # Start web server thread
  web_thread = threading.Thread(target=web_interface.run, args=(public,))
  web_thread.daemon = True
  web_thread.start()
  logging.info('Web Interface Thread started.')

  # Specify updates per second
  ups = 0.2
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
