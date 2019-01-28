import argparse
import grp
import threading
import logging

import scheduler
from web import web_interface as wi

# Parse CL arguments
parser = argparse.ArgumentParser()
parser.add_argument('--public', action='store_true',
                    help='Whether to publish web server to the local network')

args = parser.parse_args()


def run():
  num_devices_per_worker = 2
  workers = ['127.0.0.1']

  user_name_list = []
  group_database = grp.getgrnam('researchers')

  for user in group_database.gr_mem:
    user_name_list.append(user)

  experiment_scheduler = scheduler.Scheduler(
    num_devices_per_worker=num_devices_per_worker, workers=workers,
    user_name_list=user_name_list)

  web_interface = wi.WebInterface(scheduler_ref=experiment_scheduler)

  public = args.public
  # Start web server thread
  web_thread = threading.Thread(target=web_interface.run, args=(public,))
  web_thread.daemon = True
  web_thread.start()
  logging.info('Web Interface Thread started.')

  while True:
    experiment_scheduler.update()


if __name__ == '__main__':
  run()
