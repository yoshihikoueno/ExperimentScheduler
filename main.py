import argparse
import threading
import logging
import time
import os
import atexit
import json
from collections import namedtuple

import scheduler
import worker_interface
from utils import logger
from utils import util_ops
from web import web_interface as wi

def load_config(config_path):
    with open(config_path) as f:
        scheduler_config = json.load(f)

    scheduler_config = namedtuple(
        'config',
        scheduler_config.keys()
    )(*scheduler_config.values())

    return scheduler_config

def run(logdir, config, port, public):
    web_port = port
    logger.init_logger(logdir)

    if not os.path.exists(logdir):
        logging.warn(f'logdir {logdir} not found. creating...')
        os.makedirs(logdir)
        raise ValueError("Invalid logdir.")
    if not os.path.exists(config):
        raise ValueError("Invalid config file.")

    config = load_config(config)

    # In hours, 0 for no limit
    experiment_time_limit = config.experiment_time_limit
    initial_tf_port = config.initial_tf_port
    # We could run up to one tf server per device on one worker, so we need to
    # have that many ports
    tf_ports = list(range(initial_tf_port, initial_tf_port + 7))
    hosts = config.host_addresses

    workers = {
        host: worker_interface.WorkerInterface(
            host=host,
            tf_ports=tf_ports,
            logdir=logdir,
            resource_folder=config.resource_folder,
            docker_resource_folder=config.docker_resource_folder,
        ) for host in hosts
    }

    experiment_scheduler = scheduler.Scheduler(
        workers=workers,
        logdir=logdir,
        experiment_time_limit=experiment_time_limit,
        reorganize_experiments_interval=config.reorganize_experiments_interval,
    )

    # Register shutdown callback
    atexit.register(experiment_scheduler.shutdown)

    web_interface = wi.WebInterface(
        scheduler_ref=experiment_scheduler,
        resource_folder=config.resource_folder,
        docker_resource_folder=config.docker_resource_folder,
    )

    # Start web server thread
    web_thread = threading.Thread(
        target=web_interface.run,
        args=(public, web_port),
        daemon=True,
    )
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
    # Parse CL arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logdir',
        help="Directory where all log files should be stored",
        default='/tmp/ExperimentSchedulerTemp/log',
    )
    parser.add_argument(
        '--config',
        help="Config file for the scheduler",
        default='config.json',
    )
    parser.add_argument(
        '--public',
        action='store_true',
        help='Whether to publish web server to the local network',
    )
    parser.add_argument(
        '--port',
        help='The port to use for the web interface. Defaults to 5000',
        default=5000,
        type=int,
    )

    args = parser.parse_args()
    run(**vars(args))
