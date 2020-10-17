import queue
import logging
import datetime
import os
import atexit
import pickle

import numpy as np

from task import TaskType


class Scheduler:
    def __init__(
            self,
            workers,
            logdir,
            experiment_time_limit,
            auto_save_sessions=True,
    ):
        assert experiment_time_limit >= 1

        # List of pending experiments is handled as a queue. Contains Experiment
        # objects
        self.pending_experiments = []

        # Maps experiment_id to experiment
        self.active_experiments = dict()

        # Maps experiment_id to experiment
        self.finished_experiments = dict()

        # Dict mapping worker host to WorkerInterface
        self.workers = workers

        # The task queue storing tasks dedicated for the scheduler,
        # coming from the web interface
        self.task_queue = queue.Queue()

        # Maps experiment_id to tuple (worker_to_num_devices)
        self._active_experiment_clusters = dict()

        self._logdir = logdir

        # In hours, 0 means no limit
        self._experiment_time_limit = experiment_time_limit

        self.auto_save_sessions = auto_save_sessions

    @property
    def experiment_time_limit(self):
        return self._experiment_time_limit

    def get_experiment_stdout_path(self, experiment_id):
        return os.path.join(self._logdir, f'{experiment_id}_stdout')

    def get_experiment_stderr_path(self, experiment_id):
        return os.path.join(self._logdir, f'{experiment_id}_stderr')

    def get_session_path(self):
        return os.path.join(self._logdir, f'session.pkl')

    def get_experiment(self, experiment_id):
        for experiment in self.pending_experiments:
            if experiment.unique_id == experiment_id:
                return experiment
        return dict(
            **self.finished_experiments,
            **self.active_experiments,
        ).get(experiment_id)

    def save_session(self):
        with open(self.get_session_path(), 'wb') as f:
            pickle.dump(dict(finished_experiments=self.finished_experiments), f)
        return

    def load_session(self):
        if not os.path.exists(self.get_session_path()): return
        with open(self.get_session_path(), 'rb') as f:
            data = pickle.load(f)
        self.finished_experiments = data['finished_experiments']
        return

    def get_device_states(self):
        device_states = {
            worker.host: worker.device_states
            for worker in self.workers.values()
        }

        return device_states

    def update(self):
        self._handle_web_interface_tasks()
        self._handle_finished_experiments()
        self._stop_invalid_experiments()
        self._try_start_experiments()

        if self.auto_save_sessions:
            self.save_session()

    def shutdown(self):
        logging.info("Shutting Down.")
        for worker in self.workers.values():
            worker.shutdown()

    def _handle_web_interface_tasks(self):
        # Handle all tasks that came from the web interface
        tasks = []
        while not self.task_queue.empty():
            tasks.append(self.task_queue.get())

        for task in tasks:
            if task.task_type == TaskType.NEW_EXPERIMENT:
                new_experiment = task.kvargs['experiment']
                self.pending_experiments.append(new_experiment)
                logging.info(f"Experiment '{new_experiment.name}' of user '{new_experiment.user_name}' queued.")

            elif task.task_type == TaskType.STOP_EXPERIMENT:
                experiment_id = task.kvargs['experiment_id']
                stop_experiment = False
                # Check if in pending queue
                for i, experiment in enumerate(self.pending_experiments):
                    if experiment.unique_id == experiment_id:
                        logging.info("Stop request from host '{}' for experiment '{}' "
                                     "from user '{}'".format(task.kvargs['host'],
                                                             experiment.name,
                                                             experiment.user_name))
                        del self.pending_experiments[i]
                        stop_experiment = True
                        break

                if stop_experiment is True:
                    continue

                # Check if active
                if experiment_id in self._active_experiment_clusters:
                    experiment = self.active_experiments[experiment_id]
                    logging.info("Stop request from host '{}' for experiment '{}' "
                                 " from user '{}'".format(task.kvargs['host'],
                                                          experiment.name,
                                                          experiment.user_name))
                    self._stop_experiment(experiment_id, 'Killed: Web request')
            else:
                logging.error(
                    "Task {} not implemented!".format(task.task_type))

    def _handle_finished_experiments(self):
        # Check if experiments have finished
        for worker in self.workers.values():
            finished_experiments = worker.poll_experiments()
            for finished_experiment_id, return_code in finished_experiments:
                self._stop_experiment(
                    finished_experiment_id, reason=return_code)

    def _stop_invalid_experiments(self):
        invalid_experiments = self._get_invalid_experiments()
        for experiment_id, reason in invalid_experiments:
            self._stop_experiment(experiment_id, reason)

    def _try_start_experiments(self):
        index = 0
        while self.pending_experiments and len(self.pending_experiments) > index:
            free_worker_devices = self._get_free_devices()
            can_fit, assigned_devices = _can_fit(
                self.pending_experiments[index],
                free_worker_devices=free_worker_devices,
            )
            if not can_fit:
                # determine which hosts may accept this experiment in the future
                # and prevent that host being used by the other pending experiments
                can_be_run_on = self.pending_experiments[index].can_be_run_on
                if not self.pending_experiments[index].use_multiple_workers:
                    reserved_workers = can_be_run_on & {
                        worker_id for worker_id, worker in self.workers.items()
                        if len(worker.device_states) >= self.pending_experiments[index].gpu_settings
                    }
                    for worker in reserved_workers:
                        free_worker_devices.pop(worker)
                index += 1
                continue
            self._start_experiment(
                self.pending_experiments.pop(index),
                worker_to_devices=assigned_devices,
                is_restart=False,
            )

    def _start_experiment(
            self,
            experiment,
            worker_to_devices,
            is_restart,
    ):
        worker_hosts = list(worker_to_devices.keys())
        num_devices_list = list(worker_to_devices.values())

        # Build and start with cluster config (only necessary if more
        # than one worker)
        if experiment.framework == 'tensorflow' and len(worker_hosts) > 1:
            chief_host = worker_hosts[0] + f':{self.workers[worker_hosts[0]].get_free_port()}'
            slave_hosts = [
                worker_host + f':{self.workers[worker_host].get_free_port()}'
                for worker_host in worker_hosts[1:]
            ]
            cluster_config = {'chief': [chief_host]}
            if slave_hosts:
                cluster_config['worker'] = slave_hosts

            # Start slave servers
            for i, worker_host in enumerate(worker_hosts[1:]):
                num_devices = num_devices_list[i + 1]
                task_config = {'type': 'worker', 'index': i}
                tf_config_env = {
                    'cluster': cluster_config, 'task': task_config}
                self.workers[worker_host].start_tf_server(
                    experiment.unique_id, num_devices=num_devices,
                    tf_config_env=tf_config_env)

            # Start experiment
            task_config = {'type': 'chief', 'index': 0}
            tf_config_env = {'cluster': cluster_config, 'task': task_config}
            self.workers[worker_hosts[0]].start_experiment(
                experiment, num_devices=num_devices_list[0],
                tf_config_env=tf_config_env, is_restart=is_restart)
        else:
            # Other
            self.workers[worker_hosts[0]].start_experiment(
                experiment, num_devices=num_devices_list[0], tf_config_env=None,
                is_restart=is_restart)

        if is_restart is False:
            experiment.start_time = datetime.datetime.now()
            logging.info(f"Experiment '{experiment.name}' of user '{experiment.user_name}' started.")
        assert experiment.unique_id not in self.active_experiments
        self.active_experiments[experiment.unique_id] = experiment
        assert experiment.unique_id not in self._active_experiment_clusters
        self._active_experiment_clusters[experiment.unique_id] = worker_to_devices

    def _stop_experiment(self, experiment_id, reason, pending_restart=False):
        experiment = self.active_experiments[experiment_id]

        if pending_restart is False:
            experiment.finish_time = datetime.datetime.now()
            experiment.finish_return_code = reason

        worker_hosts = list(self._active_experiment_clusters[
            experiment_id].keys())

        # Stop chief
        self.workers[worker_hosts[0]].stop_experiment(
            experiment_id, reason=reason)

        if experiment.framework == 'tensorflow':
            # Stop tf servers
            for worker_host in worker_hosts[1:]:
                self.workers[worker_host].stop_tf_server(experiment_id)

        if pending_restart is False:
            assert(experiment_id) not in self.finished_experiments
            self.finished_experiments[experiment_id] = experiment
            logging.info("Experiment '{}' of user '{}' stopped.".format(
                experiment.name, experiment.user_name))

        del self.active_experiments[experiment_id]
        del self._active_experiment_clusters[experiment_id]

    def _get_invalid_experiments(self):
        '''
        Returns a list of tuples of (experiment_id, reason))
        '''
        result = []
        for experiment_id, experiment in self.active_experiments.items():
            # Check if timeout
            if (self._experiment_time_limit > 0 and (
                datetime.datetime.now() - experiment.start_time).total_seconds()
                    / 3600.0 > self._experiment_time_limit):
                result.append((experiment_id, 'Timeout'))

        return result

    def _get_free_devices(self):
        result = dict()

        for worker in self.workers.values():
            free_indices = np.where(worker.device_states)[0]
            result[worker.host] = len(free_indices)

        return result


def _can_fit(experiment, free_worker_devices):
    '''
    Checks if the experiment can fit in free_worker_devices
    and returns a tuple consisting of a boolean indicating this result and
    the assigned worker to devices map
    '''
    total_len = sum(free_worker_devices.values())

    if total_len == 0:
        return False, {}

    if experiment.use_multiple_workers:
        return True, free_worker_devices

    # Select worker with fewest devices available
    suitable_worker = None
    num_min_devices = 99999
    for worker, num_devices in free_worker_devices.items():
        if worker not in experiment.can_be_run_on: continue
        if experiment.gpu_settings <= num_devices < num_min_devices:
            suitable_worker = worker
            num_min_devices = num_devices

    if suitable_worker is None: return False, {}
    return True, {suitable_worker: experiment.gpu_settings}
