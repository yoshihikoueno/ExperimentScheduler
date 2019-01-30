import queue
import logging
import datetime

import numpy as np

from task import TaskType


class Scheduler:
  def __init__(self, workers, user_name_list):
    # List of pending experiments is handled as a queue
    self.pending_experiments = []
    # Maps experiment_id to experiment
    self.active_experiments = dict()
    # Maps experiment_id to experiment
    self.finished_experiments = dict()

    self.user_name_list = user_name_list
    self.workers = workers

    # The task queue storing tasks dedicated for the scheduler,
    # coming from the web interface
    self.task_queue = queue.Queue()

    # Maps experiment_id to tuple (worker_to_device_indices)
    self._active_experiment_clusters = dict()

  def get_device_states(self):
    device_states = dict()
    for worker in self.workers:
      device_states[worker.host] = worker.device_states

    return device_states

  def update(self):
    self._handle_web_interface_tasks()

    self._handle_finished_experiments()

    self._try_start_experiments()

  def _handle_web_interface_tasks(self):
    # Handle all tasks that came from the web interface
    tasks = []
    while not self.task_queue.empty():
      tasks.append(self.task_queue.get())

    for task in tasks:
      if task.task_type == TaskType.NEW_EXPERIMENT:
        experiment = task.kvargs['experiment']
        logging.info("Experiment '{}' of user '{}' queued.".format(
          experiment.name, experiment.user_name))
        self.pending_experiments.append(experiment)

  def _handle_finished_experiments(self):
    # Check if experiments have finished
    for worker in self.workers.values():
      finished_experiments = worker.poll_experiments()
      for finished_experiment_id, return_code in finished_experiments:
        experiment = self.active_experiments[finished_experiment_id]
        experiment.finish_time = datetime.datetime.now()
        experiment.finish_return_code = return_code

        worker_to_device_indices = self._active_experiment_clusters[
          finished_experiment_id]
        worker_hosts = list(worker_to_device_indices.keys())

        # Stop chief
        self.workers[worker_hosts[0]].stop_experiment(
          finished_experiment_id, reason='')

        if experiment.framework == 'tensorflow':
          # Stop tf servers
          for worker_host in worker_hosts[1:]:
            self.workers[worker_host].stop_tf_server(finished_experiment_id)
            logging.debug(
              'Stopped tf server of experiment {} of user {}'.format(
                experiment.name, experiment.user_name))

        assert(finished_experiment_id) not in self.finished_experiments
        self.finished_experiments[finished_experiment_id] = experiment
        del self.active_experiments[finished_experiment_id]
        del self._active_experiment_clusters[finished_experiment_id]

        logging.info("Experiment '{}' of user {} finished.".format(
          experiment.name, experiment.user_name))

  def _try_start_experiments(self):
    while (len(self.pending_experiments) > 0
           and self._can_accomodate(self.pending_experiments[0])):
      experiment = self.pending_experiments.pop(0)
      worker_to_device_indices = self._assign_devices(experiment)
      worker_hosts = list(worker_to_device_indices.keys())
      devices = list(worker_to_device_indices.values())

      if experiment.framework == 'tensorflow':
        # Build cluster config
        chief_host = worker_hosts[0] + ':{}'.format(
          self.workers[worker_hosts[0]].get_free_port())
        slave_hosts = [worker_host + ':{}'.format(
          self.workers[worker_host].get_free_port())
                       for worker_host in worker_hosts[1:]]
        cluster_config = {'chief': [chief_host], 'worker': slave_hosts}

        # Start slave servers
        for i, worker_host in enumerate(worker_hosts[1:]):
          device_indices = devices[i + 1]
          task_config = {'type': 'worker', 'index': i}
          tf_config_env = {'cluster': cluster_config, 'task': task_config}
          self.workers[worker_host].start_tf_server(
            experiment.unique_id, device_indices, tf_config_env)
          logging.debug(
              'Started tf server of experiment {} of user {}'.format(
                experiment.name, experiment.user_name))

        # Start experiment
        task_config = {'type': 'chief', 'index': 0}
        tf_config_env = {'cluster': cluster_config, 'task': task_config}
        self.workers[worker_hosts[0]].start_experiment(
          experiment, devices[0], tf_config_env)

      else:
        # Chainer
        self.workers[worker_hosts[0]].start_experiment(
          experiment, devices[0], None)

      experiment.start_time = datetime.datetime.now()
      assert(experiment.unique_id not in self.active_experiments)
      self.active_experiments[experiment.unique_id] = experiment
      assert(experiment.unique_id not in self._active_experiment_clusters)
      self._active_experiment_clusters[experiment.unique_id] = (
        worker_to_device_indices)
      logging.info("Experiment '{}' of user '{}' started.".format(
          experiment.name, experiment.user_name))

  # Returns dict mapping worker to device indices to be used by the
  # experiment
  def _assign_devices(self, experiment):
    free_device_indices = self._get_free_device_indices()

    if len(free_device_indices) == 0:
      return False, free_device_indices

    result_dict = dict()
    if experiment.gpu_settings == 'forcesinglegpu':
      # Take worker with fewest available devices
      suitable_worker = None
      num_devices = 99999
      for worker, device_indices in free_device_indices.items():
        if len(device_indices) < num_devices:
          suitable_worker = worker
          num_devices = len(device_indices)

      result_dict[suitable_worker] = [free_device_indices[suitable_worker][0]]
    else:
      if experiment.use_multiple_workers:
        if experiment.gpu_settings == 'forcemultigpu':
          # Make sure we either have more than one worker, or if it is only one
          # atleast two devices on that worker
          if (len(free_device_indices) < 2
              and len(free_device_indices.values()[0]) < 2):
            return False, dict()

        result_dict = free_device_indices
      else:
        suitable_worker = None
        num_devices = 0
        # Find worker with most available devices
        for worker, device_indices in free_device_indices.items():
          if len(device_indices) > num_devices:
            suitable_worker = worker
            num_devices = len(device_indices)

        if experiment.gpu_settings == 'forcemultigpu':
          # Make sure we have at least two devices
          if num_devices < 2:
            return False, dict()

        result_dict[suitable_worker] = free_device_indices[suitable_worker]

    return result_dict

  def _can_accomodate(self, experiment):
    free_device_indices = list(self._get_free_device_indices().values())

    total_len = 0
    for indices in free_device_indices:
      total_len += len(indices)

    if experiment.gpu_settings == 'forcesinglegpu':
      return total_len >= 1
    elif experiment.gpu_settings == 'forcemultigpu':
      return total_len > 1
    else:
      return total_len >= 1

  def _get_free_device_indices(self):
    result = dict()

    for worker in self.workers.values():
      free_indices = np.where(worker.device_states)[0]
      result[worker.host] = free_indices

    return result
