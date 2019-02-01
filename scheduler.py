import queue
import logging
import datetime
import os
import itertools
import atexit

import numpy as np

from task import TaskType


class Scheduler:
  def __init__(self, workers, user_name_list, logdir, experiment_time_limit):
    assert(experiment_time_limit >= 0)
    # List of pending experiments is handled as a queue
    self.pending_experiments = []
    # Maps experiment_id to experiment
    self.active_experiments = dict()
    # Maps experiment_id to experiment
    self.finished_experiments = dict()

    self.user_name_list = user_name_list
    # Dict mapping worker host to WorkerInterface
    self.workers = workers

    # The task queue storing tasks dedicated for the scheduler,
    # coming from the web interface
    self.task_queue = queue.Queue()

    # Maps experiment_id to tuple (worker_to_device_indices)
    self._active_experiment_clusters = dict()

    self._logdir = logdir
    # In hours, 0 means no limit
    self._experiment_time_limit = experiment_time_limit

    atexit.register(self.shutdown)

  def get_experiment_stdout_path(self, experiment_id):
    return os.path.join(self._logdir, '{}_stdout'.format(experiment_id))

  def get_experiment_stderr_path(self, experiment_id):
    return os.path.join(self._logdir, '{}_stderr'.format(experiment_id))

  def get_device_states(self):
    device_states = dict()
    for worker in self.workers.values():
      device_states[worker.host] = worker.device_states

    return device_states

  def update(self):
    self._handle_web_interface_tasks()

    self._handle_finished_experiments()

    self._try_start_experiments()

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
        experiment = task.kvargs['experiment']
        logging.info("Experiment '{}' of user '{}' queued.".format(
          experiment.name, experiment.user_name))
        self.pending_experiments.append(experiment)
      elif task.task_type == TaskType.STOP_EXPERIMENT:
        experiment_id = int(task.kvargs['experiment_id'])
        if experiment_id in self._active_experiment_clusters:
          experiment = self.active_experiments[experiment_id]
          logging.info(
            "Stop request from host '{}' for experiment '{}' from user '{}'".format(
              task.kvargs['host'], experiment.name, experiment.user_name))
          self._stop_experiment(experiment_id, 'Killed: Web request')
        else:
          logging.info('ID not there')

      else:
        logging.error("Task {} not implemented!".format(task.task_type))

  def _handle_finished_experiments(self):
    # Check if experiments have finished
    for worker in self.workers.values():
      finished_experiments = worker.poll_experiments()
      for finished_experiment_id, return_code in finished_experiments:
        self._stop_experiment(finished_experiment_id, reason=return_code)

  def _try_start_experiments(self):
    while (len(self.pending_experiments) > 0):
      can_accomodate, worker_to_device_indices, stop_experiment_ids = (
        self._can_accomodate(self.pending_experiments[0]))

      if can_accomodate is False:
        return

      # Stop necessary experiments
      for experiment_id, reason in stop_experiment_ids:
        self._stop_experiment(experiment_id, reason)

      experiment = self.pending_experiments.pop(0)
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

  def _stop_experiment(self, experiment_id, reason):
    experiment = self.active_experiments[experiment_id]
    experiment.finish_time = datetime.datetime.now()
    experiment.finish_return_code = reason

    worker_to_device_indices = self._active_experiment_clusters[
      experiment_id]
    worker_hosts = list(worker_to_device_indices.keys())

    # Stop chief
    self.workers[worker_hosts[0]].stop_experiment(
      experiment_id, reason=reason)

    if experiment.framework == 'tensorflow':
      # Stop tf servers
      for worker_host in worker_hosts[1:]:
        self.workers[worker_host].stop_tf_server(experiment_id)
        logging.debug(
          'Stopped tf server of experiment {} of user {}'.format(
            experiment.name, experiment.user_name))

    assert(experiment_id) not in self.finished_experiments
    self.finished_experiments[experiment_id] = experiment
    del self.active_experiments[experiment_id]
    del self._active_experiment_clusters[experiment_id]

    logging.info("Experiment '{}' of user {} finished.".format(
      experiment.name, experiment.user_name))

  # Checks whether an experiment can be accomodated.
  # Returns tuple of boolean indicating the results,
  # a map indicating the assigned devices, and a list of tuples of
  # (experiment id, reason) that need to be stopped
  def _can_accomodate(self, experiment):
    free_device_indices = self._get_free_device_indices()

    can_fit, assigned_worker_devices = self._can_fit(
      experiment, free_device_indices)

    if can_fit is True:
      return True, assigned_worker_devices, []

    # Check if we can stop some invalid experiments to make enough room
    invalid_experiments = self._get_invalid_experiments()

    # We want to stop as few as possible experiments, so we need to search
    # beginning from removing single experiments
    num_remove = 1
    for num_remove in range(1, len(invalid_experiments) + 1):
      remove_permutations = self._get_remove_permutations(
        list(range(len(invalid_experiments))), num_remove)
      for remove_indices in remove_permutations:
        # Assuming we remove the experiments at remove_indices, will we have
        # enough room for our new experiment?
        new_free_device_indices = dict(free_device_indices)
        for remove_id, reason in remove_experiments:
          additional_device_indices = self._active_experiment_clusters[
            remove_id]
          # We need to merge both dicts
          for k, v in additional_device_indices.items():
            if k in new_free_device_indices:
              new_free_device_indices[k] += v
            else:
              new_free_device_indices[k] = v

        can_fit, worker_to_device_indices = self._can_fit(
          experiment, new_free_device_indices)
        if can_fit is True:
          # We have found a suitable remove combination!
          return (True, worker_to_device_indices, remove_experiments)

    return (False, {}, [])

  # Assuming we want to remove 2 elements from four removable indices.
  # From each permutation, we then want to take the first 2 indices,
  # And only take the uniques (also counting (1,2) and (2,1) as the same)
  def _get_remove_permutations(indices, num_removes):
    perms = list(itertools.permutations(indices))

    # Slice and sort permutations
    sliced_perms = []
    for perm in perms:
      sliced_perm = perm[0:num_removes + 1]
      sliced_perm.sort()
      # We need to make a tuple as a set cannot work on a list
      sliced_perms.append(tuple(sliced_perm))

    # Remove duplicates
    sliced_perms = list(set(sliced_perms))

    return sliced_perms

  # Checks if the experiment can fit in free_worker_devices
  # and returns a tuple consisting of a boolean indicating this result and
  # the assigned worker to devices map
  def _can_fit(self, experiment, free_worker_devices):
    total_len = 0
    for worker, device_indices in free_worker_devices.items():
      total_len += len(device_indices)

    if total_len == 0:
      return False, {}

    if experiment.use_multiple_workers:
      if experiment.gpu_settings == 'forcemultigpu':
        if total_len > 1:
          return True, free_worker_devices
        else:
          return False, {}
      else:
        if total_len > 0:
          return True, free_worker_devices
        else:
          return False, {}
    else:
      if experiment.gpu_settings == 'forcesinglegpu':
        # Select worker with fewest devices available
        suitable_worker = None
        num_devices = 99999
        for worker, device_indices in free_worker_devices.items():
          if len(device_indices) < num_devices:
            suitable_worker = worker
            num_devices = len(device_indices)

        return True, {suitable_worker: free_worker_devices[suitable_worker]}

      else:
        # Select worker with most devices available
        suitable_worker = None
        num_devices = 0
        for worker, device_indices in free_worker_devices.items():
          if len(device_indices) > num_devices:
            suitable_worker = worker
            num_devices = len(device_indices)

        if experiment.gpu_settings == 'forcemultigpu':
          if num_devices > 1:
            return True, {
              suitable_worker: free_worker_devices[suitable_worker]}
          else:
            return False, {}
        else:
          # Use available
          return True, {suitable_worker: free_worker_devices[suitable_worker]}

  # Returns a list of tuples of (experiment_id, reason))
  def _get_invalid_experiments(self):
    result = []
    for experiment_id, experiment in self.active_experiments.items():
      # Check if timeout
      if ((datetime.datetime.now() - experiment.start_time).seconds / 3600.0
          > self._experiment_time_limit):
        result.append((experiment_id, 'Timeout'))

    return result

  def _get_free_device_indices(self):
    result = dict()

    for worker in self.workers.values():
      free_indices = np.where(worker.device_states)[0]
      result[worker.host] = free_indices

    return result
