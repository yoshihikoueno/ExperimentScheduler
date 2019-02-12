import queue
import logging
import datetime
import os
import itertools
import atexit

import numpy as np

from task import TaskType


class Scheduler:
  def __init__(self, workers, user_name_list, logdir, experiment_time_limit,
               reorganize_experiments_interval):
    assert(experiment_time_limit >= 0)
    # List of experiments handled as a queue. Contains Experiment objects.
    # As long as another experiment is running or is in the pending queue
    # of that same user, experiments will be put in this queue
    self.waiting_experiments = []
    # List of pending experiments is handled as a queue. Contains Experiment
    # objects
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

    # Maps experiment_id to tuple (worker_to_num_devices)
    self._active_experiment_clusters = dict()

    self._logdir = logdir
    # In hours, 0 means no limit
    self._experiment_time_limit = experiment_time_limit
    # In hours, 0 means reorganization at every update step
    self._reorganize_experiments_interval = reorganize_experiments_interval
    self._t_last_reorganize = datetime.datetime.now()

    # Register shutdown callback
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

    self._stop_invalid_experiments()

    self._handle_waiting_experiments()

    if ((datetime.datetime.now() - self._t_last_reorganize).seconds / 3600.0
        > self._reorganize_experiments_interval):
      self._reorganize_experiments()
      self._t_last_reorganize = datetime.datetime.now()
    else:
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
        new_experiment = task.kvargs['experiment']

        # Check if this experiment should be put in waiting or pending queue.
        handled = False
        for experiment in self.pending_experiments:
          if experiment.user_name == new_experiment.user_name:
            self.waiting_experiments.append(new_experiment)
            handled = True
            break

        if handled:
          continue

        for experiment in self.active_experiments.values():
          if experiment.user_name == new_experiment.user_name:
            self.waiting_experiments.append(new_experiment)
            handled = True
            break

        if handled:
          continue

        self.pending_experiments.append(new_experiment)

        logging.info("Experiment '{}' of user '{}' queued.".format(
          new_experiment.name, new_experiment.user_name))

      elif task.task_type == TaskType.STOP_EXPERIMENT:
        experiment_id = task.kvargs['experiment_id']
        stop_experiment = False
        # Check if in waiting queue
        for i, experiment in enumerate(self.waiting_experiments):
          if experiment.unique_id == experiment_id:
            logging.info("Stop request from host '{}' for experiment '{}' from user '{}'".format(
              task.kvargs['host'], experiment.name,
              experiment.user_name))
            del self.waiting_experiments[i]
            stop_experiment = True
            break

        if stop_experiment is True:
          continue

         # Check if in pending queue
        for i, experiment in enumerate(self.pending_experiments):
          if experiment.unique_id == experiment_id:
            logging.info("Stop request from host '{}' for experiment '{}' from user '{}'".format(
              task.kvargs['host'], experiment.name,
              experiment.user_name))
            del self.pending_experiments[i]
            stop_experiment = True
            break

        if stop_experiment is True:
          continue

        # Check if active
        if experiment_id in self._active_experiment_clusters:
          stop_experiment = self.active_experiments[experiment_id]

        if stop_experiment is not None:
          logging.info("Stop request from host '{}' for experiment '{}' from user '{}'".format(
            task.kvargs['host'], stop_experiment.name,
            stop_experiment.user_name))
          self._stop_experiment(experiment_id, 'Killed: Web request')
        else:
          logging.debug('ID not there')

      else:
        logging.error("Task {} not implemented!".format(task.task_type))

  def _handle_finished_experiments(self):
    # Check if experiments have finished
    for worker in self.workers.values():
      finished_experiments = worker.poll_experiments()
      for finished_experiment_id, return_code in finished_experiments:
        self._stop_experiment(finished_experiment_id, reason=return_code)

  def _stop_invalid_experiments(self):
    invalid_experiments = self._get_invalid_experiments()
    for experiment_id, reason in invalid_experiments:
      self._stop_experiment(experiment_id, reason)

  def _handle_waiting_experiments(self):
    # Check if we can move an experiment from waiting to pending queue
    move_indices = []
    for i, waiting_experiment in enumerate(self.waiting_experiments):
      continue_wait = False
      for experiment in self.pending_experiments:
        if experiment.user_name == waiting_experiment.user_name:
          continue_wait = True
          break

      if continue_wait is True:
        continue

      for experiment in self.active_experiments.values():
        if experiment.user_name == waiting_experiment.user_name:
          continue_wait = True
          break

      if continue_wait is False:
        move_indices.append(i)
        self.pending_experiments.append(waiting_experiment)

    # We need to delete some from the waiting list
    move_indices.sort()
    move_indices[::-1]
    for i in move_indices:
      del self.waiting_experiments[i]

  # This function reassigns devices among currently running and pending
  # experiments, so that each experiments gets a fair amount of computational
  # resources
  def _reorganize_experiments(self):
    # Reset all devices
    free_worker_to_devices = dict()
    num_free_devices = 0
    for worker in self.workers.values():
      free_worker_to_devices[worker.host] = len(worker.device_states)
      num_free_devices += len(worker.device_states)

    experiment_id_to_device_assignments = dict()

    active_experiment_list = list(self.active_experiments.values())

    # First, we need to make sure that experiments that cannot be restarted
    # or are assigned only one device get the same assignment as before
    for i, experiment in enumerate(active_experiment_list):
      if (experiment.can_restart is False
          or experiment.gpu_settings == 'forcesinglegpu'):
        experiment_id_to_device_assignments[experiment.unique_id] = (
          self._active_experiment_clusters[experiment.unique_id])
        for worker, num_devices in \
            experiment_id_to_device_assignments[experiment.unique_id]:
          free_worker_to_devices[worker] -= num_devices
          num_free_devices -= num_devices
          assert(free_worker_to_devices[worker] >= 0)
          if free_worker_to_devices[worker] == 0:
            del free_worker_to_devices[worker]
        # We no longer need to care about those experiments
        del active_experiment_list[i]

    # Now assign remaining devices in a round-robin fashion
    combined_experiment_list = (active_experiment_list
                                + self.pending_experiments)
    # List of experiment indices for single GPU experiments. We want to assign
    # them after all multi GPU experiments, but still we want to reserve a
    # device of course.
    reserved_devices = []
    assignment_occured = True
    while assignment_occured:
      assignment_occured = False
      if num_free_devices <= 0 + len(reserved_devices):
        break

      for experiment_list_index in range(len(combined_experiment_list)):
        experiment = combined_experiment_list[experiment_list_index]
        if experiment.gpu_settings == 'forcesinglegpu':
          if experiment_list_index not in reserved_devices:
            reserved_devices.append(experiment_list_index)

          continue
        else:
          # First, try to assign a device from a worker where this experiment
          # is already present
          if experiment.unique_id in experiment_id_to_device_assignments:
            device_assigned = False
            for worker, num_devices in experiment_id_to_device_assignments[
                experiment.unique_id].items():
              if worker in free_worker_to_devices:
                # We found a suitable worker device
                experiment_id_to_device_assignments[experiment.unique_id][
                  worker] += 1
                free_worker_to_devices[worker] -= 1
                num_free_devices -= 1
                assert(free_worker_to_devices[worker] >= 0)
                if free_worker_to_devices[worker] == 0:
                  del free_worker_to_devices[worker]

                device_assigned = True

                break
            if device_assigned is True:
              # We are done with this experiment for now
              assignment_occured = True
              continue

        if (not experiment.use_multiple_workers
            and experiment.unique_id in experiment_id_to_device_assignments):
          # We were not able to assign to the same worker, but are unable to
          # assign to other workers. This experiment cannot get any
          # more experiments
          continue

        # Assign to the worker with the most free devices
        suitable_worker = None
        num_max_devices = 0
        for worker, num_devices in free_worker_to_devices.items():
          if num_devices > num_max_devices:
            num_max_devices = num_devices
            suitable_worker = worker

        if experiment.unique_id not in experiment_id_to_device_assignments:
          experiment_id_to_device_assignments[experiment.unique_id] = dict()

        experiment_id_to_device_assignments[experiment.unique_id][
          suitable_worker] = 1
        free_worker_to_devices[suitable_worker] -= 1
        num_free_devices -= 1
        assignment_occured = True
        if free_worker_to_devices[suitable_worker] == 0:
          del free_worker_to_devices[suitable_worker]

    # Handle reserved devices
    for experiment_index in reserved_devices:
      experiment = combined_experiment_list[experiment_index]
      worker = list(free_worker_to_devices.keys())[0]
      experiment_id_to_device_assignments[experiment.unique_id] = dict()
      experiment_id_to_device_assignments[experiment.unique_id][worker] = 1
      free_worker_to_devices[worker] -= 1
      num_free_devices -= 1
      assert(free_worker_to_devices[worker] >= 0)
      if free_worker_to_devices[worker] == 0:
        del free_worker_to_devices[worker]

    # If the device assignment of an experiment differs from the previous
    # assignment, then we need to restart the experiment
    restart_list = []
    start_list = []
    for experiment_id, device_assignment in \
        experiment_id_to_device_assignments.items():
      if experiment_id in self._active_experiment_clusters:
        if device_assignment != self._active_experiment_clusters[
            experiment_id]:
          # Restart necessary
          restart_list.append(self.active_experiments[experiment_id])
      else:
        found_experiment = False
        for i in range(len(self.pending_experiments)):
          if self.pending_experiments[i].unique_id == experiment_id:
            start_list.append(self.pending_experiments.pop(i))
            found_experiment = True
            break

        assert(found_experiment)

    # Stop all experiments
    for experiment in restart_list:
      self._stop_experiment(experiment.unique_id, '', pending_restart=True)

    # Restart experiments
    for experiment in restart_list:
      self._start_experiment(
        experiment, experiment_id_to_device_assignments[experiment.unique_id],
        is_restart=True)

    # Start new experiments
    for experiment in start_list:
      self._start_experiment(
        experiment, experiment_id_to_device_assignments[experiment.unique_id],
        is_restart=False)

  def _try_start_experiments(self):
    while (len(self.pending_experiments) > 0):
      free_worker_devices = self._get_free_devices()

      can_fit, assigned_devices = (
          _can_fit(self.pending_experiments[0],
                   free_worker_devices=free_worker_devices))
      if can_fit is True:
        self._start_experiment(
          self.pending_experiments.pop(0),
          worker_to_devices=assigned_devices, is_restart=False)
      else:
        break

  def _start_experiment(self, experiment, worker_to_devices,
                        is_restart):
    worker_hosts = list(worker_to_devices.keys())
    num_devices_list = list(worker_to_devices.values())

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
        num_devices = num_devices_list[i + 1]
        task_config = {'type': 'worker', 'index': i}
        tf_config_env = {'cluster': cluster_config, 'task': task_config}
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
      # Chainer
      self.workers[worker_hosts[0]].start_experiment(
        experiment, num_devices=num_devices_list[0], tf_config_env=None)

    if is_restart is False:
      experiment.start_time = datetime.datetime.now()
      logging.info("Experiment '{}' of user '{}' started.".format(
          experiment.name, experiment.user_name))
    assert(experiment.unique_id not in self.active_experiments)
    self.active_experiments[experiment.unique_id] = experiment
    assert(experiment.unique_id not in self._active_experiment_clusters)
    self._active_experiment_clusters[experiment.unique_id] = (
      worker_to_devices)

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

  # Returns a list of tuples of (experiment_id, reason))
  def _get_invalid_experiments(self):
    result = []
    for experiment_id, experiment in self.active_experiments.items():
      # Check if timeout
      if (self._experiment_time_limit > 0 and (
          datetime.datetime.now() - experiment.start_time).seconds / 3600.0
          > self._experiment_time_limit):
        result.append((experiment_id, 'Timeout'))

    return result

  def _get_free_devices(self):
    result = dict()

    for worker in self.workers.values():
      free_indices = np.where(worker.device_states)[0]
      result[worker.host] = len(free_indices)

    return result


# Checks if the experiment can fit in free_worker_devices
# and returns a tuple consisting of a boolean indicating this result and
# the assigned worker to devices map
def _can_fit(experiment, free_worker_devices):
  total_len = 0
  for worker, num_devices in free_worker_devices.items():
    total_len += num_devices

  if total_len == 0:
    return False, {}

  if experiment.use_multiple_workers:
    return True, free_worker_devices

  if experiment.gpu_settings == 'forcesinglegpu':
    # Select worker with fewest devices available
    suitable_worker = None
    num_min_devices = 99999
    for worker, num_devices in free_worker_devices.items():
      if num_devices < num_min_devices:
        suitable_worker = worker
        num_min_devices = num_devices

    return True, {suitable_worker: free_worker_devices[suitable_worker]}
  else:
    # Select worker with most devices available
    suitable_worker = None
    num_max_devices = 0
    for worker, num_devices in free_worker_devices.items():
      if num_devices > num_max_devices:
        suitable_worker = worker
        num_max_devices = num_devices
    return True, {suitable_worker: free_worker_devices[suitable_worker]}
