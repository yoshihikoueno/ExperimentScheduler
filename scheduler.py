import time
import queue
import copy
import subprocess
import os
import functools

import numpy as np

import task


class Scheduler:
  def __init__(self, num_devices_per_worker, workers, user_name_list):
    self.pending_experiments = []
    self.active_experiments = []
    self.finished_experiments = []

    self.user_name_list = user_name_list
    self.num_devices_per_worker = num_devices_per_worker
    self.workers = workers

    # Each index indicates whether the GPU is free or not
    self.device_states = dict()
    for worker in self.workers:
      self.device_states[worker] = [True] * self.num_devices_per_worker

    # The task queue storing tasks dedicated for the scheduler,
    # coming from the web interface
    self.task_queue = queue.Queue()

  def update(self):
    # Handle all tasks that came from the web interface
    tasks = []
    while not self.task_queue.empty():
      tasks.append(state.task_queue.get())

    for task in tasks:
      if task.task_type == TaskType.NEW_EXPERIMENT:
        self.pending_experiments.append(task.kvargs['experiment'])

    while len(self.pending_experiments) > 0:
      # Check if we can accomodate the next experiment
      if self._can_accomodate(self.pending_experiments[0]):
        experiment = self.pending_experiments.pop(0)
        worker_to_device_indices = self._assign_devices(experiment)

        # Mark relevant devices as used
        for worker, device_indices in worker_to_device_indices.items():
          for device_index in device_indices:
            assert(device_index is True)
            self.device_states[worker][device_index] = False

        #env = os.environ.copy()
        #  env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
       #   env['CUDA_VISIBLE_DEVICES'] = ['{},']
       #   call_fn = functools.partial(
       #     subprocess.Popen, args=experiment.exec_cmd.split(),
       #     cwd=os.path.join('/', 'home', experiment.user_name),)
       #   env =os.environ.copy()
       #   subprocess.Popen()

  # Returns dict mapping worker to device indices to be used by the
  # experiment
  def _assign_devices(self, experiment):
    free_device_indices = self._get_free_device_indices()

    if len(free_device_indices) == 0:
      return False, free_device_indices

    result_dict = dict()
    if experiment.gpu_settings == 'forcesinglegpu':
      worker = free_device_indices.keys()[0]
      device = free_device_indices[worker][0]

      result_dict[worker] = device
    else:
      if experiment.use_multiple_workers:
        result_dict = free_device_indices
        if experiment.gpu_settings == 'forcemultigpu':
          # Make sure we either have more than one worker, or if it is only one
          # atleast two devices on that worker
          if len(result_dict) < 2 and len(result_dict.values()[0]) < 2:
            return False, dict()
      else:
        result_dict = dict()

        suitable_worker
        # Find worker with most available devices
        for worker, device_indices in free_device_indices.items():

        if experiment.gpu_settings == 'forcemultigpu':
          # Make sure we h
          if len(result_dict) < 2 and len(result_dict.values()[0]) < 2:
            return False, dict()


    elif experiment.gpu_settings == 'forcemultigpu':
      if experiment.use_multiple_workers:
        result_indices = worker_to_device_indices
      else:
        result_indices = [worker_to_device_indices[0]]
    else:
      if experiment.use_multiple_workers:
        result_indices = worker_to_device_indices
      else:
        result_indices = [worker_to_device_indices[0]]

    return worker_to_device_indices

  def _can_accomodate(self, experiment):
    free_device_indices = np.transpose(np.where(self.device_states))

    if experiment.gpu_settings == 'forcesinglegpu':
      return len(free_device_indices) >= 1
    elif experiment.gpu_settings == 'forcemultigpu':
      return len(free_device_indices) > 1
    else:
      return len(free_device_indices) >= 1

  def _get_free_device_indices(self):
    result = dict()
    for worker, device_states in self.device_states.items():
      free_indices = np.where(device_states)[0]
      result[worker] = free_indices

    return result
