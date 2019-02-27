import os
import json
import subprocess
import logging


class WorkerInterface:
  def __init__(self, host, tf_ports, num_devices, logdir, resource_folder,
               docker_resource_folder):
    self.host = host
    # Contains booleans whether the device at its index is free or not
    self._device_states = [True] * num_devices

    # Maps experiment_id to a tuple of (process, port, device_ids)
    self._tf_server_processes = dict()
    # Maps experiment_id to process
    self._experiment_processes = dict()
    # Maps experiment_id to (experiment, device_indices)
    self._active_experiments = dict()

    self._tf_ports = tf_ports
    # Contains ports that are in use
    self._used_tf_ports = []

    self._logdir = logdir

    self.resource_folder = resource_folder
    self.docker_resource_folder = docker_resource_folder

  @property
  def device_states(self):
    return self._device_states

  # Get the experiment id running on device with device_index
  def get_experiment_id(self, device_index):
    for experiment, device_indices in self._active_experiments.values():
      if device_index in device_indices:
        return experiment.unique_id

    assert(False)

  def shutdown(self):
    for p in self._tf_server_processes.values():
      logging.info("Terminating TF Server Process.")
      p.terminate()
    for p in self._experiment_processes.values():
      logging.info("Terminating Experiment Process.")
      p.terminate()
  
  # Returns a list of tuples of (experiment_id, return_code) that are finished
  def poll_experiments(self):
    result = []
    # Check if some thread finished
    for experiment_id, p in self._experiment_processes.items():
      return_code = p.poll()
      if return_code is not None:
        # Terminated
        if return_code == 0:
          result.append((experiment_id, 'Success.'))
        else:
          result.append((experiment_id, 'Error: {}'.format(return_code)))

    return result

  def get_free_port(self):
    res = 0
    for port in self._tf_ports:
      if port not in self._used_tf_ports:
        res = port

    assert(port != 0)

    return res

  def start_tf_server(self, experiment_id, num_devices, tf_config_env):
    assert(experiment_id not in self._tf_server_processes)

    # Extract port from config
    port = int(tf_config_env['cluster'][tf_config_env['task']['type']][
      tf_config_env['task']['index']].split(':')[1])
    assert(port not in self._used_tf_ports)
    assert(port in self._tf_ports)

    self._used_tf_ports.append(port)

    # Assign devices
    device_indices = self._assign_free_device_indices(num_devices)

    env = self._get_env(device_indices, tf_config_env)

    cmd = ['ssh', '-t', self.host, 'python3',
           'ExperimentScheduler/start_tf_server.py']

    p = subprocess.Popen(cmd, env=env, shell=False, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    self._tf_server_processes[experiment_id] = (p, port, device_indices)

  def stop_tf_server(self, experiment_id):
    assert(experiment_id in self._tf_server_processes)

    p, port, device_indices = self._tf_server_processes[experiment_id]

    p.kill()

    assert(port in self._used_tf_ports)
    self._used_tf_ports.remove(port)

    for device_index in device_indices:
      assert(self.device_states[device_index] is False)
      self.device_states[device_index] = True

    del self._tf_server_processes[experiment_id]

  def start_experiment(self, experiment, num_devices, tf_config_env,
                       is_restart):
    assert(experiment.unique_id not in self._active_experiments)

    # Assign devices
    device_indices = self._assign_free_device_indices(num_devices)

    env = self._get_env(device_indices, tf_config_env)

    env_args = []
    for k, v in env.items():
      env_args += ['-e', "{}='{}'".format(k, v)]

    user_resource_folder = os.path.join(self.resource_folder,
                                        experiment.user_name)
    resource_folder_arg = ['--mount', 'type=bind,source={},target={}'.format(
      user_resource_folder, self.docker_resource_folder)]

    cmd = ['ssh', '-t', '{}'.format(self.host),
           'echo', '"{}"'.format(experiment.docker_file), '|',
           'docker', 'build', '--no-cache', '-t',
           '{}'.format(experiment.user_name), '-', '&&', 'docker', 'run',
           '--rm', '--runtime=nvidia'] + resource_folder_arg + env_args + [
             '{}'.format(experiment.user_name)]

    # Create log files for this experiment
    with open(os.path.join(self._logdir, '{}_stdout'.format(
        experiment.unique_id)), 'w') as out, open(os.path.join(
          self._logdir, '{}_stderr'.format(experiment.unique_id)),
                                                   'w') as err:
      self._experiment_processes[experiment.unique_id] = subprocess.Popen(
        cmd, stdout=out, stderr=err, shell=False)

    self._active_experiments[experiment.unique_id] = (
      experiment, device_indices)

  def stop_experiment(self, experiment_id, reason):
    assert(experiment_id in self._experiment_processes)
    assert(experiment_id in self._active_experiments)

    p = self._experiment_processes[experiment_id]
    return_code = p.poll()
    if return_code is None:
      p.kill()
      return_code = 'Killed: {}'.format(reason)
    else:
      if return_code == 0:
        return_code = 'Success'
      else:
        return_code = 'Error: {}'.format(return_code)

    del self._experiment_processes[experiment_id]

    experiment, device_indices = self._active_experiments[experiment_id]

    for device_index in device_indices:
      assert(self.device_states[device_index] is False)
      self.device_states[device_index] = True

    del self._active_experiments[experiment_id]

    return return_code

  def _assign_free_device_indices(self, num_devices):
    device_indices = []
    for i, device_state in enumerate(self._device_states):
      if device_state is True:
        device_indices.append(i)
        self._device_states[i] = False
        if len(device_indices) == num_devices:
          break

    assert(len(device_indices) == num_devices)

    return device_indices

  def _get_env(self, device_indices, tf_config_env):
    env = os.environ.copy()
    if tf_config_env:
      env['TF_CONFIG'] = json.dumps(tf_config_env)

    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    device_list = ''
    for device in device_indices:
      device_list += '{},'.format(device)
    device_list = device_list[:-1]
    env['CUDA_VISIBLE_DEVICES'] = device_list

    return env
