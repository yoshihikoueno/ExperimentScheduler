import os
import json
import subprocess
import logging


class WorkerInterface:
  def __init__(self, host, tf_ports, num_devices, logdir):
    self.host = host
    self.device_states = [True] * num_devices

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

  def start_tf_server(self, experiment_id, device_indices, tf_config_env):
    assert(experiment_id not in self._tf_server_processes)

    # Extract port from config
    port = int(tf_config_env['cluster'][tf_config_env['task']['type']][
      tf_config_env['task']['index']].split(':')[1])
    assert(port not in self._used_tf_ports)
    assert(port in self._tf_ports)

    self._used_tf_ports.append(port)

    # Mark relevant devices as used
    for device_index in device_indices:
      assert(self.device_states[device_index] is True)
      self.device_states[device_index] = False

    env = self._get_env(device_indices, tf_config_env)

    cmd = ('ssh -t {} python3 ExperimentScheduler/start_tf_server.py'
           .format(self.host))

    p = subprocess.Popen(cmd, env=env, shell=True)

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

  def start_experiment(self, experiment, device_indices, tf_config_env):
    assert(experiment.unique_id not in self._active_experiments)

    # Mark devices as used
    for device_index in device_indices:
      assert(self.device_states[device_index] is True)
      self.device_states[device_index] = False

    env = self._get_env(device_indices, tf_config_env)

    cmd = 'ssh -t {} {}'.format(self.host, experiment.exec_cmd)

    # Create log files for this experiment
    with open(os.path.join(self._logdir, '{}_stdout'.format(
        experiment.unique_id)), 'w') as out, open(os.path.join(
          self._logdir, '{}_stderr'.format(experiment.unique_id)),
                                                   'w') as err:
      self._experiment_processes[experiment.unique_id] = subprocess.Popen(
        cmd, env=env, universal_newlines=True,
        stdout=out, stderr=err, shell=True, cwd='/home/{}'.format(
          experiment.user_name))

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

  def _get_env(self, device_indices, tf_config_env):
    env = os.environ.copy()
    env['TF_CONFIG'] = (None if tf_config_env is None
                        else json.dumps(tf_config_env))
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    device_list = ''
    for device in device_indices:
      device_list += '{},'.format(device)
    device_list = device_list[:-1]
    env['CUDA_VISIBLE_DEVICES'] = device_list

    return env
