import os
import tempfile
import json
import subprocess
import logging
import sys
import re
import xmltodict
import pprint


class WorkerInterface:
    def __init__(
        self,
        host,
        tf_ports,
        logdir,
        resource_folder,
        docker_resource_folder,
        mem_hard_limit_ratio,
        mem_soft_limit_ratio,
    ):
        self.host = host
        # Contains booleans whether the device at its index is free or not
        self._device_states = [True] * self.get_num_gpus()

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

        # Check if our program stdout and sterr are attached to a tty.
        # If true, we need to tell ssh to allocate a pseudo tty, otherwise
        # the ssh program cannot be killed correctly.
        self.is_tty = sys.stdout.isatty() and sys.stderr.isatty()

        self.resource_folder = resource_folder
        self.docker_resource_folder = docker_resource_folder

        self.mem_hard_limit_ratio = mem_hard_limit_ratio
        self.mem_soft_limit_ratio = mem_soft_limit_ratio

    @property
    def device_states(self):
        return self._device_states

    # Get the experiment id running on device with device_index
    def get_experiment_id(self, device_index):
        for experiment, device_indices in self._active_experiments.values():
            if device_index in device_indices:
                return experiment.unique_id
        raise RuntimeError(
            f'Failed to find an experiment occupying device:{device_index}.\n'
            f'Active Experiments:\n{pprint.pformat(self._active_experiments)}'
        )

    def shutdown(self):
        for p in self._tf_server_processes.values():
            logging.info("Terminating TF Server Process.")
            p.terminate()
            p.wait()
        for p in self._experiment_processes.values():
            logging.info("Terminating Experiment Process.")
            p.terminate()
            p.wait()

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
                    result.append((experiment_id, f'Error: {return_code}'))

        return result

    def get_free_port(self):
        res = 0
        for port in self._tf_ports:
            if port not in self._used_tf_ports:
                res = port

        assert port != 0
        return res

    def start_tf_server(self, experiment_id, num_devices, tf_config_env):
        assert experiment_id not in self._tf_server_processes

        # Extract port from config
        port = int(tf_config_env['cluster'][tf_config_env['task']['type']][
            tf_config_env['task']['index']].split(':')[1])
        assert port not in self._used_tf_ports
        assert port in self._tf_ports

        self._used_tf_ports.append(port)

        # Assign devices
        device_indices = self._assign_free_device_indices(num_devices)

        env = self._get_env(device_indices, tf_config_env)

        tty = ['-t'] if self.is_tty else []
        cmd = ['ssh'] + tty + [self.host, 'python3',
                               'ExperimentScheduler/start_tf_server.py']

        p = subprocess.Popen(cmd, env=env, shell=False,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self._tf_server_processes[experiment_id] = (p, port, device_indices)

    def stop_tf_server(self, experiment_id):
        assert experiment_id in self._tf_server_processes

        p, port, device_indices = self._tf_server_processes[experiment_id]

        p.terminate()
        p.wait()

        assert port in self._used_tf_ports
        self._used_tf_ports.remove(port)

        for device_index in device_indices:
            assert self.device_states[device_index] is False
            self.device_states[device_index] = True

        del self._tf_server_processes[experiment_id]

    def start_experiment(self, experiment, num_devices, tf_config_env, is_restart):
        assert experiment.unique_id not in self._active_experiments

        # Assign devices
        device_indices = self._assign_free_device_indices(num_devices)

        env = self._get_env(tf_config_env=tf_config_env)

        env_args = []
        for k, v in env.items():
            env_args += ['-e', f"{k}='{v}'"]

        user_resource_folder = os.path.join(
            self.resource_folder, experiment.user_name)
        resource_folder_arg = [
            '--mount', f'type=bind,source={user_resource_folder},target={self.docker_resource_folder}']

        # We need to mount user and group folder, otherwise the docker environment
        # will create stuff as root in the result folders
        # We also need to add the localtime file, so that the timezone of the host
        # and the container will be the same
        options = ['-v', '/etc/passwd:/etc/passwd:ro',
                   '-v', '/etc/group:/etc/group:ro',
                   '-v', '/etc/localtime:/etc/localtime:ro',
                   '-v', '/sys:/sys:ro',
                   '-v', '/dev/shm:/dev/shm',
                   '-u', f'$(id -u {experiment.user_name}):$(id -g {experiment.user_name})',
                   ]
        tty = ['-t'] if self.is_tty else []

        # Add ram_safe_guard
        try:
            mem_total_mb = self.get_ram_size()
            hard_limit = int(mem_total_mb * self.mem_hard_limit_ratio)
            soft_limit = int(mem_total_mb * self.mem_soft_limit_ratio)
            ram_guard = ['--memory', f'{hard_limit}m', '--memory-reservation', f'{soft_limit}m']
        except Exception as e:
            logging.error(f'Failed to set memory limit.\n{e}')
        else:
            options += ram_guard

        docker_build_cmd = ['docker', 'build', '--no-cache', '-t', experiment.unique_id, '-']
        docker_run_cmd = [
            'docker', 'run', '--rm', '--name', experiment.unique_id,
            '--gpus', '\'"device=' + ','.join(map(str, device_indices)) + '"\'',
        ] + resource_folder_arg + options + env_args + [experiment.unique_id]
        remote_cmd = docker_build_cmd + ['&&'] + docker_run_cmd
        cmd = ['ssh'] + tty + [self.host] + remote_cmd

        # Create log files for this experiment
        stdout_file = os.path.join(self._logdir, f'{experiment.unique_id}_stdout')
        stderr_file = os.path.join(self._logdir, f'{experiment.unique_id}_stderr')
        with open(stdout_file, 'w') as out, open(stderr_file, 'w') as err:
            pid = subprocess.Popen(cmd, stdout=out, stderr=err, stdin=subprocess.PIPE, shell=False)
            pid.stdin.write(experiment.docker_file.encode())
            pid.stdin.close()

        self._experiment_processes[experiment.unique_id] = pid
        self._active_experiments[experiment.unique_id] = experiment, device_indices

    def stop_experiment(self, experiment_id, reason):
        assert experiment_id in self._experiment_processes
        assert experiment_id in self._active_experiments
        experiment = self._active_experiments[experiment_id][0]
        p = self._experiment_processes[experiment_id]
        return_code = p.poll()

        if return_code is None:
            # Process did not stop yet

            # First kill the ssh process
            p.kill()
            p.wait()

            # Stop container and remove image
            subprocess.call(
                ['ssh', f'{self.host}', 'docker',
                    'kill', experiment.unique_id],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            subprocess.call(
                ['ssh', f'{self.host}', 'docker',
                    'rmi', '-f', experiment.unique_id],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            return_code = f'Killed: {reason}'
        else:
            # Remove image
            subprocess.call(
                ['ssh', f'{self.host}', 'docker',
                    'rmi', '-f', experiment.unique_id],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            if return_code == 0:
                return_code = 'Success'
            else:
                return_code = f'Error: {return_code}'

        # If the image was not completely built before it was stopped,
        # it may still be dangling, so remove all dangling images
        subprocess.call(
            ['ssh', self.host, 'docker', 'rmi', '-f',
                '$(docker images -f "dangling=true" -q)'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        del self._experiment_processes[experiment_id]

        experiment, device_indices = self._active_experiments[experiment_id]

        for device_index in device_indices:
            assert self.device_states[device_index] is False
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

        assert len(device_indices) == num_devices
        return device_indices

    def _get_env(self, device_indices=None, tf_config_env=None, filtered=True):
        '''
        Get environment values in dict, which will be passed for `docker run`.

        Args:
            device_indices: indices of GPU
                **This option is deprecated. Use `--gpus` option in docker instead.**
            tf_config_env: environment settings for tensorflow, if any.
                Common usage of this is to set up an environment for multi-worker
                utilization.
            filtered: whether default enviroment values should be filtered
                in order not to ruine the default ENV in docker.
                For example, passing `PATH` to docker container may cause issues
                by overwriting `PATH` that the container had.
        '''
        filter_list = {
            'SHELL', 'SUDO_GID', 'LC_ADDRESS', 'LC_NAME', 'SUDO_COMMAND',
            'LC_MONETARY', 'SUDO_USER', 'PWD', 'KRB5CCNAME', 'LOGNAME',
            'HOME', 'LC_PAPER', 'LANG', 'LS_COLORS', 'LESSCLOSE', 'LC_IDENTIFICATION',
            'TERM', 'LESSOPEN', 'USER', 'SHLVL', 'LC_TELEPHONE', 'LC_MEASUREMENT',
            'LC_CTYPE', 'LC_TIME', 'XDG_DATA_DIRS', 'PATH', 'SUDO_UID',
            'MAIL', 'LC_NUMERIC', '_'}

        env = os.environ.copy()
        if filtered:
            for key in set(env) & filter_list:
                env.pop(key)

        if tf_config_env:
            env['TF_CONFIG'] = json.dumps(tf_config_env)
        env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        if device_indices:
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_indices))
        return env

    def get_ram_size(self):
        '''Retrieve worker RAM size in MB.'''
        try:
            meminfo = subprocess.check_output(['ssh', self.host, 'cat', '/proc/meminfo']).decode()
        except Exception as e:
            raise RuntimeError(f'Failed to retrieve meminfo.\n{e}')

        matched = re.search(r'^MemTotal:\s+(\d+)', meminfo).groups()
        if len(matched) != 1: raise RuntimeError(f'Unexpected content in meminfo.\nlen(matched) == {len(matched)}\n{meminfo}')
        mem_size = int(matched[0]) // 1024
        return mem_size

    def get_num_gpus(self):
        try:
            nvidia_smi_out = subprocess.check_output(['ssh', self.host, 'nvidia-smi', '-x', '-q'])
        except Exception as e:
            raise RuntimeError(f'Failed to access GPUS\n{e}')
        data = xmltodict.parse(nvidia_smi_out)
        num_gpu = int(data['nvidia_smi_log']['attached_gpus'])

        if num_gpu < 1: raise RuntimeError('No GPU found')
        return num_gpu
