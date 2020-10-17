import os

import experiment


class ExperimentBuilder():
  def __init__(self, resource_folder):
    self.resource_folder = resource_folder

  def build_experiment(self, experiment_dict):
    # Validity should be checked before calling this function
    assert(self.is_valid_experiment(experiment_dict)[0])

    return experiment.Experiment(
      docker_file=experiment_dict['dockerfile'],
      name=experiment_dict['experimentname'],
      use_multiple_workers='multiworker' in experiment_dict,
      framework=experiment_dict['framework'],
      gpu_settings=int(experiment_dict['gpusettings']),
      user_name=experiment_dict['username'],
      can_be_run_on=experiment_dict['can_be_run_on'],
    )

  def is_valid_experiment(self, experiment_dict):
    if 'dockerfile' not in experiment_dict:
      return False, 'No docker file specified.'
    if 'experimentname' not in experiment_dict:
      return False, 'No experiment name specified.'
    if 'framework' not in experiment_dict:
      return False, 'Framework missing.'
    if 'gpusettings' not in experiment_dict:
      return False, 'GPU settings missing.'
    if 'username' not in experiment_dict:
      return False, 'User name missing.'
    if not os.path.exists(os.path.join(self.resource_folder,
                                       experiment_dict['username'])):
      return False, 'User directory {} not found in workstation resource folder.'.format(os.path.join(self.resource_folder, experiment_dict['username']))

    if experiment_dict['framework'] not in ['tensorflow', 'other']:
      return False, 'Invalid development framework.'

    if not experiment_dict['gpusettings'].isdigit() or int(experiment_dict['gpusettings']) <= 0:
      return False, 'GPU settings must be integer.'

    if not isinstance(experiment_dict['can_be_run_on'], set):
      return False, 'field can_be_run_on is invalid'

    if not experiment_dict['can_be_run_on']:
      return False, 'at least one worker should be selected'

    if ('multiworker' in experiment_dict and experiment_dict['framework']
        != 'tensorflow'):
      return False, 'Using multi worker and not tensorflow is not implemented.'

    if ('multiworker' in experiment_dict and experiment_dict['gpusettings'] == 1):
      return False, 'Cannot use multiple workers and force single GPU use.'

    return True, 'Success.'
