import os

import experiment


class ExperimentBuilder():
  def __init__(self):
    pass

  def build_experiment(self, experiment_dict):
    # Validity should be checked before calling this function
    assert(self.is_valid_experiment(experiment_dict)[0])

    return experiment.Experiment(
      exec_cmd=experiment_dict['execcmd'],
      name=experiment_dict['experimentname'],
      user_name=experiment_dict['username'],
      gpu_settings=experiment_dict['gpusettings'],
      use_multiple_workers='multiworker' in experiment_dict,
      framework=experiment_dict['framework'])

  def is_valid_experiment(self, experiment_dict):
    keys = experiment_dict.keys()
    if 'execcmd' not in keys:
      return False, 'No execution command specified.'
    if 'experimentname' not in keys:
      return False, 'No experiment name specified.'
    if 'username' not in keys:
      return False, 'No user name specified.'
    if 'framework' not in experiment_dict:
      return False, 'Framework missing.'
    if experiment_dict['framework'] not in ['tensorflow', 'chainer']:
      return False, 'Invalid development framework.'
    if 'gpusettings' not in experiment_dict:
      return False, 'GPU settings missing.'
    if experiment_dict['gpusettings'] not in ['useavailable', 'forcesinglegpu',
                                              'forcemultigpu']:
      return False, 'Invalid GPU settings.'

    user_name = experiment_dict['username']

    user_dir = '/home/{}'.format(user_name)
    if (not os.path.exists(user_dir)):
      return False, 'Invalid user name.'

    if ('multiworker' in experiment_dict and experiment_dict['framework']
        != 'tensorflow'):
      return False, 'Using multi worker and not tensorflow is not implemented.'

    return True, 'Success.'
