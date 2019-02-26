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
      gpu_settings=(experiment_dict['gpusettings'] if 'gpusettings'
                    in experiment_dict else 'forcesinglegpu'),
      use_multiple_workers='multiworker' in experiment_dict,
      can_restart='canrestart' in experiment_dict,
      framework=experiment_dict['framework'],
      user_name=experiment_dict['username'],
      input_res=os.path.join(self.resource_folder, experiment_dict['username'],
                             experiment_dict['inputres']),
      output_folder=os.path.join(self.resource_folder,
                                 experiment_dict['username'],
                                 experiment_dict['outputfolder']))

  def is_valid_experiment(self, experiment_dict):
    if 'dockerfile' not in experiment_dict:
      return False, 'No docker file specified.'
    if 'outputfolder' not in experiment_dict:
      return False, 'No output folder specified.'
    if 'experimentname' not in experiment_dict:
      return False, 'No experiment name specified.'
    if 'framework' not in experiment_dict:
      return False, 'Framework missing.'
    if 'username' not in experiment_dict:
      return False, 'User name missing.'

    if experiment_dict['framework'] not in ['tensorflow', 'other']:
      return False, 'Invalid development framework.'
    gpusettings = (experiment_dict['gpusettings'] if 'gpusettings'
                   in experiment_dict else 'forcesinglegpu')
    if (gpusettings not in ['useavailable', 'forcesinglegpu']):
      return False, 'Invalid GPU settings.'

    if ('canrestart' not in experiment_dict
        and gpusettings != 'forcesinglegpu'):
      return False, 'Only single GPU can be used when restart is impossible.'

    if ('multiworker' in experiment_dict and experiment_dict['framework']
        != 'tensorflow'):
      return False, 'Using multi worker and not tensorflow is not implemented.'

    if ('multiworker' in experiment_dict and gpusettings == 'forcesinglegpu'):
      return False, 'Cannot use multiple workers and force single GPU use.'

    input_res = os.path.join(self.resource_folder, experiment_dict['username'],
                             experiment_dict['inputres'])
    if (not os.path.exists(input_res)):
      return False, 'Input resource {} does not exist'.format(input_res)

    input_res_folder = os.path.join(
      self.resource_folder, experiment_dict['username'],
      experiment_dict['inputres'])
    if (not os.path.exists(input_res_folder)):
      return False, 'Input resource folder {} does not exist'.format(
        input_res_folder)

    return True, 'Success.'
