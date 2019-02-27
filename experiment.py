import datetime


class Experiment():
  def __init__(self, docker_file, name, gpu_settings,
               use_multiple_workers, can_restart, user_name, framework):
    self.docker_file = docker_file
    self.name = name
    self.gpu_settings = gpu_settings
    self.use_multiple_workers = use_multiple_workers
    self.can_restart = can_restart
    self.user_name = user_name
    # tensorflow or chainer
    self.framework = framework
    self.schedule_time = datetime.datetime.now()
    self.start_time = None
    self.finish_time = None
    # Can be 'killed: <reason>', 'success', 'error: <code>'
    self.finish_return_code = None
    self.unique_id = str(hash(self.name + self.user_name + str(
      self.schedule_time)))
