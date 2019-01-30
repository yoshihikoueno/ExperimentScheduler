import datetime


class Experiment():
  def __init__(self, exec_cmd, name, user_name, gpu_settings,
               use_multiple_workers, framework):
    self.exec_cmd = exec_cmd
    self.name = name
    self.user_name = user_name
    self.gpu_settings = gpu_settings
    self.use_multiple_workers = use_multiple_workers
    # tensorflow or chainer
    self.framework = framework
    self.schedule_time = datetime.datetime.now()
    self.start_time = None
    self.finish_time = None
    # Can be 'killed: <reason>', 'success', 'error: <code>'
    self.finish_return_code = None
    self.unique_id = hash(self.name + self.user_name + str(self.schedule_time))
