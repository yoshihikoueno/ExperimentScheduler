import enum


class TaskType(enum.Enum):
  NEW_EXPERIMENT = 1


class Task:
  def __init__(self, task_type, **kvargs):
    if not isinstance(task_type, TaskType):
      raise ValueError('Invalid task type!')

    self.task_type = task_type
    self.kvargs = kvargs
