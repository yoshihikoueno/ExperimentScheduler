import datetime


class Experiment():
    exposed_attributes = (
        'docker_file', 'name', 'gpu_settings', 'use_multiple_workers', 'can_restart',
        'user_name', 'framework', 'schedule_time', 'start_time', 'finish_time', 'finish_return_code',
        'unique_id', 'can_be_run_on'
    )

    def __init__(
        self, docker_file, name, gpu_settings,
        use_multiple_workers, user_name, framework, can_restart=False,
        can_be_run_on=None,
    ):
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
        self.unique_id = f'{self.get_hash():x}'
        self.can_be_run_on = can_be_run_on

    def get_hash(self):
        value = hash(self.name + self.user_name + str(self.schedule_time))
        value = 2 * abs(value) + (value < 0)
        return value

    def to_dict(self):
        def _filter(value):
            if isinstance(value, set):
                return list(value)
            else: return value


        return dict(
            (attr, _filter(getattr(self, attr))) if hasattr(self, attr) else (attr, None)
            for attr in self.exposed_attributes
        )

    def __repr__(self):
        def is_target(attr_name):
            excludes = ('docker_file')

            if attr_name in excludes: return False
            if attr_name[0] == '_': return False
            if callable(vars(self)[attr_name]): return False
            return True

        target_attrs = list(filter(is_target, vars(self)))
        self_repr = ', '.join(map(lambda attr: f'{attr}: {vars(self)[attr]}', target_attrs))
        return f'Experiment ({self_repr})'
