import flask

from server import experiment_builder as eb
from server import task


class WebInterface():
  def __init__(self, scheduler_ref):
    self.app = flask.Flask(__name__)

    self.experiment_builder = eb.ExperimentBuilder()

    @self.app.route('/', methods=['POST', 'GET'])
    def index():
      msg = ''
      success = False
      if flask.request.method == 'POST':
        success, msg = self.experiment_builder.is_valid_experiment(
          flask.request.form)

        if success:
          experiment = self.experiment_builder.build_experiment(
              flask.request.form)
          t = task.Task(
            task_type=task.TaskType.NEW_EXPERIMENT,
            experiment=experiment)
          scheduler_ref.task_queue.put(t)

      return flask.render_template(
        'index.html',
        num_available_gpu_per_worker=(scheduler_ref.
                                      num_available_gpu_per_worker),
        form_msg=msg,
        form_success=success, workers=scheduler_ref.device_states.keys(),
        user_name_list=scheduler_ref.user_name_list,
        pending_experiments=scheduler_ref.pending_experiments,
        active_experiments=scheduler_ref.active_experiments,
        finished_experiments=scheduler_ref.finished_experiments,
        device_states=scheduler_ref.device_states)

  def run(self, public):
    # Host 0.0.0.0 is required to make server visible in local network.
    host = '0.0.0.0' if public else None
    self.app.run(debug=False, host=host)
