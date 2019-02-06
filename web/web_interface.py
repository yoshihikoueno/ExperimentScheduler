import os

import flask

import experiment_builder
import task


class WebInterface():
  def __init__(self, scheduler_ref, num_devices_per_worker):
    self.app = flask.Flask(__name__)
    self.app.secret_key = os.urandom(16)

    @self.app.route('/', methods=['GET'])
    def index():
      msg = flask.session['msg'] if 'msg' in flask.session else ''
      success = msg == 'Success.'

      return flask.render_template(
        'index.html',
        form_msg=flask.session['msg'] if msg in flask.session else '',
        num_devices_per_worker=num_devices_per_worker,
        form_success=success, workers=scheduler_ref.workers,
        user_name_list=scheduler_ref.user_name_list,
        pending_experiments=scheduler_ref.pending_experiments,
        active_experiments=scheduler_ref.active_experiments,
        finished_experiments=scheduler_ref.finished_experiments)

    @self.app.route('/post', methods=['GET', 'POST'])
    def post():
      if flask.request.method == 'GET':
        return flask.redirect('/')

      msg = ''
      # Check if it was a stop experiment button
      if 'Stop' in list(flask.request.form.values()):
        # We have to stop an experiment
        experiment_id = list(flask.request.form.keys())[0]
        t = task.Task(task_type=task.TaskType.STOP_EXPERIMENT,
                      experiment_id=experiment_id, host=flask.request.host)
        scheduler_ref.task_queue.put(t)

      # Check if it was stdout button
      elif 'Stdout' in list(flask.request.form.values()):
       experiment_id = list(flask.request.form.keys())[0]
       if (int(experiment_id) in scheduler_ref.active_experiments
           or int(experiment_id) in scheduler_ref.finished_experiments):
         log_path = scheduler_ref.get_experiment_stdout_path(experiment_id)

         with open(log_path, 'r') as f:
           return flask.Response(f.read(), mimetype='text/plain')
       else:
         msg = 'Log not found.'

      # Check if it was stderr button
      elif 'Stderr' in list(flask.request.form.values()):
        experiment_id = list(flask.request.form.keys())[0]
        if (int(experiment_id) in scheduler_ref.active_experiments
           or int(experiment_id) in scheduler_ref.finished_experiments):
          log_path = scheduler_ref.get_experiment_stderr_path(experiment_id)

          with open(log_path, 'r') as f:
            return flask.Response(f.read(), mimetype='text/plain')
        else:
          msg = 'Log not found.'

      else:
        # Create experiment request
        success, msg = experiment_builder.is_valid_experiment(
          flask.request.form)

        if success:
          experiment = experiment_builder.build_experiment(
            flask.request.form)
          t = task.Task(
            task_type=task.TaskType.NEW_EXPERIMENT,
            experiment=experiment)
          scheduler_ref.task_queue.put(t)

      flask.session['msg'] = msg

      return flask.redirect('/')

  def run(self, public):
    # Host 0.0.0.0 is required to make server visible in local network.
    host = '0.0.0.0' if public else None
    self.app.run(debug=False, host=host)
