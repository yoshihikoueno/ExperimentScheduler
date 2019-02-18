import os

import flask

import experiment_builder
import task


class WebInterface():
  def __init__(self, scheduler_ref):
    self.app = flask.Flask(__name__)
    self.app.secret_key = os.urandom(16)
    # We need to provide a unique file name if the css file changed,
    # so that browsers dont keep an old version of it cached
    self.css_file = 'styles/index.css?v={}'.format(os.path.getmtime(
      os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static',
                   'styles', 'index.css')))

    @self.app.route('/', methods=['GET'])
    def index():
      msg = flask.session['msg'] if 'msg' in flask.session else ''
      success = msg == 'Success.'

      color_list = ['red', 'green', 'blue', 'gold', 'orange', 'olive',
                    'violet', 'indigo']
      active_experiment_to_color = dict()
      for i, experiment_id in enumerate(list(
          scheduler_ref.active_experiments.keys())):
        active_experiment_to_color[experiment_id] = color_list[
          i % len(color_list)]

      workstation_load_table_content = []
      for row in range(len(scheduler_ref.workers)):
        worker = list(scheduler_ref.workers.keys())[row]
        workstation_load_table_content.append([
          (worker, '')])
        for column in range(len(list(
            scheduler_ref.workers.values())[row].device_states)):
          if (scheduler_ref.workers[worker].device_states[column] is True):
            workstation_load_table_content[row].append(('Free', ''))
          else:
            workstation_load_table_content[row].append((
              'Used', active_experiment_to_color[
                scheduler_ref.workers[worker].get_experiment_id(column)]))

      max_num_gpu = 0
      for worker in scheduler_ref.workers.values():
        if len(worker.device_states) > max_num_gpu:
          max_num_gpu = len(worker.device_states)

      return flask.render_template(
        'index.html',
        form_msg=msg,
        workstation_load_table_content=workstation_load_table_content,
        max_num_gpu=max_num_gpu,
        active_experiment_to_color=active_experiment_to_color,
        form_success=success, workers=scheduler_ref.workers,
        user_name_list=scheduler_ref.user_name_list,
        waiting_experiments=scheduler_ref.waiting_experiments,
        pending_experiments=scheduler_ref.pending_experiments,
        active_experiments=scheduler_ref.active_experiments,
        finished_experiments=scheduler_ref.finished_experiments,
        css_file=self.css_file,
        max_time=scheduler_ref.experiment_time_limit)

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
       if (experiment_id in scheduler_ref.active_experiments
           or experiment_id in scheduler_ref.finished_experiments):
         log_path = scheduler_ref.get_experiment_stdout_path(experiment_id)

         with open(log_path, 'r') as f:
           return flask.Response(f.read(), mimetype='text/plain')
       else:
         msg = 'Log not found.'

      # Check if it was stderr button
      elif 'Stderr' in list(flask.request.form.values()):
        experiment_id = list(flask.request.form.keys())[0]
        if (experiment_id in scheduler_ref.active_experiments
           or experiment_id in scheduler_ref.finished_experiments):
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
