import os

import ldap
from flask import request, render_template, flash, redirect, \
    url_for, g, Flask, Response
from flask_login import current_user, login_required, LoginManager

import experiment_builder as eb
import task
from web.auth.models import UserManager, LoginForm


class WebInterface():
    def __init__(self, scheduler_ref, resource_folder, docker_resource_folder):
        self.app = Flask(__name__)
        self.app.config['WTF_CSRF_SECRET_KEY'] = os.urandom(16)
        self.app.config['LDAP_PROVIDER_URL'] = 'ldap://ipa.kumalab.local:389'
        # The login manager should store the next value in the session
        self.app.config['USE_SESSION_FOR_NEXT'] = True

        self.app.secret_key = os.urandom(16)

        self.login_manager = LoginManager()
        self.login_manager.init_app(self.app)
        self.login_manager.login_view = 'login'
        self.login_manager.login_message_category = 'info'

        self.conn = ldap.ldapobject.ReconnectLDAPObject(
            self.app.config['LDAP_PROVIDER_URL'])

        self.css_file = 'base.css?v={}'.format(os.path.getmtime(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static',
                         'css', 'base.css')))

        self.experiment_builder = eb.ExperimentBuilder(resource_folder)
        self.docker_resource_folder = docker_resource_folder

        @self.login_manager.user_loader
        def load_user(uid):
            return UserManager.get_user(self.conn, uid)

        @self.app.before_request
        def get_current_user():
            g.user = current_user

        @self.app.route('/')
        def root():
            if current_user.is_authenticated:
                return redirect(url_for('home'))
            else:
                return redirect(url_for('login'))

        @self.app.route('/login', methods=['GET', 'POST'])
        def login():
            if current_user.is_authenticated:
                flash("You are already logged in!")
                return redirect(url_for('home'))
            else:
                form = LoginForm(request.form)

                if request.method == 'POST' and form.validate():
                    # Login Request
                    username = request.form.get('username')
                    password = request.form.get('password')

                    try:
                        UserManager.try_login(self.conn, username, password)
                    except ldap.INVALID_CREDENTIALS:
                        flash(
                            "Invalid username or password. Please try again.", 'danger')
                        return render_template('login.html', form=form,
                                               css_file=self.css_file)
                    except ldap.SERVER_DOWN:
                        flash(
                            "LDAP authentication server seems to be down. Please contact a system administrator.")
                        return render_template('login.html', form=form,
                                               css_file=self.css_file)
                    except Exception as e:
                        flash("Unhandled LDAP exception: {}".format(e))
                        return render_template('login.html', form=form,
                                               css_file=self.css_file)

                    flash("You have successfully logged in!", 'success')
                    return redirect(url_for('home'))

                if form.errors:
                    flash(form.errors, 'danger')

                return render_template('login.html', form=form, css_file=self.css_file)

        @self.app.route('/home', methods=['GET'])
        @login_required
        def home():
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
                        workstation_load_table_content[row].append(
                            ('Free', ''))
                    else:
                        workstation_load_table_content[row].append((
                            'Used', active_experiment_to_color[
                                scheduler_ref.workers[worker].get_experiment_id(column)]))

            max_num_gpu = max(map(lambda x: len(x.device_states), scheduler_ref.workers.values()))
            total_num_gpu = sum(map(lambda x: len(x.device_states), scheduler_ref.workers.values()))

            return render_template(
                'home.html',
                docker_resource_folder=self.docker_resource_folder,
                workstation_load_table_content=workstation_load_table_content,
                max_num_gpu=max_num_gpu,
                total_num_gpu=total_num_gpu,
                active_experiment_to_color=active_experiment_to_color,
                workers=scheduler_ref.workers,
                pending_experiments=scheduler_ref.pending_experiments,
                active_experiments=scheduler_ref.active_experiments,
                finished_experiments=scheduler_ref.finished_experiments,
                css_file=self.css_file,
                max_time=scheduler_ref.experiment_time_limit,
                user_name=current_user.given_name)

        @self.app.route('/home/docker_example')
        @login_required
        def docker_example():
            return render_template('docker_example.html', css_file=self.css_file)

        @self.app.route('/home/post', methods=['GET', 'POST'])
        @login_required
        def post():
            if request.method == 'GET':
                return redirect(url_for('home'))

            msg = ''
            # Check if it was a stop experiment button
            if 'Stop' in list(request.form.values()):
                # We have to stop an experiment
                experiment_id = list(request.form.keys())[0]
                t = task.Task(task_type=task.TaskType.STOP_EXPERIMENT,
                              experiment_id=experiment_id, host=request.host)
                scheduler_ref.task_queue.put(t)

            # Check if it was stdout button
            elif 'Stdout' in list(request.form.values()):
                experiment_id = list(request.form.keys())[0]
                if (experiment_id in scheduler_ref.active_experiments
                        or experiment_id in scheduler_ref.finished_experiments):
                    log_path = scheduler_ref.get_experiment_stdout_path(
                        experiment_id)

                    with open(log_path, 'r') as f:
                        return Response(f.read(), mimetype='text/plain')
                else:
                    flash('Log not found.', 'danger')

            # Check if it was stderr button
            elif 'Stderr' in list(request.form.values()):
                experiment_id = list(request.form.keys())[0]
                if (experiment_id in scheduler_ref.active_experiments
                        or experiment_id in scheduler_ref.finished_experiments):
                    log_path = scheduler_ref.get_experiment_stderr_path(
                        experiment_id)

                    with open(log_path, 'r') as f:
                        return Response(f.read(), mimetype='text/plain')
                else:
                    flash('Log not found.', 'danger')
            elif 'Dockerfile' in list(request.form.values()):
                experiment_id = list(request.form.keys())[0]
                try:
                    experiment = scheduler_ref.get_experiment(experiment_id)
                    return Response(experiment.docker_file, mimetype='text/plain')
                except Exception as e:
                    flash(f'Error: {e} ', 'danger')

            else:
                request_dict = dict(request.form)
                request_dict['username'] = current_user.uid
                request_dict['can_be_run_on'] = {
                    key[len('include_'):] for key in request_dict if key.startswith('include_')
                }
                # Create experiment request
                success, msg = self.experiment_builder.is_valid_experiment(
                    request_dict)

                if success:
                    experiment = self.experiment_builder.build_experiment(
                        request_dict)
                    t = task.Task(
                        task_type=task.TaskType.NEW_EXPERIMENT,
                        experiment=experiment)
                    scheduler_ref.task_queue.put(t)
                    flash(msg, 'success')

                else:
                    flash(msg, 'danger')

            return redirect(url_for('home'))

        @self.app.route('/logout', methods=['GET', 'POST'])
        @login_required
        def logout():
            UserManager.logout(current_user)

            return redirect(url_for('login'))

    def run(self, public, port):
        # Host 0.0.0.0 is required to make server visible in local network.
        host = '0.0.0.0' if public else None
        self.app.run(debug=False, host=host, port=port)
