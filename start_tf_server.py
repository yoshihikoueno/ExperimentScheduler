import tensorflow as tf
import os
import json


def run():
  env = os.environ.copy()
  tf_config = json.loads(env['TF_CONFIG'])
  cluster = tf_config['cluster']
  task = tf_config['task']

  server = tf.train.Server(cluster, job_name=task['type'],
                           task_index=task['index'])

  server.start()
  server.join()


if __name__ == '__main__':
  run()
