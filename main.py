from absl import app
from absl import flags
from absl import logging

import jax
import tensorflow as tf

from clu import platform
from ml_collections import config_flags

import train

FLAGS = flags.FLAGS

flags.DEFINE_string('save_dir', None, 'Directory to store model data.')
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_boolean('colab', False, 'Running on google colab or not')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configurations.',
    lock_config=True)

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Hide any GPUs from Tensorflow.
  tf.config.experimental.set_visible_devices([], 'GPU')

  if FLAGS.colab:
    logging.info('Start colab TPU setup...')
    from jax.tools.colab_tpu import setup_tpu
    setup_tpu()

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                       f'process_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.save_dir, 'save_dir')
  train.train_and_evaluate(FLAGS.config, FLAGS.save_dir, FLAGS.seed)


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'save_dir'])
  app.run(main)
