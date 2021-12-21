from typing import Any
from functools import partial
import time

import jax
import jax.numpy as jnp
from jax import random
from jax import lax

import flax
from flax import jax_utils
from flax import optim
import optax

from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state

from clu import metric_writers
from clu import periodic_actions
from absl import logging

import ml_collections

import models
import input_pipeline


def create_model(model_cls, half_precision):
  platform = jax.local_devices()[0].platform
  if half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
    else:
      model_dtype = jnp.bfloat16

  else:
    model_dtype = jnp.float32
  return model_cls(dtype=model_dtype)

def initialize(rng, model, *args):
  params_key, dropout_key = random.split(rng)
  @jax.jit
  def init(*args):
    return model.init(*args)
  variables = init({'params': params_key, 'dropout': dropout_key}, *args)
  return variables['params'], variables['batch_stats']

def reconstruction_loss(y, y_pred):
  mae = jnp.mean(jnp.abs(y - y_pred))
  return mae

def binary_cross_entropy_loss(y, y_pred):
  cross_entropy = y * -jax.nn.log_sigmoid(y_pred) # label == 1
  # log(1 - sigmoid(x)) = log_sigmoid(-x)
  cross_entropy += (1 - y) * -jax.nn.log_sigmoid(-y_pred) # label == 0
  return jnp.mean(cross_entropy)

def generator_loss(real_image, fake_image, fake_logit, lambda_: float = 100.):
  gen_loss = binary_cross_entropy_loss(jnp.ones_like(fake_logit), fake_logit)
  gen_loss += lambda_ * reconstruction_loss(real_image, fake_image)
  return gen_loss

def discriminator_loss(real_logit, fake_logit):
  disc_loss = binary_cross_entropy_loss(jnp.ones_like(real_logit), real_logit)
  disc_loss += binary_cross_entropy_loss(jnp.zeros_like(fake_logit), fake_logit)
  return disc_loss

def compute_metrics(real_image, fake_image, real_logit, fake_logit):
  # loss_g = binary_cross_entropy_loss(jnp.ones_like(fake_logit), fake_logit)
  # loss_d = binary_cross_entropy_loss(jnp.zeros_like(fake_logit), fake_logit)
  loss_g = generator_loss(real_image, fake_image, fake_logit)
  loss_d = discriminator_loss(real_logit, fake_logit)
  loss_r = reconstruction_loss(real_image, fake_image)
  metrics = {
      'loss_g': loss_g,
      'loss_d': loss_d,
      'loss_r': loss_r,
  }
  metrics = lax.pmean(metrics, axis_name='batch')
  return metrics

class TrainState(train_state.TrainState):
  batch_stats: Any
  dynamic_scale: optim.DynamicScale

def create_train_state(rng,
                       config: ml_collections.ConfigDict,
                       model,
                       *args,
                       learning_rate_fn=2e-4):
  platform = jax.local_devices()[0].platform
  dynamic_scale = None
  if config.half_precision and platform == 'gpu':
    dynamic_scale = optim.DynamicScale()
  
  params, batch_stats = initialize(rng, model, *args)
  # tx = optax.chain(
  #     # optax.clip_by_global_norm(1.0),
  #     optax.scale_by_adam(b1=0.5, b2=0.999, eps=1e-8),
  #     optax.scale_by_schedule(learning_rate_fn),
  #     optax.scale(-1.0) # scale updates by -1 since optax.apply_updates is additive
  # )
  tx = optax.adam(learning_rate_fn, b1=0.5, b2=0.999, eps=1e-8)
  state = TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx,
      batch_stats=batch_stats,
      dynamic_scale=dynamic_scale)
  return state

cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')

def sync_batch_stats(state):
  return state.replace(batch_stats=cross_replica_mean(state.batch_stats))

def save_checkpoint(states, save_dir):
  if jax.process_index() == 0:
    state = jax.device_get(jax.tree_map(lambda x: x[0], states))
    step = int(state.step)
    checkpoints.save_checkpoint(save_dir, state, step, keep=3)

def restore_checkpoint(state, save_dir):
  return checkpoints.restore_checkpoint(save_dir, state)

def train_step(gen_state, disc_state, batch, dropout_rng, learning_rate=2e-4):
  """Perform a single train step."""

  dropout_rng, new_dropout_rng = random.split(dropout_rng)
  def gen_loss_fn(gen_params, disc_params):
    fake_image, new_gen_vars = gen_state.apply_fn(
        {'params': gen_params, 'batch_stats': gen_state.batch_stats},
        batch['input_image'],
        mutable=['batch_stats'],
        rngs={'dropout': dropout_rng})
    fake_logit, _ = disc_state.apply_fn(
        {'params': disc_params, 'batch_stats': disc_state.batch_stats},
        batch['input_image'], fake_image,
        mutable=['batch_stats'])
    loss = generator_loss(batch['target_image'], fake_image, fake_logit)
    return loss, (new_gen_vars, fake_image, fake_logit)
  def disc_loss_fn(disc_params, fake_image):
    fake_logit, new_disc_vars = disc_state.apply_fn(
        {'params': disc_params, 'batch_stats': disc_state.batch_stats},
        batch['input_image'], fake_image,
        mutable=['batch_stats'])
    real_logit, new_disc_vars = disc_state.apply_fn(
        {'params': disc_params, 'batch_stats': new_disc_vars['batch_stats']},
        batch['input_image'], batch['target_image'],
        mutable=['batch_stats'])
    loss = discriminator_loss(real_logit, fake_logit)
    return loss, (new_disc_vars, real_logit)

  gen_step, disc_step = gen_state.step, disc_state.step
  gen_dynamic_scale, disc_dynamic_scale = gen_state.dynamic_scale, disc_state.dynamic_scale
  gen_lr, disc_lr = learning_rate, learning_rate

  if gen_dynamic_scale:
    gen_grad_fn = gen_dynamic_scale.value_and_grad(gen_loss_fn, has_aux=True,
                                                   axis_name='batch')
    gen_dynamic_scale, is_fin, aux, grads = gen_grad_fn(gen_state.params,
                                                        disc_state.params)
  else:
    gen_grad_fn = jax.value_and_grad(gen_loss_fn, has_aux=True)
    aux, grads = gen_grad_fn(gen_state.params, disc_state.params)
    grads = lax.pmean(grads, axis_name='batch')
  new_gen_vars, fake_image, fake_logit = aux[1]
  new_gen_state = gen_state.apply_gradients(
      grads=grads, batch_stats=new_gen_vars['batch_stats'])
  if gen_dynamic_scale:
    new_gen_state = new_gen_state.replace(
        opt_state=jax.tree_multimap(
            partial(jnp.where, is_fin),
            new_gen_state.opt_state,
            gen_state.opt_state),
        params=jax.tree_multimap(
            partial(jnp.where, is_fin),
            new_gen_state.params,
            gen_state.params))

  if disc_dynamic_scale:
    disc_grad_fn = disc_dynamic_scale.value_and_grad(disc_loss_fn, has_aux=True,
                                                     axis_name='batch')
    dynamic_scale, is_fin, aux, grads = disc_grad_fn(disc_state.params)
  else:
    disc_grad_fn = jax.value_and_grad(disc_loss_fn, has_aux=True)
    aux, grads = disc_grad_fn(disc_state.params, fake_image)
    grads = lax.pmean(grads, axis_name='batch')
  new_disc_vars, real_logit = aux[1]
  new_disc_state = disc_state.apply_gradients(
      grads=grads, batch_stats=new_disc_vars['batch_stats'])
  if disc_dynamic_scale:
    new_disc_state = new_disc_state.replace(
        opt_state=jax.tree_multimap(
            partial(jnp.where, is_fin),
            new_disc_state.opt_state,
            disc_state.opt_state),
        params=jax.tree_multimap(
            partial(jnp.where, is_fin),
            new_disc_state.params,
            disc_state.params))
  
  metrics = compute_metrics(batch['target_image'], fake_image,
                            real_logit, fake_logit)
  metrics['lr_g'] = gen_lr
  metrics['lr_d'] = disc_lr
  if gen_dynamic_scale:
    metrics['scale_g'] = gen_dynamic_scale.scale
  if disc_dynamic_scale:
    metrics['scale_d'] = disc_dynamic_scale.scale
  return new_gen_state, new_disc_state, metrics, new_dropout_rng

def eval_step(gen_state, disc_state, batch, dropout_rng):
  dropout_rng, new_dropout_rng = random.split(dropout_rng)
  fake_image, _ = gen_state.apply_fn(
        {'params': gen_state.params, 'batch_stats': gen_state.batch_stats},
        batch['input_image'],
        mutable=['batch_stats'],
        rngs={'dropout': dropout_rng})
  fake_logit, _ = disc_state.apply_fn(
        {'params': disc_state.params, 'batch_stats': disc_state.batch_stats},
        batch['input_image'], batch['target_image'],
        mutable=['batch_stats'])
  real_logit, _ = disc_state.apply_fn(
        {'params': disc_state.params, 'batch_stats': disc_state.batch_stats},
        batch['input_image'], batch['target_image'],
        mutable=['batch_stats'])
  metrics = compute_metrics(batch['target_image'], fake_image,
                            real_logit, fake_logit)
  return metrics, new_dropout_rng

def prepare_tf_data(xs):
  local_device_count = jax.local_device_count()
  def _prepare(x):
    x = x._numpy()
    return x.reshape((local_device_count, -1) + x.shape[1:])
  return jax.tree_map(_prepare, xs)

def create_input_iter(batch_size, dtype, train, seed):
  ds, n = input_pipeline.create_dataset(batch_size, train, dtype, seed)
  it = map(prepare_tf_data, ds)
  it = jax_utils.prefetch_to_device(it, 2)
  return it, n
  
def train_and_evaluate(config: ml_collections.ConfigDict,
                       save_dir: str, seed: int) -> train_state.TrainState:

  writer = metric_writers.create_default_writer(
      logdir=save_dir, just_logging=bool(jax.process_index()))
  
  base_rng = random.PRNGKey(seed)

  image_size = 256

  if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  
  local_batch_size = config.batch_size // jax.process_count()

  platform = jax.local_devices()[0].platform
  
  if config.half_precision:
    if platform == 'tpu':
      input_dtype = jnp.bfloat16
    else:
      input_dtype = jnp.float16
  else:
    input_dtype = jnp.float32

  train_iter, n_train = create_input_iter(local_batch_size, input_dtype, True, seed)
  eval_iter, n_eval = create_input_iter(local_batch_size, input_dtype, False, seed)

  steps_per_epoch = n_train // config.batch_size

  if config.num_train_steps == -1:
    num_steps = int(steps_per_epoch * config.num_epochs)
  else:
    num_steps = config.num_train_steps

  steps_per_eval = n_eval // config.batch_size
  steps_per_checkpoint = steps_per_epoch * 10

  generator = create_model(models.Generator, config.half_precision)
  discriminator = create_model(models.Discriminator, config.half_precision)

  gen_rng, disc_rng, dropout_rng = random.split(base_rng, 3)
  gen_dummy_input = jnp.ones((1, image_size, image_size, 3), generator.dtype)
  disc_dummy_input_1 = jnp.ones((1, image_size, image_size, 3), discriminator.dtype)
  disc_dummy_input_2 = jnp.ones((1, image_size, image_size, 3), discriminator.dtype)
  gen_state = create_train_state(gen_rng, config, generator, gen_dummy_input)
  disc_state = create_train_state(disc_rng, config, discriminator,
                                  disc_dummy_input_1, disc_dummy_input_2)

  gen_state = jax_utils.replicate(gen_state)
  disc_state = jax_utils.replicate(disc_state)
  train_dropout_rng, eval_dropout_rng = random.split(dropout_rng, 2)
  train_dropout_rngs = random.split(train_dropout_rng, jax.local_device_count())
  eval_dropout_rngs = random.split(eval_dropout_rng, jax.local_device_count())

  p_train_step = jax.pmap(train_step, axis_name='batch')
  p_eval_step = jax.pmap(eval_step, axis_name='batch')

  train_metrics = []
  hooks = []
  if jax.process_index() == 0:
    hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=save_dir)]
  train_metrics_last_t = time.time()
  logging.info('Initial compilation, this might take some minutes...')
  for step, batch in zip(range(num_steps), train_iter):
    gen_state, disc_state, metrics, train_dropout_rngs = p_train_step(
        gen_state, disc_state, batch, dropout_rng=train_dropout_rngs)
    for h in hooks:
      h(step)
    if step == 0:
      logging.info('Initial compilation completed.')

    if config.get('log_every_steps'):
      train_metrics.append(metrics)
      if (step + 1) % config.log_every_steps == 0:
        train_metrics = common_utils.get_metrics(train_metrics)
        summary = {
            f'train_{k}': v
            for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
        }
        summary['steps_per_second'] = config.log_every_steps / (
            time.time() - train_metrics_last_t)
        writer.write_scalars(step + 1, summary)
        train_metrics = []
        train_metrics_last_t = time.time()
    
    if (step + 1) % steps_per_epoch == 0:
      epoch = step // steps_per_epoch
      eval_metrics = []

      gen_state = sync_batch_stats(gen_state)
      disc_state = sync_batch_stats(disc_state)
      for _ in range(steps_per_eval):
        eval_batch = next(eval_iter)
        metrics, eval_dropout_rngs = p_eval_step(
            gen_state, disc_state, eval_batch, dropout_rng=eval_dropout_rngs)
        eval_metrics.append(metrics)
      eval_metrics = common_utils.get_metrics(eval_metrics)
      summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
      logging.info('eval epoch: %d, loss_g: %.4f, loss_d: %.4f, loss_r: %.4f',
                   epoch, summary['loss_g'], summary['loss_d'], summary['loss_r'])
      writer.write_scalars(
          step + 1, {f'eval_{key}': val for key, val in summary.items()})
      writer.flush()
    
    if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
      gen_state = sync_batch_stats(gen_state)
      save_checkpoint(gen_state, save_dir)

  # Wait until computations are done
  random.normal(random.PRNGKey(0), ()).block_until_ready()
  return gen_state, disc_state
  
