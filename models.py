from typing import Any, Callable, Sequence, Tuple, Union, Iterable
from functools import partial

from flax import linen as nn
import jax
import jax.numpy as jnp

ModuleDef = Any

class ConvBlock(nn.Module):
  filters: int
  conv: ModuleDef
  norm: Union[None, ModuleDef]
  act: Union[None, Callable]
  strides: Tuple[int, int]
  padding: Union[str, Iterable[Tuple[int, int]]] = 'SAME'
  dropout: bool = False

  @nn.compact
  def __call__(self, x):
    x = self.conv(self.filters, (4, 4), self.strides, self.padding)(x)
    if self.norm:
      x = self.norm()(x)
    if self.dropout:
      x = nn.Dropout(0.5, deterministic=False)(x)
    if self.act:
      x = self.act(x)
    return x


class Generator(nn.Module):
  filter_multipliers: Sequence[int] = (1, 2, 4, 8, 8, 8, 8, 8)
  base_filters: int = 64
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, input_image):
    down = partial(nn.Conv, use_bias=False, dtype=self.dtype)
    up   = partial(nn.ConvTranspose, use_bias=False, dtype=self.dtype)
    norm = partial(nn.BatchNorm, use_running_average=False, dtype=self.dtype)

    xs = []
    x = input_image
    for i, multiplier in enumerate(self.filter_multipliers):
      filters = multiplier * self.base_filters
      _norm = None if i == 0 else norm
      act = partial(nn.leaky_relu, negative_slope=0.2)
      x = ConvBlock(filters,
                    conv=down,
                    norm=_norm,
                    act=act,
                    strides=(2, 2),
                    padding='SAME',
                    dropout=False)(x)
      xs.append(x)
    xs = reversed(xs[:-1])
    filter_multipliers = reversed(self.filter_multipliers[:-1])
    for i, (multiplier, _x) in enumerate(zip(filter_multipliers, xs)):
      filters = multiplier * self.base_filters
      dropout = i < 3
      act = nn.relu
      x = ConvBlock(filters,
                    conv=up,
                    norm=norm,
                    act=act,
                    strides=(2, 2),
                    padding='SAME',
                    dropout=dropout)(x)
      x = jnp.concatenate((x, _x), axis=-1)
    x = ConvBlock(3, up, norm, act=nn.tanh, strides=(2, 2), padding='SAME')(x)
    return x
      

class Discriminator(nn.Module):
  filter_multipliers: Sequence[int] = (1, 2, 4, 8)
  base_filters: int = 64
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, input_image, target_image):
    down = partial(nn.Conv, use_bias=False, dtype=self.dtype)
    norm = partial(nn.BatchNorm, use_running_average=False, dtype=self.dtype)

    x = jnp.concatenate((input_image, target_image), axis=-1)
    for i, multiplier in enumerate(self.filter_multipliers):
      filters = multiplier * self.base_filters
      _norm = None if i == 0 else norm
      act = partial(nn.leaky_relu, negative_slope=0.2)
      strides = (2, 2) if i < len(self.filter_multipliers) - 1 else (1, 1)
      padding = 'SAME' if i < len(self.filter_multipliers) - 1 else [(1, 1), (1, 1)]
      x = ConvBlock(filters,
                    conv=down,
                    norm=_norm,
                    act=act,
                    strides=strides,
                    padding=padding,
                    dropout=False)(x)
    x = ConvBlock(1, down, norm,
                  act=None,
                  strides=(1, 1),
                  padding=[(1, 1), (1, 1)],
                  dropout=False)(x)
    return x


class Pix2Pix(nn.Module):
  dtype: Any = jnp.float32

  def setup(self):
    self.generator = Generator(dtype=self.dtype)
    self.discriminator = Discriminator(dtype=self.dtype)

  def __call__(self, x, y, train=False):
    y_fake = self.generate_fake(x)
    logit_fake = self.discriminator(x, y_fake)
    logit_real = self.discriminator(x, y)
    return y_fake, logit_real, logit_fake

  def generate_fake(self, input_image):
    fake_image = self.generator(input_image)
    return fake_image
