import os
import tensorflow as tf

_URL = 'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz'
_PATH_TO_ZIP = tf.keras.utils.get_file('facades.tar.gz',
                                      origin=_URL,
                                      extract=True)
_PATH = os.path.join(os.path.dirname(_PATH_TO_ZIP), 'facades/')

def load_image(path):
  jpg = tf.io.read_file(path)
  img = tf.io.decode_jpeg(jpg, channels=3, dct_method='INTEGER_ACCURATE')
  return img

def split_image(img):
  w = tf.shape(img)[1] // 2
  real_img = img[..., :w, :]
  input_img = img[..., w:, :]
  return input_img, real_img

def random_jitter(img, resizing_size, target_size, seed, seed2):
  seed = (seed, seed2)
  img = tf.image.resize(img, (resizing_size, resizing_size))
  img = tf.image.stateless_random_crop(img, (target_size, target_size, 3), seed=seed)
  img = tf.image.stateless_random_flip_left_right(img, seed=seed)
  img = tf.image.stateless_random_flip_up_down(img, seed=seed)
  return img

# [0, 255] -> [-1, 1]
def normalize(img):
  img = tf.cast(img, tf.float32)
  img = img / 127.5 - 1
  return img

@tf.function
def preprocess_train(path, resizing_size, target_size, seed, seed2, dtype):
  img = load_image(path)
  input_img, target_img = split_image(img)
  input_img = random_jitter(input_img, resizing_size, target_size, seed, seed2)
  target_img = random_jitter(target_img, resizing_size, target_size, seed, seed2)
  input_img, target_img = normalize(input_img), normalize(target_img)
  return input_img, target_img

@tf.function
def preprocess_test(path, dtype):
  img = load_image(path)
  input_img, target_img = split_image(img)
  input_img, target_img = normalize(input_img), normalize(target_img)
  return input_img, target_img

def create_dataset(batch_size, train, dtype, seed):
  if train:
    tf.random.set_seed(seed)
    pattern = _PATH + 'train/*.jpg'
  else:
    pattern = _PATH + 'val/*.jpg'

  def preprocess(path, seed2=None):
    resizing_size = 286
    target_size = 256
    if train:
      input_image, target_image = preprocess_train(
          path, resizing_size, target_size, seed, seed2, dtype)
    else:
      input_image, target_image = preprocess_test(path, dtype)
    return {'input_image': input_image, 'target_image': target_image}
  
  ds = tf.data.Dataset.list_files(pattern, shuffle=train, seed=seed)
  num_data = len(ds)
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  if train:
    ds = ds.repeat()
    ds = tf.data.Dataset.zip((ds, tf.data.Dataset.random(seed)))
    ds = ds.shuffle(16 * batch_size, seed=seed)
  
  ds = ds.map(preprocess, num_parallel_calls=-1)
  ds = ds.batch(batch_size, drop_remainder=True)

  if not train:
    ds = ds.repeat()
  
  ds = ds.prefetch(10)
  return ds, num_data
