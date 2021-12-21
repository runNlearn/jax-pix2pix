import ml_collections

def get_config():
  config = ml_collections.ConfigDict()

  config.batch_size = 1
  config.num_epochs = 200.0
  config.num_train_steps = -1
  config.log_every_steps = 100

  config.half_precision = False

  return config
