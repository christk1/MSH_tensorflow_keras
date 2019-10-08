import tensorflow as tf
import numpy as np
import json
from os.path import join


def get_callbacks(log_dir):
    callbacks = list()

    modelckpt = tf.keras.callbacks.ModelCheckpoint(filepath=join(log_dir, 'weights.hdf5'), verbose=1,
                                                   save_best_only=True, save_weights_only=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    callbacks.append(modelckpt)
    callbacks.append(tensorboard_callback)

    return callbacks


def normalize_meanstd(a, axis=None):
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    print(mean.shape)
    std = np.sqrt(((a - mean) ** 2).mean(axis=axis, keepdims=True))
    image_batch = (a - mean) / std
    return image_batch, mean, std


class Params:
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__
