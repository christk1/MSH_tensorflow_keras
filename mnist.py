import tensorflow as tf
import misc
import argparse
from os import makedirs
from os.path import exists, join, isfile
import numpy as np
from tensorboard.plugins import projector

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

parser = argparse.ArgumentParser()
args = parser.parse_args()
json_path = "params.json"
assert isfile(json_path), "No json configuration file found at {}".format(json_path)
params = misc.Params(json_path)

if params.data_format is None:
    data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

if not exists(params.log_dir):
    makedirs(params.log_dir)

# keep track of the epoch
try:
    with open("epochs_ckpts.txt", "r") as f:
        epoch = int(f.read())
except IOError:
    epoch = 0
    print("Started from epoch 0")
epoch_write = epoch + params.epochs
with open("epochs_ckpts.txt", "w") as f:
    f.write(str(epoch_write))


class ArcLoss(tf.keras.layers.Layer):

    def __init__(self, n_classes, m=0.5, s=64., **kwargs):
        self.num_classes = n_classes
        self.m = m
        self.s = s
        super(ArcLoss, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][-1], self.num_classes),
                                      initializer=tf.random_normal_initializer(stddev=0.01),
                                      trainable=True)
        # input_shape[0] contains the batch
        super(ArcLoss, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inps):
        # emb (N x embs), labels (N, 10) = labels one hot
        emb, labels = inps

        # normalize feature
        emb = tf.nn.l2_normalize(emb, axis=1) * self.s  # (n, 512)
        # normalize weights
        W = tf.nn.l2_normalize(self.kernel, axis=0)  # (512, 10)

        fc7 = tf.matmul(emb, W)  # n x 10
        # pick elements along axis 1
        zy = tf.reduce_max(input_tensor=tf.multiply(fc7, labels), axis=1)  # (n, 1)

        cos_t = zy / self.s
        t = tf.acos(cos_t)
        body = tf.cos(t + self.m)
        new_zy = body * self.s
        diff = new_zy - zy
        diff = tf.expand_dims(diff, 1)
        body = tf.multiply(labels, diff)
        fc7 = fc7 + body

        return fc7

    def get_config(self):
        config = super(ArcLoss, self).get_config()
        config.update({'num_classes': self.num_classes, 'm': self.m, 's': self.s})
        return config

    def compute_output_shape(self, input_shape):
        return None, self.num_classes


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# input image dimensions
img_rows, img_cols = 32, 32
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2)), mode='constant')
x_test = np.squeeze(x_test)
x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), mode='constant')
x_train = np.squeeze(x_train)

if data_format == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# global standardizing
x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)
x_train = (x_train - x_train_mean) / x_train_std
x_test = (x_test - x_train_mean) / x_train_std
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# save class labels to disk to color data points in TensorBoard accordingly
with open(join(params.log_dir, 'metadata.tsv'), 'w') as f:
    np.savetxt(f, y_test)

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, params.num_classes)
y_test = tf.keras.utils.to_categorical(y_test, params.num_classes)

input_tensor = tf.keras.layers.Input(shape=input_shape)
base_model = tf.keras.applications.VGG16(input_tensor=input_tensor, include_top=False, weights=None)

x = base_model.output

if params.use_arcloss:
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(params.embedding_size, activation='relu', name='feats0')(x)
    x = tf.keras.layers.Dense(params.embedding_size, name='features')(x)
    aux_input = tf.keras.Input(shape=(params.num_classes,))
    predictions = ArcLoss(n_classes=params.num_classes, name='arclosslayer')([x, aux_input])
    predictions = tf.keras.activations.softmax(predictions)
    model = tf.keras.models.Model(inputs=[base_model.input, aux_input], outputs=predictions)
else:
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(params.embedding_size, activation='relu', name='feats0')(x)
    x = tf.keras.layers.Dense(params.embedding_size, activation='relu', name='features')(x)
    predictions = tf.keras.layers.Dense(params.num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

Weights_path = join(params.log_dir, 'weights.hdf5')
if isfile(Weights_path):
    if params.fine_tune:
        # freeze early layers
        for layer in base_model.layers[:-2]:
            layer.trainable = False
    model.load_weights(Weights_path, by_name=True)
    print("weights loaded.")
else:
    print("error in loading weights.")

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(lr=params.learning_rate),
              metrics=['accuracy'])

# print layers
for i, layer in enumerate(model.layers):
    print(i, layer.name)

if params.use_arcloss:
    x_inps = [x_train, y_train]
    x_test_inps = [x_test, y_test]
else:
    x_inps = x_train
    x_test_inps = x_test

callbacks = misc.get_callbacks(log_dir=params.log_dir)

model.fit(x_inps, y_train,
          batch_size=params.batch_size,
          epochs=epoch_write,
          initial_epoch=epoch,
          callbacks=callbacks,
          verbose=1,
          validation_data=(x_test_inps, y_test))
score = model.evaluate(x_test_inps, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# the following code replaces embedding visualization of tf.keras.callbacks.TensorBoard
# because it does not save checkpoints as in keras.callbacks.TensorBoard
side_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('features').output)
feats = side_model.predict(x_test_inps)
feats /= np.linalg.norm(feats, axis=1, keepdims=True)

nofembs = feats.shape[0]
"""write checkpoints with embeddings"""
features = tf.Variable(feats, name='features')
with tf.Session() as sess:
    saver = tf.compat.v1.train.Saver([features])

    sess.run(features.initializer)
    saver.save(sess, join(params.log_dir, 'images_10_classes.ckpt'))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = features.name
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(tf.compat.v1.summary.FileWriter(params.log_dir), config)
