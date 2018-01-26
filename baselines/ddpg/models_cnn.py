import tensorflow as tf
import tensorflow.contrib as tc
from baselines.ddpg import models


class ActorCNN(models.Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True):
        super(ActorCNN, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.conv2d(x, 32, 4, 2, 'SAME')
            x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x) # 16 x 16 x 32

            x = tf.layers.conv2d(x, 32, 4, 2, 'SAME')
            x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x) # 8 x 8 x 32

            x = tf.layers.conv2d(x, 32, 4, 2, 'SAME')
            x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)  # 4 x 4 x 32

            x = tf.reshape(x, [-1, 4*4*32])

            x = tf.layers.dense(x, 200)
            x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 200)
            x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, self.nb_actions,
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-4, maxval=3e-4))
            x = tf.nn.tanh(x)
        return x



class CriticCNN(models.Model):
    def __init__(self, name='critic', layer_norm=True):
        super(CriticCNN, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            #x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

            x = obs
            x = tf.layers.conv2d(x, 32, 4, 2, 'SAME')
            x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)  # 16 x 16 x 32

            x = tf.layers.conv2d(x, 32, 4, 2, 'SAME')
            x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)  # 8 x 8 x 32

            x = tf.layers.conv2d(x, 32, 4, 2, 'SAME')
            x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)  # 4 x 4 x 32

            x = tf.reshape(x, [-1, 4 * 4 * 32])
            x = tf.concat([x, action], axis=1)

            x = tf.layers.dense(x, 200)
            x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 200)
            x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1,
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-4, maxval=3e-4))
            x = tf.nn.tanh(x)
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars


