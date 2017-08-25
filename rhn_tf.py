from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.util import nest

from rhncell import *

RNNCell = tf.nn.rnn_cell.RNNCell


class Model(object):
  """A Variational RHN model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.depth = depth = config.depth
    self.size = size = config.hidden_size
    self.num_layers = num_layers = config.num_layers
    vocab_size = config.vocab_size
    if vocab_size < self.size and not config.tied:
      in_size = vocab_size
    else:
      in_size = self.size
    self.in_size = in_size
    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._noise_x = tf.placeholder(tf.float32, [batch_size, num_steps, 1])
    self._noise_i = tf.placeholder(tf.float32, [batch_size, in_size, num_layers])
    self._noise_h = tf.placeholder(tf.float32, [batch_size, size, num_layers])
    self._noise_o = tf.placeholder(tf.float32, [batch_size, 1, size])

    with tf.device("/gpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, in_size])
      inputs = tf.nn.embedding_lookup(embedding, self._input_data) * self._noise_x

    outputs = []
    self._initial_state = [0] * self.num_layers
    state = [0] * self.num_layers
    self._final_state = [0] * self.num_layers
    for l in range(config.num_layers):
      with tf.variable_scope('RHN' + str(l)):
        cell = RHNCell(size, in_size, is_training, depth=depth, forget_bias=config.init_bias)
        self._initial_state[l] = cell.zero_state(batch_size, tf.float32)
        state[l] = [self._initial_state[l], self._noise_i[:, :, l], self._noise_h[:, :, l]]
        for time_step in range(num_steps):
          if time_step > 0:
            tf.get_variable_scope().reuse_variables()
          (cell_output, state[l]) = cell(inputs[:, time_step, :], state[l])
          outputs.append(cell_output)
        inputs = tf.stack(outputs, axis=1)
        outputs = []

    output = tf.reshape(inputs * self._noise_o, [-1, size])
    softmax_w = tf.transpose(embedding) if config.tied else tf.get_variable("softmax_w", [size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    logits = tf.reshape(logits, [-1, 1, vocab_size])

    loss = tf.contrib.seq2seq.sequence_loss(
      logits,
      self._targets,
      tf.ones([batch_size, num_steps]),
      average_across_timesteps = False, average_across_batch = False)
    self._final_state = [s[0] for s in state]
    pred_loss = tf.reduce_sum(loss) / batch_size
    self._cost = cost = pred_loss
    if not is_training:
      return
    tvars = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars])
    self._cost = cost = pred_loss + config.weight_decay * l2_loss

    self._lr = tf.Variable(0.0, trainable=False)
    self._nvars = np.prod(tvars[0].get_shape().as_list())
    print(tvars[0].name, tvars[0].get_shape().as_list())
    
    for var in tvars[1:]:
      sh = var.get_shape().as_list()
      print(var.name, sh)
      self._nvars += np.prod(sh)
    
    print(self._nvars, 'total variables')
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)

    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def noise_x(self):
    return self._noise_x

  @property
  def noise_i(self):
    return self._noise_i

  @property
  def noise_h(self):
    return self._noise_h

  @property
  def noise_o(self):
    return self._noise_o

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def nvars(self):
    return self._nvars

