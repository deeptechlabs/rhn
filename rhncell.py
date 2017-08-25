from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.util import nest

RNNCell = tf.nn.rnn_cell.RNNCell

class RHNCell(RNNCell):
  """Variational Recurrent Highway Layer

  Reference: https://arxiv.org/abs/1607.03474
  """

  def __init__(self, num_units, in_size, is_training, depth=3, forget_bias=None):
    self._num_units = num_units
    self._in_size = in_size
    self.is_training = is_training
    self.depth = depth
    self.forget_bias = forget_bias

  @property
  def input_size(self):
    return self._in_size

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    current_state = state[0]
    noise_i = state[1]
    noise_h = state[2]
    for i in range(self.depth):
      with tf.variable_scope('h_'+str(i)):
        if i == 0:
          h = tf.tanh(linear([inputs * noise_i, current_state * noise_h], self._num_units, True))
        else:
          h = tf.tanh(linear([current_state * noise_h], self._num_units, True))
      with tf.variable_scope('t_'+str(i)):
        if i == 0:
          t = tf.sigmoid(linear([inputs * noise_i, current_state * noise_h], self._num_units, True, self.forget_bias))
        else:
          t = tf.sigmoid(linear([current_state * noise_h], self._num_units, True, self.forget_bias))
      current_state = (h - current_state)* t + current_state

    return current_state, [current_state, noise_i, noise_h]

def linear(args, output_size, bias, bias_start=None, scope=None):
  """
  This is a slightly modified version of _linear used by Tensorflow rnn.
  The only change is that we have allowed bias_start=None.

  Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with vs.variable_scope(scope or "Linear"):
    matrix = vs.get_variable(
        "Matrix", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = math_ops.matmul(args[0], matrix)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), matrix)
    if not bias:
      return res
    elif bias_start is None:
      bias_term = vs.get_variable("Bias", [output_size], dtype=dtype)
    else:
      bias_term = vs.get_variable("Bias", [output_size], dtype=dtype,
                                  initializer=tf.constant_initializer(bias_start, dtype=dtype))
  return res + bias_term