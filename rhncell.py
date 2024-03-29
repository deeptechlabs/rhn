from __future__ import absolute_import, division, print_function
import tensorflow as tf

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

    '''
    inner loop of feature refinering
    l=0 to l=4 starts here
    the default settings using the hyperparameters is set with depth
    depth = l = 4
    '''
    
    for i in range(self.depth):
      with tf.variable_scope('h_'+str(i)):
        # modify this line to accomodate for the input cell
        if i == 0:
        # the if else is the indicator function
        # basically this is where h is computed based on the current state and the input state
          h = tf.tanh(linear_inputcell([inputs * noise_i, current_state * noise_h], self._num_units, True))
        else:
        # the if else is the indicator function
        # basically this is where h is computed based on the current state
          h = tf.tanh(linear_non_inputcell([current_state * noise_h], self._num_units, True))
      with tf.variable_scope('t_'+str(i)):
        if i == 0:
          t = tf.sigmoid(linear_inputcell([inputs * noise_i, current_state * noise_h], self._num_units, True, self.forget_bias))
        #else:
          #t = tf.sigmoid(linear([current_state * noise_h], self._num_units, True, self.forget_bias))
      # we start by modifying this part
      # since we want to return the current state that would be 
      if i == 0:
        current_state = (h + current_state)
      else:
        current_state = (h - current_state)* t + current_state

    return current_state, [current_state, noise_i, noise_h]

def linear_inputcell(args, output_size, bias, bias_start=None, scope=None):
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
  
  This is still needed for c, but not for h in i>0
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
    weights_input = vs.get_variable(
        "weights_input", [total_arg_size, output_size], dtype=dtype)
      
    #res = math_ops.matmul(args[0], weights1)
    #else:
    
    res_input = math_ops.matmul(array_ops.concat(args, 1), weights_input)
    
    if not bias:
      return res_input
    elif bias_start is None:
      bias_input = vs.get_variable("Bias_input", [output_size], dtype=dtype)
    else:
      bias_input = vs.get_variable("Bias_input", [output_size], dtype=dtype,
                                  initializer=tf.constant_initializer(bias_start, dtype=dtype))
  return res_input + bias_input

def linear_non_inputcell(args, output_size, bias, bias_start=None, scope=None):
  """
  Used when i > 0

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
    weights_noninput = vs.get_variable(
                                "weights_noninput", 
                                [total_arg_size, output_size], 
                                dtype=dtype)
    
    # this is a dot product operation
    #  args[0] is the input vector
    
    res_noninput = math_ops.matmul(args[0], weights_noninput)
    
    if not bias:
      return res_noninput
    elif bias_start is None:
      bias_term_noninput = vs.get_variable("bias_noninput", 
                                [output_size], 
                                dtype=dtype)
    else:
      bias_term_noninput = vs.get_variable("bias_noninput", 
                                [output_size], 
                                dtype=dtype,
                                initializer=tf.constant_initializer(bias_start, dtype=dtype))
  return res_noninput + bias_term_noninput
