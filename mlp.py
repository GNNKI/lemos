"""
Defines a lightweight multi‑layer perceptron (MLP) class for use in
the graph colouring neural network. The original code relied on
`tf.layers.Dense`, which has been removed from TensorFlow 2.x. To
preserve compatibility while enabling GPU execution under modern
TensorFlow versions, this module now uses the TF 1.x compatibility
interface (`tf.compat.v1`) and disables v2 behaviour. Dense layers are
created via `tf.compat.v1.layers.Dense`, which mirrors the behaviour
of `tf.layers.Dense` in TF 1.x.
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Mlp(object):
  def __init__(
    self,
    layer_sizes,
    output_size = None,
    activations = None,
    output_activation = None,
    use_bias = True,
    kernel_initializer = None,
    bias_initializer = tf.zeros_initializer(),
    kernel_regularizer = None,
    bias_regularizer = None,
    activity_regularizer = None,
    kernel_constraint = None,
    bias_constraint = None,
    trainable = True,
    name = None,
    name_internal_layers = True
  ):
    """
    Construct a stack of dense layers.

    A list of hidden layer sizes (`layer_sizes`) is provided, and an optional
    `output_size` can be appended as a final layer. Activation functions
    can be supplied either as a single callable (which will be reused for
    each hidden layer) or as a list. If `output_size` is specified and
    `output_activation` is provided, the output activation will be
    appended to the list of activations automatically.

    Parameters are documented in the module-level docstring. Note that
    TensorFlow 2.x has removed `tf.layers.Dense`; here we use
    `tf.compat.v1.layers.Dense` to create the layers under TF 2.x. See
    util.py and graphnn.py for details on how the overall project has
    been ported to TensorFlow 2.x.
    """
    self.layers = []
    internal_name = None
    # If object isn't a list, assume it is a single value that will be repeated for all values
    if not isinstance(activations, list):
      activations = [activations for _ in layer_sizes]
    # If there is one specifically for the output, add it to the list of layers to be built
    if output_size is not None:
      layer_sizes = layer_sizes + [output_size]
      activations = activations + [output_activation]
    for i, params in enumerate(zip(layer_sizes, activations)):
      size, activation = params
      if name_internal_layers:
        internal_name = f"{name}_MLP_layer_{i + 1}"
      # Create a dense layer using the TF 1.x compatible API.
      new_layer = tf.compat.v1.layers.Dense(
        size,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=internal_name
      )
      self.layers.append(new_layer)
  #end __init__
  
  def __call__( self, inputs, *args, **kwargs ):
    outputs = [ inputs ]
    for layer in self.layers:
      outputs.append( layer( outputs[-1] ) )
    #end for
    return outputs[-1]
  #end __call__
#end Mlp
