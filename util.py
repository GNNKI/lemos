"""
Utility functions for saving and loading model weights. TensorFlow 1.x
provided a `tf.train.Saver` class to save and restore variables.
TensorFlow 2.x removed this from the default API, but it remains
available via the v1 compatibility interface. This module uses
`tf.compat.v1` to access the Saver and related functionality. These
functions enable checkpoint management under modern CUDA-enabled
TensorFlow installations.
"""

import time, sys, os, random
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def load_weights(sess, path, scope=None):
  if os.path.exists(path):
    # Restore saved weights
    print("Restoring saved model ... ")
    # Create model saver
    if scope is None:
      saver = tf.compat.v1.train.Saver()
    else:
      saver = tf.compat.v1.train.Saver(
        var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=scope)
      )
    saver.restore(sess, f"{path}/model.ckpt")
  else:
    raise Exception('Path does not exist!')
  #end if
#end

def save_weights(sess, path, scope=None):
  # Create /tmp/ directory to save weights
  if not os.path.exists(path):
    os.makedirs(path)
  #end if
  # Create model saver
  if scope is None:
    saver = tf.compat.v1.train.Saver()
  else:
    saver = tf.compat.v1.train.Saver(
      var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    )
  saver.save(sess, f"{path}/model.ckpt")
  print(f"MODEL SAVED IN PATH: {path}\n")
#end
