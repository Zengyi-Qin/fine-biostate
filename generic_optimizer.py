import numpy as np
import logging
import sys
import tensorflow as tf

def build_train_op(config, loss, learning_rate):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """

    with tf.name_scope('training'):

        if config['opt'] == 'Adam':
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif config['opt'] == 'SGD':
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif config['opt'] == 'Momentum':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
        else:
            raise ValueError('Unrecognized opt type')
        all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
        var_list = [var for var in all_variables if "regression" in var.name]

        if config['train_feature_space']:
            grads_and_vars = opt.compute_gradients(loss)
        else:
            grads_and_vars = opt.compute_gradients(loss, var_list)
   
        if config['clip_norm'] > 0:
            grads, tvars = zip(*grads_and_vars)
            clip_norm = config["clip_norm"]
            clipped_grads, norm = tf.clip_by_global_norm(grads, clip_norm)
            grads_and_vars = zip(clipped_grads, tvars)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            train_op = opt.apply_gradients(grads_and_vars)

    return train_op
