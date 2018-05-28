'''
Contains all modules necessary to implement the Recognizer Model
'''

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from data_utils import mprint
import numpy as np

def ConvolutionalNet(inputs):
    '''
    Arguments:
        - inputs: Dimension (N x H x W x C)
    Returns:
        - output: Dimension ()
    '''
    conv1 = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

    conv3_1 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
    conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3_2, pool_size=[2,1], strides=[2,1])

    conv4 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[3,3], padding='same')
    norm4 = tf.layers.batch_normalization(inputs=conv4, axis=-1)
    out4 = tf.nn.relu(norm4)

    conv5 = tf.layers.conv2d(inputs=out4, filters=512, kernel_size=[3,3], padding='same')
    norm5 = tf.layers.batch_normalization(inputs=conv5, axis=-1)
    out5 = tf.nn.relu(norm5)
    pool5 = tf.layers.max_pooling2d(inputs=out5, pool_size=[2,1], strides=[2,1])

    conv6 = tf.layers.conv2d(inputs=pool5, filters=512, kernel_size=[2,2], padding='valid', activation=tf.nn.relu)

    mprint('conv6', conv6.get_shape())

    return conv6

def StackedRNN(inputs, hidden_size):
    fw_cells = [rnn_cell.LSTMCell(hidden_size), rnn_cell.LSTMCell(hidden_size)]
    bw_cells = [rnn_cell.LSTMCell(hidden_size), rnn_cell.LSTMCell(hidden_size)]
    stacked_out, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cells, bw_cells, inputs, dtype=tf.float32)

    _, W, h = inputs.get_shape().as_list() # N x W x 2h; N returns ? None when using dataset batch
    stacked_out = tf.reshape(stacked_out, [-1, h])  # NW x 2h

    output = tf.layers.dense(stacked_out, units=53)
    output = tf.reshape(output, [-1,W,53]) # N x W x 52
    # output = tf.transpose(output, [1,0,2]) # W x H x 52; Necessary for input to CTC Loss
    mprint('RNN output', output.get_shape())

    return output


def MapToSequence(inputs):
    '''
    Arguments:
        - inputs: Dimension N x H x W x C

    Returns:
        - Channel slice along the height to give output of dimension (N x W x HC)
    '''
    output = tf.squeeze(inputs, axis=1)
    mprint('map2seq inputs', inputs.get_shape())
    mprint('map2seq output', output.get_shape())
    return output

def CRNN(inputs, hidden_size, max_char=25, batch_size=32):
    with tf.variable_scope('cnn'):
        conv_out = ConvolutionalNet(inputs) # N x 1 x W x C 

    with tf.variable_scope('map2seq'):
        map2seq_out = MapToSequence(conv_out) # N x 1 x W x C  --> # N x W x C

    # with tf.variable_scope('rnn_1', reuse=False):
    #     rnn1_out = RNNLayer(map2seq_out, hidden_size, [max_char] * tf.shape(inputs)[0])

    # with tf.variable_scope('rnn_1', reuse=False):
    #     rnn2_out = RNNLayer(rnn1_out, 53, [max_char] * tf.shape(inputs)[0])

    with tf.variable_scope('stacked_rnn'):
        rnn2_out = StackedRNN(map2seq_out, hidden_size) # (N X W x 52)

    with tf.variable_scope('ctc_beam_search'):
        model_output = tf.transpose(rnn2_out, perm=[1,0,2]) # Dim: (W x N X 52) --> (Time x Batch_Size x Num_Classes)
        decoded, logits = tf.nn.ctc_beam_search_decoder(model_output, 23 * np.ones(batch_size), merge_repeated=True)
        mprint('conv_output', model_output.get_shape())

    return model_output, decoded, logits 