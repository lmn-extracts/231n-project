import tensorflow as tf
from dataset_helper import *
from modules import *
import numpy as np
from data_provider import TFRecordReader
from data_utils import *

# Debugging
import cv2

def main(unused_args):
    reader = TFRecordReader()
    images, anno = reader._read_feature('train.tfrecords')

    inputs, labels = tf.train.shuffle_batch([images, anno], batch_size=32, capacity=1000+2*32, min_after_dequeue=100, num_threads=1)

    with tf.variable_scope('crnn', reuse=False):
        model_output, decoded, logits = CRNN(inputs, hidden_size=256)

    loss = tf.reduce_mean(tf.nn.ctc_loss(labels=labels, inputs=model_output, sequence_length=23 * np.ones(32), ignore_longer_outputs_than_inputs=True))
    # edit_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded, tf.int32), labels))


    global_step = tf.Variable(0, name='global_step', trainable=False)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(1000):
            _,cost, preds, gt_labels, iter_num = sess.run([optimizer, loss, decoded, labels, global_step])
            # print (model_output.shape)
            print (cost)
           

        coord.request_stop()
        coord.join(threads)

    return

if __name__ == '__main__':
    tf.app.run()