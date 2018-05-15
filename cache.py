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

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(3):
            testip = inputs[0,:,:,:]
            testip = tf.cast(testip, np.uint8)
            testlabs = tf.sparse_to_dense(labels.indices, labels.dense_shape, labels.values, default_value=-1)
            testip, testlabs = sess.run([testip, testlabs])
            testlabs = nd_array_to_labels(testlabs)
            print (testlabs[0])
            cv2.imshow('im', testip)
            cv2.waitKey(0)
            print (testip.shape)

        coord.request_stop()
        coord.join(threads)

    return

if __name__ == '__main__':
    tf.app.run()