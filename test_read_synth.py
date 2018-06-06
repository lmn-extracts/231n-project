import tensorflow as tf
import matplotlib.pyplot as plt
import os
from data_utils import *
from dataset_helper import *
import argparse

data_path = '/Users/afassa/Documents/Ane-lg-nonpersonal/learning/cs231n/assignments/project-git/data/SynthTf10kTestNewProvider'
data_file = os.path.join(data_path,'val.tfrecords')  # address to save the hdf5 file

tf_format = 'JPG'
#tf_format = 'RAW'

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f')
parser.add_argument('--tf_format', '-tf')
args = parser.parse_args()

full_file_path = data_file if args.file is None else args.file
tf_format = tf_format if args.tf_format is None else args.tf_format

batch_size = 5
with tf.Graph().as_default():
    image_batch, label_batch = input_fn(full_file_path, train=True, batch_size=batch_size, buffer_size=2048,
                                        tf_format=tf_format)

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        for batch in range(10):

            images, labels = sess.run([image_batch, label_batch])
            dense_label = sess.run(tf.sparse_to_dense(labels.indices, labels.dense_shape, labels.values, default_value=-1))
            print("DENSE LABEL",dense_label)
            for i in range(batch_size):
                image_i = images['image'][i]
                print("DENSE LABEL i", dense_label[i])
                label = [idx2char(c) for c in dense_label[i]]
                print("***Label", label)
                image_i = image_i.astype(np.uint8)
                print("image shape", image_i.shape)
                plt.imshow(image_i)
                plt.title(label)
                plt.show()

