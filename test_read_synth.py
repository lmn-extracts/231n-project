import tensorflow as tf
import matplotlib.pyplot as plt
import os
from data_utils import *
from dataset_helper import *

data_dir = '/Users/afassa/Documents/Ane-lg-nonpersonal/learning/cs231n/assignments/project-git/data/SynthTf2'
data_path = os.path.join(data_dir,'train.tfrecords')  # address to save the hdf5 file

batch_size = 5
with tf.Graph().as_default():
    image_batch, label_batch = input_fn(data_path, train=True, batch_size=batch_size, buffer_size=2048)

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
                #image = image['image'][0]
                image_i = image_i.astype(np.uint8)
                print("image shape", image_i.shape)
                plt.imshow(image_i)
                plt.title(label)
                plt.show()

