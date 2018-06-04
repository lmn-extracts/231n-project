import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_path = 'train.tfrecords'  # address to save the hdf5 file

with tf.Session() as sess:
    feature = {'image': tf.VarLenFeature(tf.string),
               'label': tf.VarLenFeature(tf.string)}
    # Create a list of filenames and pass it to a queuesl
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)

    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

    image = features['image'].bytes_list.value[0]
    label = features['label'].bytes_list.value[0]

    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)

    # Initialize all global and local global_variables_initializer
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for batch_index in range(2):
        img, lbl = sess.run([images, labels])
        print (img)
        print (lbl)

    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
    sess.close()