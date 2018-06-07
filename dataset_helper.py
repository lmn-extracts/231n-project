import tensorflow as tf
from data_utils import *


def get_image_shape():
    img_shape = [32, 100, 1]
    return img_shape

def parse_raw(serialized):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    # It is a bit awkward that this needs to be specified again,
    # because it could have been written in the header of the
    # TFRecords file instead.

    #MJSynth code
    features = tf.parse_single_example(serialized,
                                       features={
                                           'label': tf.VarLenFeature(tf.int64),
                                           'image': tf.FixedLenFeature([], tf.string)
                                       })

    image = tf.decode_raw(features['image'], tf.float32)
    image = tf.reshape(image, [32, 100, 3])
    label = tf.cast(features['label'], tf.int32)
    return image, label

def post_process_image(image):
    img_shape = get_image_shape()
    #image = tf.image.decode_jpeg(features['image'], channels=3)
    image = tf.image.decode_jpeg(image, channels=img_shape[2])
    # Resize already done in preprocessing
    #image = tf.image.resize_images(image, [32,100], tf.image.ResizeMethod.BICUBIC)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, img_shape)
    return image

def parse(serialized):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    # It is a bit awkward that this needs to be specified again,
    # because it could have been written in the header of the
    # TFRecords file instead.

    #Synth code
    features = tf.parse_single_example(serialized,
                                       features={
                                           'label': tf.VarLenFeature(tf.int64),
                                           'image': tf.FixedLenFeature([], tf.string)
                                       })

    # image = tf.image.decode_jpeg(features['image'], channels=3)
    # # Resize already done in preprocessing
    # #image = tf.image.resize_images(image, [32,100], tf.image.ResizeMethod.BICUBIC)
    # image = tf.cast(image, tf.float32)
    # image = tf.reshape(image, [32, 100, 3])
    image = post_process_image(features['image'])
    label = tf.cast(features['label'], tf.int32)
    return image, label

def parse_serve(serialized):
    # Define parsing function to be used at serving time
    features = tf.parse_single_example(serialized,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string)
                                       })
    image = post_process_image(features['image'])

    return image

def input_fn(filenames, train=True, batch_size=32, buffer_size=49152, parallel_calls=1, tf_format='JPG'):
    # Args:
    # filenames:   Filenames for the TFRecords files.
    # train:       Boolean whether training (True) or testing (False).
    # batch_size:  Return batches of this size.
    # buffer_size: Read buffers of this size. The random shuffling
    #              is done on the buffer, so it must be big enough.

    # Create a TensorFlow Dataset-object which has functionality
    # for reading and shuffling data from TFRecords files.
    dataset = tf.data.TFRecordDataset(filenames=filenames)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    if tf_format == 'JPG':
        dataset = dataset.map(parse, num_parallel_calls=parallel_calls)
    else:
        dataset = dataset.map(parse_raw, num_parallel_calls=parallel_calls)

    if train:
        # If training then read a buffer of the given size and
        # randomly shuffle it.
        dataset = dataset.shuffle(buffer_size=buffer_size)

        # Allow infinite reading of the data.
        num_repeat = None
        # Repeat the dataset the given number of times.
        dataset = dataset.repeat(num_repeat)

        # Get a batch of data with the given size.
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=batch_size)
    else:
        # If testing then don't shuffle the data and drop remainder
        num_repeat = 1
        # Repeat the dataset the given number of times.
        dataset = dataset.repeat(num_repeat)

        # Only go through the data once.
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()

    # Get the next batch of images and labels.
    images_batch, labels_batch = iterator.get_next()

    # # The input-function must return a dict wrapping the images.
    x = {'image': images_batch}
    y = labels_batch

    return x, y

def get_feature_columns():
    img_shape = get_image_shape()
    feature_image = tf.feature_column.numeric_column("image",
                                                 shape=img_shape)
    my_feature_columns = [feature_image]
    return my_feature_columns

def serving_input_fn():
    #my_feature_columns = get_feature_columns()

    img_shape = get_image_shape()
    serialized_tf_example = tf.placeholder(dtype=tf.string,
                                                  name='input_example_tensor')
    receiver_tensors = {'examples': serialized_tf_example}

    features = tf.parse_example(serialized_tf_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string)
                                       })

    features['image'] = post_process_image(features['image'])

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
