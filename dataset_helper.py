import tensorflow as tf

def parse(serialized):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    # It is a bit awkward that this needs to be specified again,
    # because it could have been written in the header of the
    # TFRecords file instead.
    features = tf.parse_single_example(serialized,
                                       features={
                                           'label': tf.VarLenFeature(tf.int64),
                                           'image': tf.FixedLenFeature([], tf.string)
                                       })

    image = tf.decode_raw(features['image'], tf.float32)
    image = tf.reshape(image, [32, 100, 3])
    label = tf.cast(features['label'], tf.int32)
    return image, label


def input_fn(filenames, train=True, batch_size=32, buffer_size=2048):
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
    dataset = dataset.map(parse)

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
    else:
        # If testing then don't shuffle the data and rop remainder
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

    # return x, y
    return x, y
