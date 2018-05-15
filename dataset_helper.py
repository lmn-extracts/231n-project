import tensorflow as tf

def parse(serialized):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    # It is a bit awkward that this needs to be specified again,
    # because it could have been written in the header of the
    # TFRecords file instead.
    features = \
        {
            'label': tf.VarLenFeature(tf.int64),
            'image': tf.FixedLenFeature([], tf.string)
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)

    # Get the image as raw bytes.
    image_raw = parsed_example['image']

    # Decode the raw bytes so it becomes a tensor with type.
    # This had to be done in this order
    # decode_raw tf.float32, reshape and then cast to uint8
    image = tf.decode_raw(image_raw, tf.float32)
    image = tf.reshape(image, [100, 32, 3])
    image = tf.cast(image, tf.uint8)
    #print("IMG SHAPE", image.shape)
    #image = tf.decode_raw(image_raw, tf.uint8)
    #image = tf.cast(image, tf.float32)

    # Get the label associated with the image.
    label = parsed_example['label']

    #print("LABEL", label)
    #print("IMG SHAPE", image.shape)
    #plt.imshow(image)
    #plt.show()

    # The image and label are now correct TensorFlow types.
    return image, label

def input_fn(filenames, train, batch_size=32, buffer_size=2048):
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
    else:
        # If testing then don't shuffle the data.

        # Only go through the data once.
        num_repeat = 1

    # Repeat the dataset the given number of times.
    dataset = dataset.repeat(num_repeat)

    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)

    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()

    # Get the next batch of images and labels.
    images_batch, labels_batch = iterator.get_next()

    # # The input-function must return a dict wrapping the images.
    # x = {'image': images_batch}
    # y = labels_batch

    # return x, y
    return images_batch, labels_batch
