import os
import tensorflow as tf
import numpy as np
import cv2
import math
from scipy.stats import mode
import logging
import threading

def _int64_feature(value):
    val = value
    if type(value) is not list:
        val = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=val))

def _byte_feature(value):
    val = value
    if type(value) is not list:
        val = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=val))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature_list(values):
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def get_file_names(dirname):
    fileList = []
    labelList = []
    label_dict = {}
    with open(os.path.join(dirname, 'gt.txt')) as f:
        for line in f:
            filename, label = line.split(' ')
            label_dict[filename] = label

    for subdirs, dirs, files in os.walk(dirname):
        for file in files:
            ext = file.split('.')[-1]
            if ext == 'txt':
                continue
            fileList.append(os.path.join(subdirs, file))
            labelList.append(label_dict[file.split('.')[0]])
    return fileList, labelList 

def chr2idx(c):
    # A --> 0, Z --> 25, a --> 26, z --> 51 ==> Num_classes = 52
    ord_val = ord(c)
    ord_val = (ord_val - 65) if ord_val <=91 else (ord_val - 97 + 26)
    return ord_val


def mprint(label, item):
    print('\n-------------------------------')
    print(label + ': ' + str(item))
    print('-------------------------------')

def load_image(image, input_width=100):
    """
        Resize an image to the "good" input size
    """

    print("Processing image", image)
    
    try:
        im_arr = cv2.imread(image)
        #print('im arr shape', im_arr.shape)
        r, c,_ = np.shape(im_arr)
        final_arr = cv2.resize(im_arr, (input_width,32), interpolation=cv2.INTER_CUBIC)
        final_arr = final_arr.astype(np.float32)
        #print('final arr shape', final_arr.shape)
        return final_arr
    except (ValueError, AttributeError):
        print("Error while processing image ", image)
        raise

def _parse_function(filename, label, max_char_count=10):
    image = load_image(filename)
    label = [chr2idx(c) for c in label]
    label = tf.constant([chr2idx(c) for c in label], dtype=tf.int32)
    mprint('image shape', image.shape)
    return image, label

def idx2char(i):
    if i == -1:
        return ''
    c = chr(i+65) if i <=25 else chr(i + (97-26))
    return c

def nd_array_to_labels(nd_arr):
    labels = []

    for i in range(len(nd_arr)):
        label = [idx2char(c) for c in nd_arr[i,:]]
        label = ''.join(label)
        labels.append(label)

    return labels

def get_accuracy(preds, labels):
    acc = []
    for pred, label in zip(preds, labels):
        if len(label) == 0:
            acc.append(1.0)
            continue
        total = min(len(label), len(pred))
        correct = len([i for i in range(total) if pred[i] == label[i]])
        acc.append(correct/len(label))
    return np.mean(acc)

def compute_accuracy(preds, labels):
    total = tf.shape(labels.values)[0]
    preds = tf.sparse_to_dense(preds.indices, preds.dense_shape, preds.values, default_value=-1)
    labels = tf.sparse_to_dense(labels.indices, labels.dense_shape, labels.values, default_value=-2)

    r = tf.shape(labels)[0]
    c = tf.minimum(tf.shape(labels)[1], tf.shape(preds)[1])
    preds = tf.slice(preds, [0,0], [r,c])
    labels = tf.slice(labels, [0,0], [r,c])

    preds = tf.cast(preds, tf.int32)
    labels = tf.cast(labels, tf.int32)

    correct = tf.reduce_sum(tf.cast(tf.equal(preds, labels), tf.int32))
    accuracy = tf.divide(correct, total)
    return accuracy


# Get resized image
def resized_byte_string(filename):
    image = cv2.imread(filename)
    # Convert to RGB as Tensorflow processes RGB images
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h,w,_ = image.shape

    result = np.zeros([32,100,3])
    
    # Compute the background by taking the most common value across the 3 channels
    # c0 = mode(image[:,:,0])[0][0][0]
    # c1 = mode(image[:,:,1])[0][0][0]
    # c2 = mode(image[:,:,2])[0][0][0]

    #result[:, :, 0] = np.full([32, 100], c0)
    #result[:, :, 1] = np.full([32, 100], c1)
    #result[:, :, 2] = np.full([32, 100], c2)

    result = result.astype(np.uint8)

    # Compute Stretch Factors
    sw = 100/w
    sh = 32/h
    
    stretch = min(sw,sh)
    sw = min(int(math.ceil(stretch * w)), 100)
    sh = min(int(math.ceil(stretch * h)), 32)

    image = cv2.resize(image, (sw,sh))
    result[:sh,:sw,:] = image

    # Encode resized image to byte string
    encoded_image = cv2.imencode('.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tostring()
    return encoded_image

# Consolidate image processing to leverage parallel processing
def process_sgl_image(image, label):
    try:
        #logging.debug('Processing image [%s]' % image)
        encoded_image = resized_byte_string(image)
        encoded_label = [chr2idx(c) for c in label[0]]

        feature = {
            'label': _int64_feature(encoded_label),
            'image': _bytes_feature(encoded_image)
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        return example

    except Exception as e:
        logging.error('Encountered an exception while processing [%s]. Details: %s' % (image, str(e)))

class AsyncWrite(threading.Thread):
    def __init__(self, writer, examples, name='WriteThread'):
        threading.Thread.__init__(self, name=name)
        self.writer = writer
        self.examples = examples
        self._stopevent = threading.Event()
        self._sleepperiod = 1.0
        self.running = False

    def run(self):
        self.running = True
        count = 0
        for example in self.examples:
            try:
                self.writer.write(example.SerializeToString())
                count += 1
                #logging.debug('Writing record [%d]' % count)
            except Exception as e:
                logging.error('Encountered an exception while writing tfrecord to file. Details: %s' % (str(e)))
        logging.info('%d records written to tfrecord' % (count))
        self.running = False
        while not self._stopevent.isSet():
            self._stopevent.wait(self._sleepperiod)
        logging.info( "%s ends" % (self.getName()) )

    def join(self, timeout=None):
        logging.debug('Got request to join thread')
        logging.debug('Thread running status: %s' % (self.running))
        self._stopevent.set()
        threading.Thread.join(self,timeout)

