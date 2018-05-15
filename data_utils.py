import os
import tensorflow as tf
import numpy as np
from random import shuffle
import cv2

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
    im_arr = cv2.imread(image)
    r, c,_ = np.shape(im_arr)
    final_arr = cv2.resize(im_arr, (input_width,32), interpolation=cv2.INTER_CUBIC)
    final_arr = final_arr.astype(np.float32)
    return final_arr

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
