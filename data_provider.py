from data_utils import *
from data_utils import _int64_feature, _byte_feature
import tensorflow as tf
import os
import sys
import random

class TFRecordWriter(object):
    def __init__(self, data_dir, split=False):
        '''
        Arguments:
            - data_dir: Location where the images and the ground truth label files reside
            - split: Boolean flag. Indicates whether or not to split the data into train/val/test
        '''
        fileList = []
        labelList = []
        label_dict = {}
        total = 0
        with open(os.path.join(data_dir, 'gt.txt')) as f:
            for line in f:
                filename, label = line.split(' ')
                label = label.strip()
                label = [chr2idx(c) for c in label]
                label_dict[filename] = label
                total += 1
                if total % 100000 == 0:
                    print('Annotations checked: %d'% (total))
                    sys.stdout.flush()
                    #msg = "\r- Annotations checked: {}".format(total)
                    #sys.stdout.write(msg)
                    #sys.stdout.flush()

        total = 0
        for subdirs, dirs, files in os.walk(data_dir):
            for file in files:
                ext = file.split('.')[-1]
                if not (ext == 'jpg' or ext == 'jpeg' or ext=='png'):
                    continue
                fileList.append(os.path.join(subdirs, file))
                labelList.append(label_dict[file.split('.')[0]])
                total += 1
                if total % 100000 == 0:
                    print('Files Appended: %d'% (total))
                    sys.stdout.flush()
                    #msg = "\r- Files Appended: {}".format(total)
                    #sys.stdout.write(msg)
                    #sys.stdout.flush()

        self.split = split

        self.train_files = fileList
        self.train_labels = labelList 
        self.test_files = None
        self.test_labels = None
        self.val_files = None
        self.val_labels = None

        if split:
            zipped = list(zip(fileList, labelList))
            random.seed(238)
            shuffle(zipped)
            fileList, labelList = zip(*zipped)

            N = len(fileList)

            self.train_files = fileList[0 : int(0.6*N)]
            self.train_labels = labelList[0: int(0.6*N)]
            self.val_files = fileList[int(0.6*N) : int(0.8*N)]
            self.val_labels = labelList[int(0.6*N) : int(0.8*N)]
            self.test_files = fileList[int(0.8*N):]
            self.test_labels = labelList[int(0.8*N):]

    def _write_feature(self,train_file, val_file=None, test_file=None):
        writer = tf.python_io.TFRecordWriter(train_file)
        N = len(self.train_files)
        for i in range(N):
            if (i % 1000) == 0:
                print('Train Data: %d/%d records saved' % (i,N))
                sys.stdout.flush()

            try:
              image = load_image(self.train_files[i])
            except ValueError:
               print('Ignoring image: ', self.train_files[i])
               continue
            label = self.train_labels[i]    
            feature = {
                'label': _int64_feature(label),
                'image': _byte_feature(tf.compat.as_bytes(image.tostring()))
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())
        writer.close()

        if not self.split:
            return

        writer = tf.python_io.TFRecordWriter(val_file)
        N = len(self.val_files)
        for i in range(N):
            if (i % 1000) == 0:
                print('Val Data: %d/%d records saved' % (i,N))
                sys.stdout.flush()

            image = load_image(self.val_files[i])
            label = self.val_labels[i]

            feature = {
                'label': _int64_feature(label),
                'image': _byte_feature(tf.compat.as_bytes(image.tostring()))
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())
        writer.close()

        writer = tf.python_io.TFRecordWriter(test_file)
        N = len(self.test_files)
        for i in range(N):
            if (i % 1000) == 0:
                print('Test Data: %d/%d records saved' % (i,N))
                sys.stdout.flush()

            image = load_image(self.test_files[i])
            label = self.test_labels[i]

            feature = {
                'label': _int64_feature(label),
                'mage': _byte_feature(tf.compat.as_bytes(image.tostring()))
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())
        writer.close()

class TFRecordReader(object):
    def __init__(self, num_epochs=None):
        self.num_epochs = num_epochs

    def _read_feature(self, datapath):
        filename_queue = tf.train.string_input_producer([datapath], num_epochs=self.num_epochs)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example, 
            features={
                'label': tf.VarLenFeature(tf.int64),
                'image': tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(features['image'], tf.float32)
        image = tf.reshape(image, [32,100,3])
        label = tf.cast(features['label'], tf.int32)
        return image, label



