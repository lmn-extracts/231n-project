from data_utils import *
from data_utils import _int64_feature, _bytes_feature
import tensorflow as tf
import os
import sys
import random
import scipy.io as sio
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from random import shuffle

class SynthTFRecordWriter(object):
    def __init__(self, data_dir, gt_path, split=False, n_workers=1, batch_size=10000):
        if not os.path.exists(gt_path):
            logging.error('Could not locate Ground Truth dictionary at %s'%(gt_path))
            return

        if not os.path.exists(data_dir):
            logging.error('Data Directory [%s] does not exist.'%(data_dir))

        logging.info('Reading %s' % (gt_path))
        annotations = sio.loadmat(gt_path)
        self.split = split
        self.n_workers = n_workers
        self.batch_size = batch_size
        all_files = list(annotations.keys())[3:]
        labelList = list(annotations.values())[3:]
        N = len(all_files)
        logging.info('Total files to process %d' % N)

        zipped = list(zip(all_files, labelList))
        random.seed(238)
        shuffle(zipped)
        all_files, self.labels = zip(*zipped)

        self.train_files = all_files

        if split:
            self.train_files = all_files[0: int(0.6 * N)]
            self.val_files = all_files[int(0.6*N):int(0.8*N)]
            self.test_files = all_files[int(0.8*N):]


    def _write_fn(self, out_file, image_list, mode):
        writer = tf.python_io.TFRecordWriter(out_file)
        N = len(image_list)
        logging.info('Writing %d images to %s'%(N, out_file))
        tic = time.time()
        nworkers = self.n_workers
        batch_size = self.batch_size
        report_interval = batch_size // nworkers * nworkers
        count = 0
        writing = False
        end = 0
        for i in range(0, N, batch_size):
            try:
                with ProcessPoolExecutor(max_workers=nworkers) as executor:
                    end = i + batch_size if(i+batch_size < N) else N
                    result = executor.map(process_sgl_image, image_list[i:end], self.labels[i:end])
                    count = 0
                if writing:
                    # Wait until previous writer completes
                    while background_writer.is_alive():
                        background_writer.join(10)
                batch_id = 'Writer_Batch_' + str(i) + '_' + str(end)
                background_writer = AsyncWrite(writer, result, name=batch_id)
                background_writer.start()
                writing = True
                    # for example in result:
                    #     writer.write(example.SerializeToString())
                    #     count += 1
            except Exception as e:
                logging.error('Encountered an exception while processing batch starting at index [%d]. Details: %s'%(i, str(e)))

            toc = time.time()
            if i == batch_size:
                estimated_time = N*(toc-tic)/batch_size // 60
                logging.info('Estimated time to complete %s data: %d mins' % (mode, estimated_time))
            #logging.info('%s Data: %d/%d . %d records saved in %d secs' % (mode, i+batch_size, N, count toc-tic))
            logging.info('%s Data: %d/%d records saved in %d secs' % (mode, end, N, toc-tic))
            tic = time.time()
        while background_writer.is_alive():
            background_writer.join(10)
        writer.close()

    def _write_feature(self, train_file, val_file=None, test_file=None):
        self._write_fn(train_file, self.train_files, mode="Train")

        if not self.split:
            return

        self._write_fn(val_file, self.val_files, mode="Val")

        self._write_fn(test_file, self.test_files, mode="Test")



class TFRecordWriter(object):
    def __init__(self, data_dir, split=False, max_imgs=None):
        '''
        Arguments:
            - data_dir: Location where the images and the ground truth label files reside
            - split: Boolean flag. Indicates whether or not to split the data into train/val/test
            - max_imgs: Number of max images to write to tfrecord files
        Operation:
         - Intializes self.train_files to contain the list of image file names
         - Initialzes self.train_labels to contain the list of labels correspondingi to self.train_files
         - Similarly initializes self.val_files, self.val_labels, self.test_files, self.test_labels
        '''
        fileList = []
        labelList = []
        label_dict = {}
        total = 0

        # Store the label for each file in a label dictionary
        # example: label_dict[15] = STOP
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

        self.split = split

        N = len(fileList)
        if max_imgs is not None and max_imgs < N:
            N = max_imgs
            print('Capping number of files at defined max_imgs: ', max_imgs)
        self.train_files = fileList[:N]
        self.train_labels = labelList[:N]
        self.test_files = None
        self.test_labels = None
        self.val_files = None
        self.val_labels = None

        if split:
            zipped = list(zip(fileList, labelList))
            random.seed(238)
            shuffle(zipped)
            fileList, labelList = zip(*zipped)

            self.train_files = fileList[0 : int(0.6*N)]
            self.train_labels = labelList[0: int(0.6*N)]
            self.val_files = fileList[int(0.6*N) : int(0.8*N)]
            self.val_labels = labelList[int(0.6*N) : int(0.8*N)]
            self.test_files = fileList[int(0.8*N):N]
            self.test_labels = labelList[int(0.8*N):N]

    def _write_fn(self, out_file, image_list, label_list, mode):
        writer = tf.python_io.TFRecordWriter(out_file)
        N = len(image_list)
        for i in range(N):
            if (i % 1000) == 0:
                print('%s Data: %d/%d records saved' % (mode, i,N))
                sys.stdout.flush()

            # Resize image to 32 x 100 and get encoded byte string
            encoded_image = resized_byte_string(image_list[i])

            # write the label (string) and image filename (string)
            feature = {
                'label': _int64_feature(label_list[i]),
                'image': _bytes_feature(encoded_image)
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())
        writer.close()

    def _write_feature(self, train_file, val_file=None, test_file=None):
        self._write_fn(train_file, self.train_files, self.train_labels, mode="Train")

        if not self.split:
            return

        self._write_fn(val_file, self.val_files, self.val_labels, mode="Val")

        self._write_fn(test_file, self.test_files, self.test_labels, mode="Test")

class TFRecordReader(object):
    def __init__(self, num_epochs=None):
        self.num_epochs = num_epochs

    def _read_feature(self, datapath):
        filename_queue = tf.train.string_input_producer([datapath], num_epochs=self.num_epochs)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example, 
            features={
                'label': tf.VarLenFeature(tf.string),
                'image': tf.VarLenFeature(tf.string)
            })

        image = tf.load_image(features['image'])
        image = tf.decode_raw(image, tf.float32)
        image = tf.reshape(image, [32,100,3])
        label = [chr2idx(c) for c in features['label']]
        label = tf.cast(label, tf.int32)

        # image = tf.decode_raw(features['image'], tf.float32)
        # image = tf.reshape(image, [32,100,3])
        # label = tf.cast(features['label'], tf.int32)
        return image, label



