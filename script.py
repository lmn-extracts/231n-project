from data_provider import *
import os


def main():
    data_dir = 'SampleSet'
    tfWriter = TFRecordWriter(data_dir, split=True)
    trainFile = os.path.join(data_dir, 'train.tfrecords')
    valFile = os.path.join(data_dir, 'val.tfrecords')
    testFile = os.path.join(data_dir, 'test.tfrecords')
    tfWriter._write_feature(trainFile, valFile, testFile)
    return

if __name__ == '__main__':
    main()
