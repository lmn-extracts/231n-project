import argparse
from data_provider import *
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d')
    parser.add_argument('--target', '-t')
    args = parser.parse_args()

    data_dir = 'SampleSet' if args.dir is None else args.dir
    target_dir = data_dir if args.target is None else args.target
    tfWriter = TFRecordWriter(data_dir, split=True)
    trainFile = os.path.join(target_dir, 'train.tfrecords')
    valFile = os.path.join(target_dir, 'val.tfrecords')
    testFile = os.path.join(target_dir, 'test.tfrecords')
    tfWriter._write_feature(trainFile, valFile, testFile)
    return

if __name__ == '__main__':
    main()
