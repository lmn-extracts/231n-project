'''
Script to test Writing TF Records for the processed Synth dataset and the MJSynth dataset

Usage:
python script.py -d DATA_DIRECTORY -t TARGET_DIRECTORY -gt PATH_TO_GT_FILE
Example: python script.py -d D:\\231n-project\\sampleSynthResults -t D:\\231n-project\\sampleSynthResults -gt D:\\231n-project\\sampleSynthResults\\gt.mat

'''

import argparse
from data_provider import *
import os
import time


def test_synth_tfrecord(data_dir, target_dir, gt_path, trainFile, valFile, testFile):
    tic = time.time()
    tfWriter = SynthTFRecordWriter(data_dir, gt_path, split=True)
    tfWriter._write_feature(trainFile, valFile, testFile)
    toc = time.time()
    print('Writing records took %d seconds'%(toc-tic))
    return

def test_mjsynth_tfrecord(data_dir, target_dir, trainFile, valFile, testFile):
    tfWriter = TFRecordWriter(data_dir, split=True, max_imgs=1000000)
    tfWriter._write_feature(trainFile, valFile, testFile)

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(asctime)s:%(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d')
    parser.add_argument('--target', '-t')
    parser.add_argument('--gt', '-gt')
    args = parser.parse_args()

    data_dir = 'SampleSet' if args.dir is None else args.dir
    target_dir = data_dir if args.target is None else args.target
    gt_path = 'D:\\231n-project\\sampleSynthResults\\gt.mat' if args.gt is None else args.gt

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    trainFile = os.path.join(target_dir, 'train.tfrecords')
    valFile = os.path.join(target_dir, 'val.tfrecords')
    testFile = os.path.join(target_dir, 'test.tfrecords')

    test_synth_tfrecord(data_dir, target_dir, gt_path, trainFile, valFile, testFile)
    # test_mjsynth_tfrecord(data_dir, target_dir, trainFile, valFile, testFile)

    return

if __name__ == '__main__':
    main()
