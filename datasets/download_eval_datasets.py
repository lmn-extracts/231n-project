'''
Script to download the following datasets:
    - ICDAR 2003 Word Recognition (Ref: http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions)
    - ICDAR 2013 Word Recognition (Ref: http://rrc.cvc.uab.es/?ch=2&com=downloads)
    - IIIT5k (Ref: http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)
    - SVT
'''

import wget
import zipfile as zf
import os
import argparse
import shutil
import logging

def download_ICDAR_2003(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    sample_set = os.path.join(data_dir, 'sample_set.zip')
    trial_train_set = os.path.join(data_dir, 'trial_train_set.zip')
    trial_test_set = os.path.join(data_dir, 'trial_test_set.zip')
    
    sample_set_url = 'http://www.iapr-tc11.org/dataset/ICDAR2003_RobustReading/Sample/word.zip'
    trial_train_url = 'http://www.iapr-tc11.org/dataset/ICDAR2003_RobustReading/TrialTrain/word.zip'
    trial_test_url = 'http://www.iapr-tc11.org/dataset/ICDAR2003_RobustReading/TrialTest/word.zip'

    try:
        print ('\nDownloading Sample Set for ICDAR03...')
        wget.download(sample_set_url, sample_set)
    except Exception as e:
        logging.error('Error while downloading ICDAR03 Sample Set. Details: %s'%(str(e)))
        shutil.rmtree(sample_set)

    try:
        print ('\nDownloading Train Set for ICDAR03...')
        wget.download(trial_train_url, trial_train_set)
    except Exception as e:
        logging.error('Error while downloading ICDAR03 Train Set. Details: %s'%(str(e)))
        shutil.rmtree(trial_train_set)

    try:        
        print ('\nDownloading Test Set for ICDAR03...')
        wget.download(trial_test_url, trial_test_set)
    except Exception as e:
        logging.error('Error while downloading ICDAR03 Test Set. Details: %s'%(str(e)))
        shutil.rmtree(trial_test_set)
    return

def download_ICDAR_2013(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train_set = os.path.join(data_dir, 'train_set.zip')
    test_set = os.path.join(data_dir, 'test_set.zip')
    gt = os.path.join(data_dir, 'gt.txt')
    
    train_set_url = 'http://rrc.cvc.uab.es/downloads/Challenge2_Training_Task3_Images_GT.zip' 
    test_set_url = 'http://rrc.cvc.uab.es/downloads/Challenge2_Test_Task3_Images.zip'
    gt_url = 'http://rrc.cvc.uab.es/downloads/Challenge2_Test_Task3_GT.txt'

    # try:
    #     print ('\nDownloading Train Set for ICDAR 2013...')
    #     wget.download(train_set_url, train_set)
    # except Exception as e:
    #     logging.error('Error while downloading ICDAR03 Train Set. Details: %s'%(str(e)))
    #     shutil.rmtree(train_set)

    # try:        
    #     print ('\nDownloading Test Set for ICDAR 2013...')
    #     wget.download(test_set_url, test_set)
    # except Exception as e:
    #     logging.error('Error while downloading ICDAR 2013 Test Set. Details: %s'%(str(e)))
    #     shutil.rmtree(test_set)

    try:        
        print ('\nDownloading Ground Truth for ICDAR 2013...')
        wget.download(gt_url, gt)
    except Exception as e:
        logging.error('Error while downloading ICDAR 2013 Ground Truth. Details: %s'%(str(e)))
        shutil.rmtree(gt)

    return

def download_IIIT5k(data_dir):
    url = 'http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz'
    filename = os.path.join(data_dir, word)
    try:
        print ('\nDownloading IIIT5k dataset...')
        wget.download(url, filename)
    except Exception as e:
        logging.error('Error while downloading IIIT5k dataset. Details: %s'%(str(e)))
        shutil.rmtree(filename)

def main(args):
    data_dir = args.dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download ICDAR03
    # dataset = 'icdar03'
    # subdir = os.path.join(data_dir, dataset)
    # download_ICDAR_2003(subdir)

    # Download ICDAR 2013
    dataset = 'icdar13'
    subdir = os.path.join(data_dir, dataset)
    download_ICDAR_2013(subdir)

    # Download IIIT5k
    # dataset = 'iiit5k'
    # subdir = os.path.join(data_dir, dataset)
    # download_IIIT5k(subdir)

    return

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dir', default='data', help='Path to dataset directory')
    args = ap.parse_args()
    main(args)
