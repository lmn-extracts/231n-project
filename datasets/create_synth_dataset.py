'''
Processes each image in the DATA_DIRECTORY using ground truth values in file PATH_TO_GT_MAT

In particular, it performs the following preprocessing:
    - Extract alphabetic text strings corresponding to each image
    - Crop and warp each alphabetic word instance and save as image in TARGET_DIR
    - Save labels corresponding to each saved word instance in TARGET_DIR\\labels.mat

Usage:
python create_synth_dataset.py -d DATA_DIRECTORY -gt PATH_TO_GT_MAT
Example: python create_synth_dataset.py -d D:\\231n-project\\synth -gt D:\\231n-project\\synth\\small_gt.mat -t D:\\231n-project\\sampleSynthResults

NOTE:
Take care to choose a TARGET_DIR outside of the DATA_DIRECTORY
'''

import cv2
import numpy as np
import scipy.io as sio
import re
import argparse
import os
import logging

from dataset_utils import *

'''
Arguments:
    - text: A list of strings corresponding to an image. Typically processed as txt[0].tolist()
'''
def get_text_strings(text):
    text = [s.strip() for s in text]
    text = [re.split(' |\n',s) for s in text]
    text = [s.strip() for sub in text for s in sub if s != '']
    return text


def main(args):
    data_dir = args['dir']
    gt_path = args['gt']
    target_dir = args['target']
    log_file_path = os.path.join(target_dir, 'log.txt')

    logging.basicConfig(filename=log_file_path, filemode='w', level=logging.DEBUG, format='%(levelname)s %(asctime)s:%(message)s')
    logging.info('Starting to execute script create_synth_dataset.py')
    logging.info('Looking for Directory [{}]'.format(data_dir))
    logging.info('Looking for ground truth MATLAB file at [{}]'.format(gt_path))

    if not os.path.exists(data_dir):
        logging.error('Could not find directory [{}]'.format(data_dir))
        # print 'Could not find directory [{}]'.format(data_dir)
        return

    if not os.path.exists(gt_path):
        logging.error('Could not locate Ground Truth MATLAB File at [{}]'.format(gt_path))
        # print 'Could not locate Ground Truth MATLAB File at [{}]'.format(gt_path)
        return

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    try:
        logging.info('Loading Ground Truth MATLAB file from %s'%(gt_path))
        mat = sio.loadmat(gt_path)        
    except Exception as e:
        logging.error('Error while reading Ground Truth file. Details: %s'%(str(e)))
        return

    image_names = mat['imnames'].reshape(-1)
    wordBB = mat['wordBB'].reshape(-1)
    txt = mat['txt'].reshape(-1)

    # Create dictionary of image names for fast retrieval of index
    imnames = dict((image_names[i][0], i) for i in range(len(image_names)))
    del image_names

    # Track success
    success_count = 1
    failure_count = 1

    # Store labels in the form: labels[IMAGE_FILE_NAME] = TEXT. Example: labels['0.jpg'] = 'Lines'
    labels = {}

    for dirpath, dirname, files in os.walk(data_dir):
        for filename in files:
            folder = dirpath.split('\\')[-1]
            imagepath = '%s/%s'%(folder,filename)
            filepath = os.path.join(dirpath, filename)
            ext = filename.split('.')[-1]
            if not (ext == 'jpg' or ext == 'jpeg' or ext == 'png'):
                continue
            logging.info('Processsing file [%s] with imagepath [%s]'%(filepath, imagepath))

            # Get index of image; Log error if unable to locate record
            try:
                i = imnames[imagepath]
            except:
                logging.debug('Skipping [%s]. Could not locate record in imnames. Failed: %d'%(imagepath, failure_count))
                failure_count += 1
                continue
            text_strings = get_text_strings(txt[i].tolist())

            try:
                # Check if number of retrieved words is correct, i.e. compare it with the number of bounding boxes
                # When the number of words is 1, then bounding box has shape (2,4). Expand dims such that dims is (2,4,1)
                if len(wordBB[i].shape) != 3:
                    wordBB[i] = np.expand_dims(wordBB[i], 2)
                if len(text_strings) != wordBB[i].shape[-1]:
                    logging.debug('Skipping [%s] due to error in num_words. Failed: %d'%(imagepath, failure_count))
                    failure_count += 1
                    continue

                # Filter words that are alphabetic
                indices = [idx for idx in range(len(text_strings)) if text_strings[idx].isalpha()]

                # Process and store each image 
                bboxes = np.transpose(wordBB[i], [2,1,0])
                image = cv2.imread(filepath)
                for idx in indices:
                    warped = four_point_transform(image, bboxes[idx,:,:])
                    target_image_file = '%d.jpg'%(success_count)
                    targetpath = os.path.join(target_dir, target_image_file)
                    cv2.imwrite(targetpath, warped)
                    logging.info('SUCCESS: [%s]'%(targetpath))
                    labels[target_image_file] = text_strings[idx][0]
                logging.info('Finished processing [%s]. Success: %d'%(imagepath, success_count))
                success_count += 1
            except Exception as e:
                logging.debug('Skipping [%s]. Details: %s. Failed: %d'%(imagepath, str(e), failure_count))
                failure_count += 1
                continue

    try:
        target_labels_path = os.path.join(target_dir, 'labels.mat')
        sio.savemat(target_labels_path, labels)
    except Exception as e:
        logging.debug('Error while writing labels to %s'%(target_labels_path))

    return


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dir', required=True, help='Path to image directory')
    ap.add_argument('-t', '--target', required=True, help='Path to target data directory where the processed word instance images should be stored')
    ap.add_argument('-gt', '--gt', required=True, help='Path to ground truth MATLAB file')
    args = vars(ap.parse_args())

    main(args)

