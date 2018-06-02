'''
Processes each image in the DATA_DIRECTORY using ground truth values in file PATH_TO_GT_MAT

In particular, it performs the following preprocessing:
    - Extract alphabetic text strings corresponding to each image
    - Crop and warp each alphabetic word instance and save as image in TARGET_DIR
    - Save labels corresponding to each saved word instance in TARGET_DIR\\labels.txt in the following format
      Example:
      Record 0 STRING corresponds to word instance stored in image 0.jpg with label STRING.

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

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

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
    del mat
    # Track success
    success_count = 1
    failure_count = 1
    file_number = 1

    # Store labels in the form: labels[IMAGE_FILE_NAME] = TEXT. Example: labels['0.jpg'] = 'Lines'
    # labels = []

    target_labels_path = os.path.join(target_dir, 'gt.txt')
    labels_file = open(target_labels_path, 'w')

    for i,imagepath in enumerate(imnames):
        imagepath.replace('/','\\')
        filepath = os.path.join(data_dir, imagepath)
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
                target_folder = os.path.dirname(imagepath)
                target_folder = os.path.join(target_dir, target_folder)
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                target_image_file = '%d.jpg'%(file_number)
                targetpath = os.path.join(target_folder, target_image_file)

                cv2.imwrite(targetpath, warped)
                logging.info('SUCCESS: [%s]'%(targetpath))
                labels_file.write('%d %s\n'%(file_number,text_strings[idx]))
                file_number += 1                    
            logging.info('Finished processing [%s]. Success: %d'%(imagepath, success_count))
            success_count += 1
        except Exception as e:
            logging.debug('Skipping [%s]. Details: %s. Failed: %d'%(imagepath, str(e), failure_count))
            failure_count += 1
            continue
    labels_file.close()
    return


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dir', required=True, help='Path to image directory')
    ap.add_argument('-t', '--target', required=True, help='Path to target data directory where the processed word instance images should be stored')
    ap.add_argument('-gt', '--gt', required=True, help='Path to ground truth MATLAB file')
    args = vars(ap.parse_args())

    main(args)

