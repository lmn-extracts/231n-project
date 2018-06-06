import xml.etree.ElementTree as ET
import numpy as np
import os
import logging
import scipy.io as sio

def check_files(data_dir, annofile):
    if not os.path.exists(annofile):
        logging.error('Could not locate Annotations file at %s' % (annofile))
        return False

    if not os.path.exists(data_dir):
        logging.error('Data Directory [%s] does not exist.' % (data_dir))
        return False

    return True


def filter_alpha(file_list, label_list):
    list_filter = np.vectorize(lambda x: bool(x.isalpha()))
    filtered_labels = list_filter(label_list)

    new_labels = list(np.array(label_list)[filtered_labels])
    new_files = list(np.array(file_list)[filtered_labels])
    logging_info('Found %d files' % len(file_list))
    logging_info('Returning %d images with only alpha labels' % len(new_files))
    logging_info('Skipped %d images with non alpha labels' %  len(file_list) - len(new_files))
    return new_files, new_labels


def mjsynth_reader(data_dir, annofile):

    if not check_files(data_dir, annofile):
        return

    fileList = []
    labelList = []

    logging.info('Reading %s' % (annofile))
    with open(os.path.join(annofile)) as f:
        for line in f:
            line.strip()
            filename = os.path.basename(line)
            label = filename.split('_')[1]
            fileList.append(os.path.join(data_dir, line))
            labelList.append(label)
    f.close()

    return filter_alpha(fileList, labelList)


def synth_reader(data_dir, annofile):
    if not check_files(data_dir, annofile):
        return

    logging.info('Reading %s' % (annofile))
    annotations = sio.loadmat(annofile)
    fileList = list(annotations.keys())[3:]
    labelList = list(annotations.values())[3:]

    return fileList, labelList


def ICDAR03_reader(data_dir, annofile):
    if not check_files(data_dir, annofile):
        return

    logging.info('Reading %s' % (annofile))
    filepath = os.path.join(data_dir,annofile)
    #label_dict = {}
    labelList = []
    fileList = []
    tree = ET.parse(filepath)
    root = tree.getroot()
    images = tree.findall('./image')
    for image in images:
        #label_dict[image.attrib['file']] = image.attrib['tag']
        labelList.append(image.attrib['tag'])
        fileList.append(image.attrib['file'])

    return filter_alpha(labelList, fileList)


