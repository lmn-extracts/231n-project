import xml.etree.ElementTree as ET
import numpy as np
import os
import logging
import scipy.io as sio

def filter_alpha(file_list, label_list):
    list_filter = np.vectorize(lambda x: bool(x.isalpha()))
    filtered_labels = list_filter(label_list)

    print("First 10 orig label", label_list[:10])
    new_labels = list(np.array(label_list)[filtered_labels])
    print('First 10 new labels', new_labels[:10])
    new_files = list(np.array(file_list)[filtered_labels])
    print('First 10 new files', new_files[:10])
    logging.info('Found %d files' % len(file_list))
    logging.info('Returning %d images with only alpha labels' % len(new_files))
    logging.info('Skipped %d images with non alpha labels' %  (len(file_list) - len(new_files)))
    return new_files, new_labels


def mjsynth_reader(data_dir, annofile):

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

    logging.info('Reading %s' % (annofile))
    annotations = sio.loadmat(annofile)
    fileList = list(annotations.keys())[3:]
    labelList = list(annotations.values())[3:]

    return fileList, labelList

def IIIT5K_reader(data_dir, annofile):

    logging.info('Reading %s' % (annofile))
    annotations = sio.loadmat(annofile)
    matkey = list(annotations.keys())[3]
    getLabelList = np.vectorize(lambda x: x[0])
    getFileList= np.vectorize(lambda x: os.path.join(data_dir, x[0]))

    labelList = getLabelList(annotations[matkey]['GroundTruth'][0])
    fileList = getFileList(annotations[matkey]['ImgName'][0])
    del annotations

    return filter_alpha(fileList, labelList)

def ICDAR03_reader(data_dir, annofile):

    logging.info('Reading %s' % (annofile))
    filepath = os.path.join(data_dir,annofile)
    #label_dict = {}
    labelList = []
    fileList = []
    tree = ET.parse(filepath)
    root = tree.getroot()
    images = tree.findall('./image')
    for image in images:
        #print('tag', image.attrib['tag'])
        #print('file',image.attrib['file'] )
        #label_dict[image.attrib['file']] = image.attrib['tag']
        labelList.append(image.attrib['tag'])
        fileList.append(os.path.join(data_dir,image.attrib['file']))

    return filter_alpha(fileList, labelList)

def ICDAR13_reader(data_dir, annofile):

    labelList = []
    fileList = []

    logging.info('Reading %s' % (annofile))
    with open(annofile) as file:
        for line in file:
            words = line.strip().split(',')
            labelList.append(words[1].replace('\"', '').strip())
            fileList.append(os.path.join(data_dir, words[0]))
    file.close()

    return filter_alpha(fileList, labelList)
