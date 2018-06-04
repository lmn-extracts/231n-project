import os
import scipy.io as sio
import xml.etree.ElementTree as ET
import argparse

# GT constructed only for images with alphabetic labels
def generate_gt_icdar03(data_dir):
    gt_path = os.path.join(data_dir, 'word.xml')
    target_gt_path = os.path.join(data_dir, 'gt.mat')

    label_dict = {}
    tree = ET.parse(gt_path)
    root = tree.getroot()
    images = tree.findall('./image')
    for image in images:
        if image.attrib['tag'].isalpha():
            label_dict[image.attrib['file']] = image.attrib['tag']                
    sio.savemat(target_gt_path, label_dict)
    return

def generate_gt_icdar13(data_dir):
    train_set = os.path.join(data_dir, 'train_set')
    gt_path = os.path.join(train_set, 'gt.txt')
    target_gt_path = os.path.join(train_set, 'gt.mat')

    label_dict = {}
    with open(gt_path) as file:
        for line in file:
            words = line.split(',')
            tag = words[1].strip().split("\"")[1]
            if tag.isalpha():
                label_dict[words[0].strip()] = tag
                print (words[0].strip(), tag)
    sio.savemat(target_gt_path, label_dict)

    # Save GT for test set
    test_set = os.path.join(data_dir, 'test_set')
    gt_path = os.path.join(data_dir, 'gt.txt')
    target_gt_path = os.path.join(test_set, 'gt.mat')

    label_dict = {}
    with open(gt_path) as file:
        for line in file:
            words = line.split(',')
            tag = words[1].strip().split("\"")[1]
            if tag.isalpha():
                label_dict[words[0].strip()] = tag
                print (words[0].strip(), tag)
    sio.savemat(target_gt_path, label_dict)


def main(args):
    data_dir = args.dir
    # dataset = 'icdar03/sample_set'
    # subdir = os.path.join(data_dir, dataset)
    # generate_gt_icdar03(subdir)

    # dataset = 'icdar03/trial_train_set'
    # subdir = os.path.join(data_dir, dataset)
    # generate_gt_icdar03(subdir)

    # test set is generating errors; cannot zip
    # dataset = 'icdar03/trial_test_set'
    # subdir = os.path.join(data_dir, dataset)
    # generate_gt_icdar03(subdir)

    dataset = 'icdar13'
    subdir = os.path.join(data_dir, dataset)
    generate_gt_icdar13(subdir)

    return



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dir', default='data', help='Path to dataset directory')
    args = ap.parse_args()
    main(args)
