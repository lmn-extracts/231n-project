import argparse
import os
import xml.etree.ElementTree as ET
from shutil import copyfile


class PreProcessData(object):
    def __init__(self):
        pass

    '''
    Make a dictionary of all the filenames, labels
    '''
    def get_label_dict_ICDAR03(self, filepath):
        label_dict = {}
        tree = ET.parse(filepath)
        root = tree.getroot()
        images = tree.findall('./image')
        for image in images:
            label_dict[image.attrib['file'].replace('/','-')] = image.attrib['tag']                
        return  label_dict

    def get_label_dict_ICDAR13(self, filepath):
        label_dict = {}
        with open(filepath) as file:
            for line in file:
                words = line.strip().split(',')
                label_dict[words[0].strip()] = words[1].replace('\"', '').strip()
        return label_dict

    '''
    PREPROCESS_ICDAR03: 
        - Removes images with non-alphabetic labels. 
        - Create a new directory named images/ with the relevant images and an updated word.XML file at <targetdir>
    Arguments:
        - dirpath: Path to the directory where the data and the word.XML file resides
        - targetdir: Specifies the directory to store the resulting images and the new word.XML file
    '''
    def preprocess_ICDAR03(self, dirpath, targetdir):
        wordfile = os.path.join(os.path.dirname(dirpath), 'word.xml')
        label_dict = self.get_label_dict_ICDAR03(wordfile)
        fileHandle = open(os.path.join(targetdir, 'word.txt'), 'w')
        count = 0
        for subdirs, dirs, files in os.walk(dirpath):
            for file in files:
                dirss = subdirs.split('\\')
                filename = (('-').join([dirss[-2], dirss[-1],file]))
                if label_dict[filename].isalpha():
                    copyfile(os.path.join(subdirs,file), os.path.join(targetdir, '{}.jpg'.format(count)))
                    fileHandle.write(' '.join([str(count), label_dict[filename]]) + '\n')
                    count += 1
        fileHandle.close()
        return

    '''
    Requires data directory to be in this format 
        - dir
            - images: Contains all images
            - gt.txt: Contains labels for all images
    '''
    def preprocess_ICDAR13(self, dirpath, targetdir):
        if not os.path.exists(targetdir):
            os.makedirs(targetdir)
        labelfile = os.path.join(os.path.dirname(dirpath), 'gt.txt')
        label_dict = self.get_label_dict_ICDAR13(labelfile)
        count = 0
        fileHandle = open(os.path.join(targetdir, 'gt.txt'), 'w')
        for subdirs, dirs, files in os.walk(dirpath):
            for file in files:
                ext = file.split('.')[-1]
                if not (ext == 'png' or ext == 'jpg' or ext == 'jpeg'):
                    continue
                if label_dict[file].isalpha():
                    copyfile(os.path.join(subdirs,file), os.path.join(targetdir, str(count) + '.' + ext))
                    fileHandle.write(' '.join([str(count), label_dict[file]]) + '\n')
                    count += 1
        fileHandle.close()
        return


    def preprocess_mjsynth(self, dirpath, targetdir):
        if not os.path.exists(targetdir):
            os.makedirs(targetdir)

        count = 0
        fileHandle = open(os.path.join(targetdir, 'gt.txt'), 'w')


        for subdirs, dirs, files in os.walk(dirpath):
            for file in files:
                label = file.split('_')[1]
                if label.isalpha():
                    copyfile(os.path.join(subdirs, file), os.path.join(targetdir, str(count) + '.jpg'))
                    fileHandle.write(str(count) + " " + label + "\n")
                    count += 1
        fileHandle.close()
        return

    def preprocess(self, dataset, dirpath, targetdir):
        if dataset == 'ICDAR03':
            self.preprocess_ICDAR03(dirpath, targetdir)
        elif dataset == 'ICDAR13':
           self.preprocess_ICDAR13(dirpath, targetdir)
        elif dataset == 'mjsynth':
            self.preprocess_mjsynth(dirpath, targetdir)
        return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-ds')
    parser.add_argument('--dir', '-d')
    parser.add_argument('--target', '-t')
    args = parser.parse_args()

    dataset = 'ICDAR03' if args.dataset is None else args.dataset
    dirpath = 'D:\\text-recognition\\datasets\\ICDAR 2003\\SampleSet\\SampleSet' if args.dir is None else args.dir
    targetdir = 'D:\\text-recognition\\datasets\\ICDAR 2003\\SampleSet\\SampleSet' if args.target is None else args.target
    ppd = PreProcessData()
    ppd.preprocess(dataset, dirpath, targetdir)
    return

if __name__ == '__main__':
    main()


