import glob
import os
from random import shuffle
from sys import platform

import numpy as np
import tensorflow as tf
from PIL import Image

if platform == 'win32':
    DATASET_PATH = 'D:/Temp/ClassifierData'
    CLEAR = 'cls'
elif platform == 'linux':
    CLEAR = 'clear'
    DATASET_PATH = '/home/jscholz/Code/ClassifierTest/Dataset'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def feature_generator(path):
    addrs = glob.glob(path)
    shuffle(addrs)
    for i, path in enumerate(addrs):
        fname = os.path.basename(path)
        try:
            im = Image.open(path)
            if im.height != 256 or im.width != 256:
                if im.height > im.width:
                    im.thumbnail((256, 999999), Image.BICUBIC)
                else:
                    im.thumbnail((999999, 256), Image.BICUBIC)
                im = im.crop(((im.width - 256) / 2,
                              (im.height - 256) / 2,
                              (im.width + 256) / 2,
                              (im.height + 256) / 2))
                # im.show()
            # img = np.array(im)/256
            img = np.array(im)
            if 'face' in fname:
                label = 1
            else:
                label = 0
            if img.shape == (256, 256, 3):
                # Black & White Pictures are discarded
                yield (img, label)
        except Exception as e:
            print(e)


def create_tfrecord(qualifier):

    max_files = 5000

    print(f'getting Data from: {DATASET_PATH}/{qualifier}/*.jpg')
    features = feature_generator(f'{DATASET_PATH}/{qualifier}/*.jpg')
    writer = None
    num_files = len(glob.glob(f'{DATASET_PATH}/{qualifier}/*.jpg'))
    print(f'found {num_files} files in {DATASET_PATH}/{qualifier}/')
    for i, feat in enumerate(features):
        if not i % 50:
            os.system(CLEAR)
            print(f'writing: {DATASET_PATH}/../ClassifierDataTfRecords/{qualifier}{i//max_files}.tfrecords')
            print(f'{qualifier} data: {i}/{num_files}')
        if not i % max_files:
            if writer is not None:
                writer.close()
            writer = tf.python_io.TFRecordWriter(
                f'{DATASET_PATH}/../ClassifierDataTfRecords/{qualifier}{i//max_files}.tfrecords')

        img, label = feat

        feature = {'label': _int64_feature(int(label)),
                   'image': _bytes_feature(img.tostring())}

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

    writer.close()


create_tfrecord('Train')
create_tfrecord('Test')
