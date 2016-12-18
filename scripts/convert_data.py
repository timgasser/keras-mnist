import sys
import numpy as np
import scipy as sp
import struct
from tqdm import tqdm
import pickle

# Filename definitions
X_TRAIN = 'train-images-idx3-ubyte'
Y_TRAIN = 'train-labels-idx1-ubyte'

X_TEST = 't10k-images-idx3-ubyte'
Y_TEST =  't10k-labels-idx1-ubyte'

INPUT_DIR = '../input/'

# Common image file settings
MAGIC_X_TRAIN = 2051
MAGIC_Y_TRAIN = 2049
MAGIC_X_TEST = 2051
MAGIC_Y_TEST = 2049

def main(argv=None):
    '''Main function called with arguments'''

    if argv is None:
        argv = sys.argv

    convert_train_images()
    convert_test_images()
    convert_train_labels()
    convert_test_labels()

    return 0

def check_hdr(format, handle, offset, expected, name):
    ''' Checks a correct value exists in binary file using struct.unpack
    INPUT: format - string to interpret binary
           handle - binary file handle
           offset - location in bytestream to start
           expected - expected value
    RETURNS: Bool indicating if there's a match
    '''
    value, = struct.unpack_from(format, handle, offset)
    if value != expected:
        print("Error {} - read {}, expected {}".format(name, value, expected))
        return False
    return True


def convert_train_images():
    ''' Saves out training images as a pickle file '''
    filename = INPUT_DIR + X_TRAIN
    images = load_images(filename, MAGIC_X_TRAIN)
    pickle.dump(images, open(filename + '.pkl', 'wb'))

def convert_test_images():
    ''' Saves out training images as a pickle file '''
    filename = INPUT_DIR + X_TEST
    images = load_images(filename, MAGIC_X_TEST)
    pickle.dump(images, open(filename + '.pkl', 'wb'))

def convert_train_labels():
    '''Save out labels as pickle file'''
    filename = INPUT_DIR + Y_TRAIN
    labels = load_labels(filename, MAGIC_Y_TRAIN)
    pickle.dump(labels, open(filename + '.pkl', 'wb'))

def convert_test_labels():
    '''Save out labels as pickle file'''
    filename = INPUT_DIR + Y_TEST
    labels = load_labels(filename, MAGIC_Y_TEST)
    pickle.dump(labels, open(filename + '.pkl', 'wb'))


def load_images(filename, magic_num):
    ''' Loads training images from `filename`
    INPUT: Filename to load images from
    RETURNS: List of image array data
    '''

    with open(filename, 'rb') as f:
        data = f.read()
        print('Loaded file {} of length {}'.format(filename, len(data)))

    N, = struct.unpack_from('>i', data, 4)
    n_row, = struct.unpack_from('>i', data, 8)
    n_col, = struct.unpack_from('>i', data, 12)
    hdr_size = 16 # in bytes

    img_size = n_col * n_row
    file_size = hdr_size + (N * img_size)
    img_format =  'B' * img_size

    assert check_hdr('>i', data, 0, magic_num, 'magic number')
    # assert check_hdr('>i', data, 4, N, 'number of images')
    # assert check_hdr('>i', data, 8, n_row, 'number of rows')
    # assert check_hdr('>i', data, 12, n_col, 'number of columns')

    # Move the pointer to the start of actual image data
    ptr = hdr_size
    images = np.zeros((N, n_row, n_col), dtype=np.uint8)
    # image_cnt = 0

    for i in tqdm(range(N)):
        image_bytes = struct.unpack_from(img_format, data, 16 + (i * img_size))
        image = np.asarray(image_bytes)
        image = image.reshape(n_row, n_col) # todo ! Check the row/col order of reshape
        images[i, :, :] = np.asarray(image)
        # print('Read image # {} - {}'.format(len(images), images[-1]))

    return images



def load_labels(filename, magic_num):
    ''' Loads labels images from `filename`
    INPUT: Filename to load labels from
    RETURNS: List of labels
    '''
    with open(filename, 'rb') as f:
        data = f.read()
        print('Loaded file {} of length {}'.format(filename, len(data)))

    assert check_hdr('>i', data, 0, magic_num, 'magic number')
    N, = struct.unpack_from('>i', data, 4)
    file_size = 8 + N

    ptr = 8 # Labels have 8-byte header
    labels = np.zeros((N,1), np.uint8)

    for i in tqdm(range(N)):
        label, = struct.unpack_from('>B', data, 8 + i)
        labels[i,:] = label

    return labels

if __name__ == '__main__':

    sys.exit(main())