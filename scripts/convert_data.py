import sys
import numpy as np
import scipy as sp
import struct
import tqdm

# Filename definitions
X_TRAIN = 'train-images.idx3-ubyte'
Y_TRAIN = 't10k-labels.idx1-ubyte'

X_TEST = 't10k-images.idx3-ubyte'
Y_TEST =  't10k-labels.idx1-ubyte'

INPUT_DIR = '../input/'


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


def load_train_images(filename):
    ''' Loads training images from `filename`
    INPUT: Filename to load images from
    RETURNS: List of image array data
    '''

    with open(filename, 'rb') as f:
        data = f.read()
        print('Loaded file {} of length {}'.format(filename, len(data)))


    N = 60000
    n_row = 28
    n_col = 28
    hdr_size = 16 # in bytes

    img_size = n_col * n_row
    file_size = hdr_size + (N * img_size)
    img_format =  'B' * img_size

    assert check_hdr('>i', data, 0, 2051, 'magic number')
    assert check_hdr('>i', data, 4, N, 'number of images')
    assert check_hdr('>i', data, 8, n_row, 'number of rows')
    assert check_hdr('>i', data, 12, n_col, 'number of columns')

    # Move the pointer to the start of actual image data
    ptr = hdr_size
    images = list()
    # image_cnt = 0

    while ptr < file_size:
        image_bytes = struct.unpack_from(img_format, data, ptr)
        images.append(image_bytes)
        ptr += img_size
        print('Read image # {} - {}'.format(len(images), images[-1]))

    return images



if __name__ == '__main__':

    load_train_images(INPUT_DIR + X_TRAIN)
    print('main')