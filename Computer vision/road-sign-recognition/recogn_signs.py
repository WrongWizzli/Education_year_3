from math import sqrt, pi

import numpy as np
from numpy import array as arr

from sklearn.svm import SVC
from sklearn.utils import shuffle
from skimage.transform import resize
'''--------------------------------------------'''

Sx = arr([[-1, 0, 1], 
          [-2, 0, 2], 
          [-1, 0, 1]])
Sx = np.dstack((Sx, Sx, Sx))

Sy = arr([[1, 2, 1], 
          [0, 0, 0], 
          [-1, -2, -1]])
Sy = np.dstack((Sy, Sy, Sy))

DEFAULT_SHAPE = (32, 32)
CELL_SIZE = [8, 8]
BLOCK_SIZE = [3, 3]
BIN_COUNT = 8
EPS = 1e-9

'''--------------------------------------------'''

def transform_image(img, op_x, op_y):

    new_img1 = np.zeros((img.shape[0], img.shape[1], 3))
    new_img2 = np.zeros((img.shape[0], img.shape[1], 3))

    for i in range(1, len(img) - 1):
        for j in range(1, len(img[0]) - 1):

            tmp = img[i-1:i+2, j-1:j+2, :]

            sum1 = np.sum(np.sum(tmp * op_x, axis=0), axis=0)
            sum2 = np.sum(np.sum(tmp * op_y, axis=0), axis=0)
            new_img1[i - 1, j - 1, :]= sum1
            new_img2[i - 1, j - 1, :]= sum2

    return new_img1, new_img2

def norm(a):
    return a / sqrt(np.sum(a ** 2) + EPS)

def extract_hog(image):

    if len(image.shape) == 3:
        image = image[..., :3]

    image = resize(image, DEFAULT_SHAPE)
    length, width = len(image), len(image[0])
    image_x, image_y = transform_image(image.copy(), Sx, Sy)
    G = np.sqrt(image_x ** 2 + image_y ** 2)
    thetas = np.arctan2(image_y, image_x)

    rtheta = np.arange(-pi - EPS, pi, 2 * pi / BIN_COUNT)
    all_hists = []
    for i in range(0, length, CELL_SIZE[0]):
        all_hists.append([])
        for j in range(0, width, CELL_SIZE[1]):
            histogram = np.histogram(thetas[i: i + CELL_SIZE[0], j: j + CELL_SIZE[1], :], bins=BIN_COUNT, range=(-pi, pi), weights=G[i: i + CELL_SIZE[0], j: j + CELL_SIZE[1], :])
            all_hists[-1].append(histogram[0])
    all_blocks = []
    for i in range(0, len(all_hists) - 2, 2):
        for j in range(0, len(all_hists[0]) - 2, 2):
            all_blocks.append(norm(np.concatenate((all_hists[i][j], all_hists[i + 1][j], all_hists[i][j + 1], all_hists[i + 1][j + 1], 
                                                   all_hists[i + 2][j], all_hists[i][j + 2], all_hists[i + 2][j + 1], all_hists[i + 1][j + 2], all_hists[i + 2][j + 2]))))
    descriptor = np.concatenate(tuple(all_blocks))
    return descriptor

def fit_and_classify(train_features, train_labels, test_features):

    #kernel='rbf', const = 11.11 or 67.76

    train_features1, train_labels1 = shuffle(train_features, train_labels)
    clf = SVC(C=67.76, kernel='rbf')
    clf.fit(train_features1, train_labels1)

    return clf.predict(test_features)

