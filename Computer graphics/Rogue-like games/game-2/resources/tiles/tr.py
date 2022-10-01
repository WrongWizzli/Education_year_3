from glob import glob
import numpy as np
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
from matplotlib import pyplot as plt

a = sorted(glob('./*hole*.png'))
print(a)
Multiplier = 2
for i in a:
    img = imread(i)
    img[:,:,:3] //= 2
    imsave(i, img)