
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
img = imread('./mainlevbuild.png')
plt.imshow(img[400:, 400:, :])
plt.show()