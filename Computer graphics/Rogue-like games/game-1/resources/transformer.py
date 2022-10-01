from glob import glob
import numpy as np
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
from matplotlib import pyplot as plt
import PIL
a = sorted(glob('./textures/*'))
def resiz(img, Multiplier):
   return (255 * resize(img, (Multiplier * img.shape[0], Multiplier * img.shape[1], 4))).astype('uint8').clip(0, 255) 
fname4 = './textures/wall_corner_left'
fname5 = './textures/wall_inner_corner_t_top_rigth'
fname6 = './textures/wall_inner_top'
img4 = imread('./{0}.png'.format(fname4))
img5 = imread('./{0}.png'.format(fname5))
img6 = imread('./{0}.png'.format(fname6))
img5 = resiz(img5, 2)
img6 = resiz(img6, 2)
img61 = img6[:,::-1,:]
img51 = img5[::-1,:,:]
def boarder_img(img , symmap):
    for i in range(0, img.shape[0], 32):
        for j in range(0, img.shape[1], 32):
            if symmap[i][j] == 1:
                if (symmap[i][j - 32] == 0):
                    for k in range(32):
                        for m in range(32):
                            if img61[k, m, :3].sum() != 0:
                                img[i + k, j + m, :3] = img61[k, m, :3]
                if (symmap[i][j + 32] == 0):
                    for k in range(32):
                        for m in range(32):
                            if img6[k, m, :3].sum() != 0:
                                img[i + k, j + m, :3] = img6[k, m, :3]
                if (symmap[i+32][j] == 0):
                    for k in range(32):
                        for m in range(32):
                            if img5[k, m, :3].sum() != 0:
                                img[i + k, j + m, :3] = img5[k, m, :3]
                if (symmap[i-32][j] == 0):
                    for k in range(32):
                        for m in range(32):
                            if img51[k, m, :3].sum() != 0:
                                img[i + k, j + m, :3] = img51[k, m, :3]
                
                      
name = 'lose_slide'
img = imread('{0}.png'.format(name))
img = ((255 * resize(img, (666, 1413, 4))).astype('uint8').clip(0, 255))[:,::-1,:]
plt.imshow(img)
plt.show()
with open('./{0}.txt'.format(name), 'w') as tfile:
    for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                tfile.write(str(img[i][j][0]) + ' ' + str(img[i][j][1]) + ' ' + str(img[i][j][2]) + ' ' + str(img[i][j][3]) + ' ')
            tfile.write('\n')
