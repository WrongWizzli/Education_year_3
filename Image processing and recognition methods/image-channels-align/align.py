import numpy as np
import math

DISTANCE = 20
MAX_SIZE = 1000

def divide(img):
    length = len(img)
    width = len(img[0])
    boarder = 0.05
    red = img[0 : length // 3,:]
    green = img[length // 3 : 2 * (length // 3),:]
    blue = img[2 * (length // 3) : 3 * (length // 3),:]
    red = red[int(boarder * len(red)) : int(len(red) * (1 - boarder)), int(boarder * width) : int(width * (1 - boarder))]
    green = green[int(boarder * len(green)) : int(len(green) * (1 - boarder)), int(boarder * width) : int(width * (1 - boarder))]
    blue = blue[int(boarder * len(blue)) : int(len(blue) * (1 - boarder)), int(boarder * width) : int(width * (1 - boarder))]
    return red, green, blue

def mse(img1, img2):
    return 1 / (len(img1) * len(img1[0])) * np.sum((img1 - img2) ** 2)
def norm(img1, img2):
    return np.sum(img1 * img2) / math.sqrt(np.sum(img1 ** 2) * np.sum(img2 ** 2))


def find_offset(red, green, blue, right=15):
    shape = np.shape(green)
    r_off = np.array((-right, -right), dtype=int)
    b_off = np.array((-right, -right), dtype=int)
    mse_r = mse(red[0: shape[0] - 2*right, 0: shape[1] - 2*right], green[right: shape[0] - right, right: shape[1] - right])
    mse_b = mse(blue[0: shape[0] - 2 * right, 0: shape[1] - 2 * right], green[right: shape[0] - right, right: shape[1] - right])
    for i in range(-right, right + 1):
        for j in range(-right, right + 1):
            t_mse_r = mse(red[right + i: shape[0] - right - 1 + i, right + j: shape[1] - right - 1 + j], green[right: shape[0] - right - 1, right: shape[1] - right - 1])
            if t_mse_r < mse_r:
                r_off[0], r_off[1] = i, j
                mse_r = t_mse_r
    for i in range(-right, right + 1):
        for j in range(-right, right + 1):
            t_mse_b = mse(blue[right + i: shape[0] - right - 1 + i, right + j: shape[1] - right - 1 + j], green[right: shape[0] - right - 1, right: shape[1] - right - 1])
            if t_mse_b < mse_b:
                b_off[0], b_off[1] = i, j
                mse_b = t_mse_b
    return tuple(r_off), tuple(b_off)

def pool(k, r, g, b):
    length = len(r) // k
    width = len(r[0]) // k
    new_r = np.zeros((length, width))
    new_g = np.zeros((length, width))
    new_b = np.zeros((length, width))
    for i in range(length):
        for j in range(width):
            new_r[i][j] = np.sum(r[k * i: k * (i + 1), k * j: k * (j + 1)]) / k
            new_g[i][j] = np.sum(g[k * i: k * (i + 1), k * j: k * (j + 1)]) / k
            new_b[i][j] = np.sum(b[k * i: k * (i + 1), k * j: k * (j + 1)]) / k
    return new_r, new_g, new_b

def transform(r, g, b, shape):
    k = 1
    new_r, new_g, new_b = r, g, b
    while shape[0] // k > MAX_SIZE or shape[1] // k > MAX_SIZE:
        k *= 2
    new_r, new_g, new_b = pool(k, r, g, b)
    return k, new_r, new_g, new_b

def align(img, g_coord):
    is_large = False
    img = img.astype('float64')
    k = 1
    off = len(img) // 3
    red, green, blue = divide(img)
    shape = np.shape(green)
    if len(red) > MAX_SIZE and len(red[0]) > MAX_SIZE:
        temp_red = red
        temp_blue = blue
        temp_green = green
        k, temp_red, temp_green, temp_blue = transform(temp_red, temp_green, temp_blue, shape)
        r_off, b_off = find_offset(temp_red, temp_green, temp_blue, DISTANCE)
    else:
        r_off, b_off = find_offset(red, green, blue)
    aligned = np.dstack((red[int(DISTANCE + r_off[0]): int(shape[0] - DISTANCE - 1 + r_off[0]), int(DISTANCE + r_off[1]): int(shape[1] - DISTANCE - 1 + r_off[1])], 
                             green[DISTANCE: int(shape[0] - DISTANCE - 1), DISTANCE: int(shape[1] - DISTANCE - 1)],
                             blue[int(DISTANCE + b_off[0]): int(shape[0] - DISTANCE - 1 + b_off[0]), int(DISTANCE + b_off[1]): int(shape[1] - DISTANCE - 1 + b_off[1])]))
    return (aligned * 256).astype('uint8'), (g_coord[0] + r_off[0] * k - off, g_coord[1] + r_off[1] * k), (g_coord[0] + b_off[0] * k + off, g_coord[1] + b_off[1] * k)

            