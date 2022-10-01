import numpy as np
import math

def get_bayer_masks(n_rows, n_cols):
    part = np.zeros((2, 2, 3), dtype='bool')
    part[0][1][0] = 1
    part[0][0][1] = 1
    part[1][1][1] = 1
    part[1][0][2] = 1
    odd_piece = np.zeros((1, 2, 3), dtype='bool')
    odd_piece[0][1][0] = 1
    odd_piece[0][0][1] = 1
    mask = np.tile(part, (n_rows // 2, 1, 1))
    if n_rows & 1:
        mask = np.concatenate((mask, odd_piece))
    mask = np.tile(mask, (1, n_cols // 2, 1))
    if n_cols & 1:
        mask = np.concatenate((mask, mask[:,0:1,:]), axis=1)
    return mask
def get_colored_img(raw_img):
    mask = get_bayer_masks(len(raw_img), len(raw_img[0])).astype('uint8')
    for i in range(3):
        mask[:,:,i] = mask[:,:,i] * raw_img
    return mask
def bilinear_interpolation(colored_img):
    colored_img = colored_img.astype('uint64')
    length = len(colored_img)
    width = len(colored_img[0])
    for row in range(1, length - 1):
        for col in range(1, width - 1):
            if (row & 1) == (col & 1):
                cnt = 2
                if (row & 1) == 0:
                    total = colored_img[row, col - 1, 0] + colored_img[row, col + 1, 0]
                    colored_img[row, col, 0] = total // cnt
                else:
                    total = colored_img[row - 1, col, 0] + colored_img[row + 1, col, 0]
                    colored_img[row, col, 0] = total // cnt
            elif (row & 1) and not(col & 1):
                cnt = 4
                total = colored_img[row - 1, col - 1, 0] + colored_img[row + 1, col - 1, 0] + colored_img[row - 1, col + 1, 0] + colored_img[row + 1, col + 1, 0]
                colored_img[row, col, 0] = total // cnt
    for row in range(1, length - 1):
        for col in range(1, width - 1):
            if (row & 1) != (col & 1):
                cnt = 4
                total = colored_img[row, col - 1, 1] + colored_img[row, col + 1, 1] + colored_img[row - 1, col, 1] + colored_img[row + 1, col, 1]
                colored_img[row, col, 1] = total // cnt
    for row in range(1, length - 1):
        for col in range(1, width - 1):
            if (row & 1) == (col & 1):
                cnt = 2
                if (row & 1) == 0:
                    total = colored_img[row - 1, col, 2] + colored_img[row + 1, col, 2]
                    colored_img[row, col, 2] = total // cnt
                else:
                    total = colored_img[row, col - 1, 2] + colored_img[row, col + 1, 2]
                    colored_img[row, col, 2] = total // cnt
            elif not(row & 1) and (col & 1):
                cnt = 4
                total = colored_img[row - 1, col - 1, 2] + colored_img[row + 1, col - 1, 2] + colored_img[row - 1, col + 1, 2] + colored_img[row + 1, col + 1, 2]
                colored_img[row, col, 2] = total // cnt
    return colored_img.astype('uint8')


def count_green(img, rimg, length, width):
    for row in range(2, length - 2):
        for col in range(2, width - 2):
            if (row & 1) == (col & 1):
                img[row, col, 1] = rimg[row, col]
            else:
                total = (rimg[row-1,col] + rimg[row+1,col] + rimg[row,col-1] + rimg[row,col+1]) / 4
                total += (rimg[row,col] - (rimg[row+2,col] + rimg[row-2,col] + rimg[row,col+2] + rimg[row,col-2]) / 4) / 2
                img[row, col, 1] = total
def count_red(img, rimg, length, width):
    for row in range(2, length, 2):
        for col in range(3, width, 2):
            img[row, col, 0] = rimg[row, col]
    for row in range(3, length - 2, 2):
        for col in range(2, width - 2, 2):
            total = 3 / 4 * rimg[row,col] - 3 / 16 * (rimg[row-2,col] + rimg[row+2,col] + rimg[row,col-2] + rimg[row,col+2])
            total += 1 / 4 * (rimg[row+1,col+1] + rimg[row+1,col-1] + rimg[row-1,col-1] + rimg[row-1,col+1])
            img[row, col, 0] = total
    for row in range(2, length - 2, 2):
        for col in range(2, width - 2, 2):
            total = (5 * rimg[row,col] + 4 * (rimg[row,col-1] + rimg[row,col+1]) + (rimg[row-2,col] + rimg[row+2,col]) / 2) / 8
            total -= (rimg[row,col-2] + rimg[row,col+2] + rimg[row+1,col+1] + rimg[row+1,col-1] + rimg[row-1,col-1] + rimg[row-1,col+1]) / 8
            img[row, col, 0] = total
    for row in range(3, length - 2, 2):
        for col in range(3, width - 2, 2):
            total = (5 * rimg[row,col] + 4 * (rimg[row-1,col] + rimg[row+1,col]) + (rimg[row,col-2] + rimg[row,col+2]) / 2) / 8
            total -= (rimg[row-2,col] + rimg[row+2,col] + rimg[row+1,col+1] + rimg[row+1,col-1] + rimg[row-1,col-1] + rimg[row-1,col+1]) / 8
            img[row, col, 0] = total
def count_blue(img, rimg, length, width):
    for row in range(3, length, 2):
        for col in range(2, width, 2):
            img[row, col, 2] = rimg[row, col]
    for row in range(2, length - 2, 2):
        for col in range(3, width - 2, 2):
            total = 1 / 8 * (6 * rimg[row,col] - 3 * (rimg[row-2,col] + rimg[row+2,col] + rimg[row,col-2] + rimg[row,col+2]) / 2)
            total += 1 / 4 * (rimg[row+1,col+1] + rimg[row+1,col-1] + rimg[row-1,col-1] + rimg[row-1,col+1])
            img[row, col, 2] = total
    for row in range(3, length - 2, 2):
        for col in range(3, width - 2, 2):
            total = (5 * rimg[row,col] + 4 * (rimg[row,col-1] + rimg[row,col+1]) + (rimg[row-2,col] + rimg[row+2,col]) / 2) / 8
            total -= (rimg[row,col-2] + rimg[row,col+2] + rimg[row+1,col+1] + rimg[row+1,col-1] + rimg[row-1,col-1] + rimg[row-1,col+1]) / 8
            img[row, col, 2] = total
    for row in range(2, length - 2, 2):
        for col in range(2, width - 2, 2):
            total = (5 * rimg[row,col] + 4 * (rimg[row-1,col] + rimg[row+1,col]) + (rimg[row,col-2] + rimg[row,col+2]) / 2) / 8
            total -= (rimg[row-2,col] + rimg[row+2,col] + rimg[row+1,col+1] + rimg[row+1,col-1] + rimg[row-1,col-1] + rimg[row-1,col+1]) / 8
            img[row, col, 2] = total
def improved_interpolation(raw_img):
    rimg = raw_img.astype('float')
    length = len(raw_img)
    width = len(raw_img[0])
    img = np.zeros((length, width, 3), dtype='int64')
    count_green(img, rimg, length, width)
    count_red(img, rimg, length, width)
    count_blue(img, rimg, length, width)
    return np.clip(img, 0, 255).astype('uint8')

def compute_psnr(img_pred, img_gt):
    shape = np.shape(img_pred)
    img_pred = img_pred.astype('float64')
    img_gt = img_gt.astype('float64')
    mse = np.sum((img_pred - img_gt) ** 2) / shape[0] / shape[1] / shape[2]
    if mse == 0.:
        raise ValueError
    return 10 * math.log10(np.max(img_gt) ** 2 / mse)
