import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
from skimage.transform import ProjectiveTransform
from skimage.transform import warp
from skimage.filters import gaussian
from numpy.linalg import inv
from math import sqrt

DEFAULT_TRANSFORM = ProjectiveTransform

def find_orb(img, n_keypoints=1400):

    img_gr = rgb2gray(img)
    
    descriptor = ORB(n_keypoints=n_keypoints)

    descriptor.detect_and_extract(img_gr)
    keypoints = descriptor.keypoints
    descriptors = descriptor.descriptors
    return keypoints, descriptors


def center_and_normalize_points(points):

    pointsh = np.row_stack([points.T, np.ones((points.shape[0]), )])
    centre = [np.mean(points[..., 0]), np.mean(points[..., 1])]

    avg_points = points - centre
    avg = 0
    for point in avg_points:
        avg += sqrt(point[0] * point[0] + point[1] * point[1])
    avg /= len(points)
    N = sqrt(2) / avg

    matrix = np.array([[N,   0,  -N * centre[0]],
                       [0,   N,  -N * centre[1]],
                       [0,   0,        1      ]])

    tr_points = np.zeros(points.shape)
    length = len(tr_points)
    for i in range(length):
        tr_points[i] = np.matmul(matrix, pointsh[..., i])[:2]
    return matrix, tr_points


def find_homography(src_keypoints, dest_keypoints):

    src_matrix, src = center_and_normalize_points(src_keypoints)
    dest_matrix, dest = center_and_normalize_points(dest_keypoints)

    vectors = []
    length = len(src)
    for i in range(length):
        vector_x = np.array([-src[i][0], -src[i][1], -1, 0, 0, 0, dest[i][0] * src[i][0], dest[i][0] * src[i][1], dest[i][0]])
        vector_y = np.array([0, 0, 0, -src[i][0], -src[i][1], -1, dest[i][1] * src[i][0], dest[i][1] * src[i][1], dest[i][1]])
        vectors.append(vector_x)
        vectors.append(vector_y)
    vectors = np.array(vectors)
    svd = np.linalg.svd(vectors)
    H_row = svd[-1]
    H_row /= sqrt(np.sum(H_row ** 2))
    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            H[i][j] = H_row[-1][3 * i + j]
    H = np.matmul(np.matmul(inv(dest_matrix), H), src_matrix)
    return H


def take_ind(max_num, amount=4):
    '''returns list of 4 random different numbers from 0 to maxnum'''
    from numpy.random import randint
    ind = []
    while len(ind) != 4:
        tmp = randint(0, max_num)
        if tmp not in ind:
            ind.append(tmp)
    return ind


def ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors, max_trials=400, residual_threshold=2,  return_matches=False):
    from numpy import array as arr
    from numpy.linalg import norm

    pre_ind = match_descriptors(src_descriptors, dest_descriptors)
    ind_max = pre_ind.copy()

    close_src = [src_keypoints[j] for j in pre_ind[..., 0]]
    close_dst = [dest_keypoints[j] for j in pre_ind[..., 1]]
    
    MAX = 0

    for trial in range(max_trials):

        ind= take_ind(len(close_src))
        src_dots = [close_src[i] for i in ind]
        dst_dots = [close_dst[i] for i in ind]

        H = find_homography(arr(src_dots), arr(dst_dots))
        M = ProjectiveTransform(H)(close_src)
        M_len = len(M)

        optimal_ind = []

        for i in range(M_len):
            dst = norm(M[i] - arr(close_dst[i]))
            if dst < residual_threshold:
                optimal_ind.append(pre_ind[i])

        if MAX < len(optimal_ind):
            MAX = len(optimal_ind)
            ind_max = arr(optimal_ind)

    src = [src_keypoints[i] for i in ind_max[..., 0]]
    dest = [dest_keypoints[i] for i in ind_max[..., 1]]

    H = find_homography(arr(src), arr(dest))

    if not return_matches:
        return ProjectiveTransform(H)
    else:
        return ProjectiveTransform(H), ind_max


from numpy.linalg import inv
def find_simple_center_warps(forward_transforms):
    image_count = len(forward_transforms) + 1
    center_index = (image_count - 1) // 2

    result = [None] * image_count
    result[center_index] = DEFAULT_TRANSFORM()

    center_down = center_index - 1
    center_up = center_index
    while center_down >= 0:
        result[center_down] = ProjectiveTransform(np.matmul(forward_transforms[center_down].params, result[center_down + 1].params))
        center_down -= 1
    while center_up < len(result) - 1:
        result[center_up + 1] = ProjectiveTransform(np.matmul(result[center_up].params, inv(forward_transforms[center_up].params)))
        center_up += 1
    return tuple(result)


def get_corners(image_collection, center_warps):
    for img, transform in zip(image_collection, center_warps):
        height, width, _ = img.shape
        corners = np.array([[0, 0],
                            [height, 0],
                            [height, width],
                            [0, width]])

        yield transform(corners)[:, ::-1]


def get_min_max_coords(corners):
    corners = np.concatenate(corners)
    return corners.min(axis=0), corners.max(axis=0)


def get_final_center_warps(image_collection, simple_center_warps):
    corners = tuple(get_corners(image_collection, simple_center_warps))
    min_coords, max_coords = get_min_max_coords(corners)
    transform = ProjectiveTransform(np.array(((1, 0, -min_coords[1]), (0, 1, -min_coords[0]), (0, 0, 1))))
    final_center_warps = list(simple_center_warps)
    for i in range(len(final_center_warps)):
        final_center_warps[i] = ProjectiveTransform(np.matmul(transform.params,final_center_warps[i].params))
    return tuple(final_center_warps), (int(max_coords[1] - min_coords[1]), int(max_coords[0] - min_coords[0]))


def rotate_transform_matrix(transform):
    matrix = transform.params[(1, 0, 2), :][:, (1, 0, 2)]
    return type(transform)(matrix)


def warp_image(image, transform, output_shape):
    shape = image.shape
    mask = np.ones((shape[0], shape[1]), dtype=np.bool8)
    new_img = warp(image, (rotate_transform_matrix(transform)).inverse, output_shape=output_shape)
    mask = warp(mask, (rotate_transform_matrix(transform)).inverse, output_shape=output_shape)
    return new_img, mask


def merge_pano(image_collection, final_center_warps, output_shape):
    result = np.zeros(output_shape + (3,))
    for i in range(len(image_collection)-1, -1, -1):
        img, mask = warp_image(image_collection[i], final_center_warps[i], output_shape)
        for i in range(len(img)):
            for j in range(len(img[0])):
                if mask[i][j] == 1:
                    result[i][j][0] = img[i][j][0]
                    result[i][j][1] = img[i][j][1]
                    result[i][j][2] = img[i][j][2]
    return result


def get_gaussian_pyramid(image, n_layers=4, sigma=1):
    K = 1.8
    gaussian_pyramid = [image]
    for i in range(n_layers):
        ds_image = gaussian(image, sigma, multichannel=True)
        gaussian_pyramid.append(ds_image)
        sigma *= K
    return tuple(gaussian_pyramid)


def get_laplacian_pyramid(image, n_layers=4, sigma=1.5):
    gaussian_pyramid = get_gaussian_pyramid(image, n_layers, sigma)
    laplacian_pyramid = []
    for i in range(0, len(gaussian_pyramid) - 1):
        laplacian_pyramid.append(gaussian_pyramid[i] - gaussian_pyramid[i + 1])
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return tuple(laplacian_pyramid)


def merge_laplacian_pyramid(laplacian_pyramid):
    laplacian_pyramid1 = list(laplacian_pyramid)
    for j in range(len(laplacian_pyramid1) - 1, 0, -1):
        laplacian_pyramid1[j - 1] += laplacian_pyramid1[j]
    return laplacian_pyramid1[0]


def increase_contrast(image_collection):
    result = []

    for img in image_collection:
        img = img.copy()
        for i in range(img.shape[-1]):
            img[:, :, i] -= img[:, :, i].min()
            img[:, :, i] /= img[:, :, i].max()
        result.append(img)

    return result


def min_xcoord_avg(corner, size):
    '''returns min and pre_min elements of corner array'''
    minimum0 = corner[0][0]
    minimum1 = corner[0][0]

    for i in range(1, len(corner)):
        if corner[i][0] < minimum0:
            minimum1 = minimum0
            minimum0 = corner[i][0]
        elif corner[i][0] < minimum1:
            minimum1 = corner[i][0]

    return int(minimum1 + minimum0) // 2

def max_xcoord_avg(corner):
    '''returns max and pre_max elements of corner array'''
    max0 = corner[0][0]
    max1 = corner[0][0]

    for i in range(1, len(corner)):
        if corner[i][0] > max1:
            max0 = max1
            max1 = corner[i][0]
        elif corner[i][0] > max0:
            max0 = corner[i][0]

    return int(max1 + max0) // 2

def gaussian_merge_pano(image_collection, final_center_warps, output_shape, n_layers=4, image_sigma=4, merge_sigma=3):
    from numpy import array as arr

    high_contrast_image_collection = increase_contrast(image_collection)
    warped_img = []

    for i in range(len(image_collection)):
        new_img, mask = warp_image(image_collection[i], final_center_warps[i], output_shape)
        warped_img.append(new_img)

    min_x = []
    max_x = []
    corners = tuple(get_corners(image_collection, final_center_warps))
    for corner in corners:
        min_x.append(min_xcoord_avg(corner, output_shape[0]))
        max_x.append(max_xcoord_avg(corner))
    
    final_pyramid = arr(get_laplacian_pyramid(warped_img[0], n_layers, sigma=image_sigma)) * arr(get_gaussian_pyramid(np.ones(output_shape + (3,)), n_layers, sigma=merge_sigma))
    
    for j in range(1, len(warped_img)):
        mask = np.zeros(output_shape + (3,))
        cross_center = (max_x[j - 1] + min_x[j]) // 2
        mask[:, cross_center:, :] += 1
        final_pyramid *= np.ones(output_shape + (3,)) - arr(get_gaussian_pyramid(mask, n_layers, sigma=merge_sigma))
        final_pyramid += arr(get_laplacian_pyramid(warped_img[j], n_layers, sigma=image_sigma)) * arr(get_gaussian_pyramid(mask, n_layers, sigma=merge_sigma))
    
    return np.clip(merge_laplacian_pyramid(final_pyramid), 0, 1), corners[0], corners[-1]

