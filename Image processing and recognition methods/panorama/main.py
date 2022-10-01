import panorama
from skimage import io
import numpy as np
import sys

def find_x_boarders(lc, rc, size):
    minimum0 = size
    minimum1 = size
    for i in range(len(lc)):
        if lc[i][0] < minimum0:
            minimum1 = minimum0
            minimum0 = lc[i][0]
        elif lc[i][0] < minimum1:
            minimum1 = lc[i][0]

    max0 = rc[0][0]
    max1 = rc[0][0]
    for i in range(len(rc)):
        if rc[i][0] > max1:
            max0 = max1
            max1 = rc[i][0]
        elif rc[i][0] > max0:
            max0 = rc[i][0]

    return [int(minimum1) + 1, int(max0)]

def fix(result, boarders):
    new_res = result[:, boarders[0]:boarders[1],:]
    shape = new_res.shape
    for i in range(len(new_res) // 2, 0, -1):
        if np.count_nonzero(new_res[i,:,:]) != shape[1] * shape[2]:
            y_cut0 = i
            break
    for i in range(len(new_res) // 2, len(new_res)):
        if np.count_nonzero(new_res[i,:,:]) != shape[1] * shape[2]:
            y_cut1 = i
            break
    return new_res[y_cut0:y_cut1, ...]


pano_image_collection = io.ImageCollection(sys.argv[1],                                      #the path to the N images to be glued together
                                           load_func=lambda f: io.imread(f).astype(np.float64) / 255)
                                        
keypoints, descriptors = zip(*(panorama.find_orb(img) for img in pano_image_collection))
forward_transforms = tuple(panorama.ransac_transform(src_kp, src_desc, dest_kp, dest_desc)
                           for src_kp, src_desc, dest_kp, dest_desc
                           in zip(keypoints[:-1], descriptors[:-1], keypoints[1:], descriptors[1:]))
simple_center_warps = panorama.find_simple_center_warps(forward_transforms)
final_center_warps, output_shape = panorama.get_final_center_warps(pano_image_collection, simple_center_warps)
result, left_corner, right_corner = panorama.gaussian_merge_pano(pano_image_collection, final_center_warps, output_shape)
x_boarders = find_x_boarders(left_corner, right_corner, len(result[0]))
result = fix(result, x_boarders)
io.imsave('./test.jpeg', result)
