from skimage import color
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn

def depthFromStereo(img1, img2, ws, method):
    image_gray1 = color.rgb2gray(img1)
    image_gray2 = color.rgb2gray(img2)
    baseline = 1
    focal = 1
    window = 55
    depthImage = np.zeros(image_gray2.shape)
    height, width = image_gray1.shape
    for y in range(ws, height - ws):
        print("\rProcessing.. %d%% complete" % (y / (height - ws) * 100), end="", flush=True)
        for x in range(ws+window, width - ws):

            patch_left = image_gray1[y - ws:y + ws + 1, x - ws:x + ws + 1]

            similarityMeasure = None
            if method == 'SSD': similarityMeasure = SSD
            elif method == 'SAD': similarityMeasure = SAD
            elif method == 'COR': similarityMeasure = COR

            best_offset = similarityMeasure(y, x, ws, patch_left, image_gray2, window)

            depthImage[y, x] = best_offset

    ## Disparity plot
    # plt.imshow(depthImage, cmap='hot', interpolation='nearest')
    # plt.axis('off')
    # plt.show()

    depth = baseline * focal / (depthImage + 1e-1)

    depth_cutoff = 0.08

    depth[depth > depth_cutoff] = 0

    ## Cropping the borders
    depth = depth[ws:-ws, ws + window:-ws]

    return depth

@staticmethod
@jit(nopython = True, parallel = False, cache = True)
def SSD(y, x, ws, patch_left, image2, window):
    prev_ssd = np.inf
    best_match = 0
    for search in range(window):
        patch_right = image2[y - ws:y + ws + 1, x - ws - search:x + ws + 1 - search]
        ssd = np.sum((patch_left - patch_right) ** 2)

        if ssd < prev_ssd:
            prev_ssd = ssd
            best_match = search

    return best_match

@staticmethod
@jit(nopython=True, parallel=False, cache=True)
def SAD(y, x, ws, patch_left, image2, window):
    prev_sad = np.inf
    best_match = 0
    for search in range(window):
        patch_right = image2[y - ws:y + ws + 1, x - ws - search:x + ws + 1 - search]
        sad = np.sum(np.absolute(patch_left - patch_right))
        if sad < prev_sad:
            prev_sad = sad
            best_match = search

    return best_match

@staticmethod
@jit(nopython = True, parallel = False, cache = True)
def COR(y, x, ws, patch_left, image2, window):
    prev_corr = np.inf
    best_match = 0
    for search in range(window):
        patch_right = image2[y - ws:y + ws + 1, x - ws - search:x + ws + 1 - search]
        l = patch_left - np.mean(patch_left)
        r = patch_right - np.mean(patch_right)
        corr = - np.sum( l * r / (np.sqrt(np.sum(l ** 2) * np.sum(r ** 2) ) ) )
        if corr < prev_corr:
            prev_corr = corr
            best_match = search

    return best_match