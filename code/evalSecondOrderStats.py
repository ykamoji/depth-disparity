import numpy as np
import skimage.io as sio
import os
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn
#

k=0.03
def second_order_statistics(ix, iy, ws):
    height, width = ix.shape

    Ixx = np.square(ix)
    Iyy = np.square(iy)
    Ixy = ix * iy

    offset = ws // 2
    adjust = 1
    if ws % 2 == 0:
        adjust = 0

    R = np.zeros((height, width))
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            Sxx = np.sum(Ixx[y - offset:y + offset + adjust, x - offset:x + offset + adjust])
            Syy = np.sum(Iyy[y - offset:y + offset + adjust, x - offset:x + offset + adjust])
            Sxy = np.sum(Ixy[y - offset:y + offset + adjust, x - offset:x + offset + adjust])

            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            R[y-offset, x-offset] = det - (k * (trace ** 2))

    return R


# read image
im_dir = "../data/disparity"
savedir = "../output/harris/"

sobelFilter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

for image_file in ["teddy_im2.png","cones_im2.png"]:
    image = sio.imread(os.path.join(im_dir, image_file))
    # convert image to gray
    image = rgb2gray(image)


    # compute differentiations along x and y axis respectively
    # x-diff
    #--------- add your code here ------------------#
    x_grad = convolve2d(image, sobelFilter, mode='same')
    Ix = x_grad / np.max(x_grad)

    # y-diff
    #--------- add your code here ------------------#
    y_grad = convolve2d(image, sobelFilter.T, mode='same')
    Iy = y_grad / np.max(y_grad)

    # set window size
    #--------- modify this accordingly ------------------#
    for ws in [5, 10]:

        heatMapImg = second_order_statistics(Ix, Iy, ws)

        plt.imshow(heatMapImg, cmap='hot')
        # plt.colorbar()
        plt.axis('off')
        plt.title(f" {image_file}, WindowSize={ws}")
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, 'ws-'+str(ws)+'_'+image_file),  bbox_inches='tight', edgecolor='auto')
        # plt.show()
