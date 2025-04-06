import matplotlib.pyplot as plt
from utils import imread
from depthFromStereo import depthFromStereo
import os

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn
#

read_path = "../data/disparity/"
images = [["teddy_im2.png", "teddy_im6.png"], ["cones_im2.png","cones_im6.png"]]
methods = ["SSD","SAD","COR"]
wss = [3, 7, 11, 15, 21]
for im_name1, im_name2 in images:
    #Read test images
    img1 = imread(os.path.join(read_path, im_name1))
    img2 = imread(os.path.join(read_path, im_name2))

    for method in methods:
        for ws in wss:
            imageName = im_name1.split("_")[0]
            print(f"\nComputing {imageName} with method {method} for window size {ws}")
            #Compute depth
            depth = depthFromStereo(img1, img2, ws, method)
            #Show result
            plt.imshow(depth, cmap='hot', interpolation='nearest')
            plt.axis('off')
            plt.title(f"WindowSize={ws}, Method={method}")
            save_path = "../output/disparity/"
            save_file = imageName +f"_{method}"+f"_w_{ws}"+'.png'
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, save_file), bbox_inches='tight', edgecolor='auto')
            # plt.show()