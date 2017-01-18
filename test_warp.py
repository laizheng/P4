import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import math
import pickle
from filters import Filter
from moviepy.editor import VideoFileClip

def main():
    #filter = Filter(image_paths="./warp_test_images/*.jpg")
    filter = Filter(image_paths="./project_video_images_debug/*.jpg")
    filter.image_paths = ["./project_video_images_debug/23.jpg","./warp_test_images/1.jpg"]
    for path in filter.image_paths:
        img = filter.imread(path)
        undist = filter.undistort(img)
        undist = cv2.resize(undist, (1280, 738))
        filter.warp(undist,dbgcode=1)

if __name__ == "__main__":
    main()

