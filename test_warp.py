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
    filter = Filter(image_paths="./warp_test_images/*.jpg")
    img = filter.imread(filter.image_paths[0])
    undist = filter.undistort(img)
    undist = cv2.resize(undist, (1280, 738))
    filter.warp(undist,dbgcode=1)

if __name__ == "__main__":
    main()

