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
    #filter = Filter(image_paths = "./project_video_images/*.jpg")
    filter = Filter(image_paths="./project_video_images_debug/*.jpg")
    for path in filter.image_paths:
        print("processing...{}".format(path))
        img = filter.imread(path)
        if img.shape[2]==4:
            img = img[:,:,:3]
        undist = filter.undistort(img)
        undist = cv2.resize(undist, (1280, 738))
        combined_thresholing, color_binary = filter.thresholding(undist)

if __name__ == "__main__":
    main()