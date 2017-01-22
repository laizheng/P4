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
    filter = Filter(image_paths = "./project_video_images/*.jpg")
    #filter = Filter(image_paths="./project_video_images_debug/*.jpg")
    #filter.image_paths = ["./project_video_images/001.jpg"]
    for path in filter.image_paths:
        print("processing...{}".format(path))
        img = filter.imread(path)
        if img.shape[2]==4:
            img = img[:,:,:3]
        ret = filter.pipline(img)
        plt.figure(figsize=(16,10))
        plt.imshow(filter.diagScreen)
        plt.subplots_adjust(left=0.03,bottom=0.03,right=1,top=1)
        plt.show()

if __name__ == "__main__":
    main()