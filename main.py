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
    filter = Filter(image_paths="./project_video_images/*.jpg")
    filter.toVideo(input_video_file_name="project_video.mp4",output_video_file_name="project_video_out.mp4")

if __name__ == "__main__":
    main()