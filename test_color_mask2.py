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

    #filter.image_paths = ["./project_video_images_debug/17.jpg"]
    #filter.image_paths = ["./project_video_images_debug/9.jpg"]
    filter.image_paths = ["./project_video_images_debug/20.jpg","./project_video_images_debug/21.jpg" \
        , "./project_video_images_debug/23.jpg"]
    for path in filter.image_paths:
        print("processing...{}".format(path))
        img = filter.imread(path)
        if img.shape[2]==4:
            img = img[:,:,:3]
        undist = filter.undistort(img)
        undist = cv2.resize(undist, (1280, 738))
        hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
        H = hls[:, :, 0]
        L = hls[:, :, 1]
        S = hls[:, :, 2]
        S = (S/np.max(S) * 255).astype(np.uint8)
        #H = cv2.equalizeHist(H)
        hls_binary = np.zeros_like(H)
        hls_binary[ ((H <= 24) & (H>=18)) & (S>=100) ] = 1
        #hls_binary[(H <= 100)] = 1
        #hls_binary[(S >= 30)] = 1
        hsv = cv2.cvtColor(undist, cv2.COLOR_RGB2HSV)
        V = hsv[:, :, 2]

        gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
        gray_binary = np.zeros_like(gray)
        gray_binary[gray >= 200] = 1

        color_binary = np.zeros_like(gray)
        color_binary[(gray_binary == 1) | (hls_binary == 1)] = 1
        warped = filter.warp(color_binary)

        f, axes = plt.subplots(2,4,figsize=(12,12))
        axes[0,0].imshow(H,cmap='gray')
        axes[0,0].set_title("H")
        axes[0,1].imshow(warped, cmap='gray')
        axes[0,1].set_title("warped")
        axes[0,2].imshow(S, cmap='gray')
        axes[0,2].set_title("S")
        axes[0,3].imshow(hls_binary, cmap='gray')
        axes[0,3].set_title("hls_binary")

        axes[1, 0].imshow(gray, cmap='gray')
        axes[1, 0].set_title("gray")
        axes[1, 1].imshow(gray_binary, cmap='gray')
        axes[1, 1].set_title("gray_binary")
        
        axes[1, 2].imshow(color_binary, cmap='gray')
        axes[1, 2].set_title("color_binary")
        axes[1, 2].plot(filter.src[0][0], filter.src[0][1], '.', markersize=12)
        axes[1, 2].plot(filter.src[1][0], filter.src[1][1], '.', markersize=12)
        axes[1, 2].plot(filter.src[2][0], filter.src[2][1], '.', markersize=12)
        axes[1, 2].plot(filter.src[3][0], filter.src[3][1], '.', markersize=12)
        lleft = plt.Line2D((filter.src[0, 0], filter.src[1, 0]), (filter.src[0, 1], filter.src[1, 1]))
        axes[1, 2].add_line(lleft)
        lright = plt.Line2D((filter.src[2, 0], filter.src[3, 0]), (filter.src[2, 1], filter.src[3, 1]))
        axes[1, 2].add_line(lright)
        
        axes[1, 3].imshow(img, cmap='gray')
        axes[1, 3].set_title("original")

        #for ax in axes:
        #    ax.set_xticks([])
        #    ax.set_yticks([])
        f.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=0.1,hspace=0.1)
        plt.show()
        pass
        #combined_thresholing, color_binary = filter.thresholding(undist)

if __name__ == "__main__":
    main()