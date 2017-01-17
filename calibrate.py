import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import math
import pickle

def calCamera(nx=9,ny=6):
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    objpoints = []
    imgpoints = []
    images = glob.glob("./camera_cal/*.jpg")
    for idx, fname in enumerate(images):
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray,(nx,ny),None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print("{} detection failed".format(fname))
    img = mpimg.imread(images[2])
    img_size = (img.shape[1],img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle,open("dist_pickle.p","wb"))
    return mtx, dist

def main():
    mtx, dist = calCamera()
    images = glob.glob('./camera_cal/*.jpg')
    img = mpimg.imread(images[0])
    dst = cv2.undistort(img,mtx,dist)
    f, (ax1,ax2) = plt.subplots(1,2,figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title("Original Image")
    ax2.imshow(dst)
    ax2.set_title("Undistorted Image")
    plt.show()

if __name__ == "__main__":
    main()