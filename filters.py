import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import math
import pickle
from moviepy.editor import VideoFileClip


class Filter():
    def __init__(self, image_paths=None):
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

        self.img = None
        self.image_paths = glob.glob(image_paths)
        f = open('dist_pickle.p', 'rb')
        dist_pickle = pickle.load(f)
        self.mtx = dist_pickle["mtx"]
        self.dist = dist_pickle["dist"]

        # slb = (265,737)
        # slt = (605,460)
        # srb = (1172,737)
        # srt = (683,460)

        #slb = (0, 680)
        #slt = (565, 460)

        #srb = (1279, 680)
        #srt = (715, 460)

        slb = (0, 650)
        slt = (555, 460)

        srb = (1279, 650)
        srt = (720, 460)

        self.src = np.float32([slb, slt, srb, srt])

        dlb = (slb[0],737)
        drb = (srb[0],737)
        dlt = (slb[0], 0)
        drt = (srb[0], 0)
        self.dst = np.float32([dlb, dlt, drb, drt])

        # Sliding Box
        self.last_left_start = 0
        self.last_right_start = 1280
        self.box_width = 50
        self.box_height = 50
        self.box_step_horizontal = 1
        self.box_step_vertical = 25

        # Cur Rad
        self.dashed_line_length_in_pixel = 62
        self.lane_width_in_pixel = 900
        self.left_curverads = []
        self.right_curverads = []
        self.offsets_meter = []

        # Diag Info
        self.font = cv2.FONT_HERSHEY_COMPLEX
        self.resetDiag()

        # Frame #
        self.frameNum = 0

    def resetDiag(self):
        self.mainDiagScreen = np.zeros((100, 100, 3), dtype=np.uint8)
        self.diag1 = np.zeros((100, 100, 3), dtype=np.uint8)
        self.diag2 = np.zeros((100, 100, 3), dtype=np.uint8)
        self.diag3 = np.zeros((100, 100, 3), dtype=np.uint8)
        self.diag4 = np.zeros((100, 100, 3), dtype=np.uint8)
        self.diag5 = np.zeros((100, 100, 3), dtype=np.uint8)
        self.diag6 = np.zeros((100, 100, 3), dtype=np.uint8)
        self.diag7 = np.zeros((100, 100, 3), dtype=np.uint8)
        self.diag8 = np.zeros((100, 100, 3), dtype=np.uint8)
        self.diag9 = np.zeros((100, 100, 3), dtype=np.uint8)
        self.diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)

    def imread(self, file_path):
        return mpimg.imread(file_path)

    def grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def to3D(self, img):
        if len(img.shape) < 3:
            img = img / np.max(img) * 255
            return np.dstack((img, img, img))
        else:
            return img

    def diagScreenUpdate(self):
        self.mainDiagScreen = self.to3D(self.mainDiagScreen)
        self.diag1 = self.to3D(self.diag1)
        self.diag2 = self.to3D(self.diag2)
        self.diag3 = self.to3D(self.diag3)
        self.diag4 = self.to3D(self.diag4)
        self.diag5 = self.to3D(self.diag5)
        self.diag6 = self.to3D(self.diag6)
        self.diag7 = self.to3D(self.diag7)
        self.diag8 = self.to3D(self.diag8)
        self.diag9 = self.to3D(self.diag9)

        middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)
        cv2.putText(middlepanel, 'Frame#: {}'.format(self.frameNum), \
                    (30, 30), self.font, 1, (255, 0, 0), 2)
        cv2.putText(middlepanel, 'Estimated lane curvature: {:.2f}m'.format(
            0.5 * (self.left_curverads[-1] + self.right_curverads[-1])), \
                    (30, 60), self.font, 1, (255, 0, 0), 2)
        cv2.putText(middlepanel, 'Estimated Meters right of center: {:.2f}m'.format(self.offsets_meter[-1]), \
                    (30, 90), self.font, 1, (255, 0, 0), 2)

        # assemble the screen example
        self.diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.diagScreen[0:720, 0:1280] = cv2.resize(self.mainDiagScreen, (1280, 720), interpolation=cv2.INTER_AREA)

        self.diagScreen[0:240, 1280:1600] = cv2.resize(self.diag1, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(self.diagScreen[0:240, 1280:1600], 'diag1', (30, 60), self.font, 1, (255, 0, 0), 2)

        self.diagScreen[0:240, 1600:1920] = cv2.resize(self.diag2, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(self.diagScreen[0:240, 1600:1920], 'diag2', (30, 60), self.font, 1, (255, 0, 0), 2)

        self.diagScreen[240:480, 1280:1600] = cv2.resize(self.diag3, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(self.diagScreen[240:480, 1280:1600], 'diag3', (30, 60), self.font, 1, (255, 0, 0), 2)

        self.diagScreen[240:480, 1600:1920] = cv2.resize(self.diag4, (320, 240), interpolation=cv2.INTER_AREA) * 4
        cv2.putText(self.diagScreen[240:480, 1600:1920], 'diag4', (30, 60), self.font, 1, (255, 0, 0), 2)

        self.diagScreen[840:1080, 0:320] = cv2.resize(self.diag5, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(self.diagScreen[840:1080, 0:320], 'diag5', (30, 60), self.font, 1, (255, 0, 0), 2)

        self.diagScreen[840:1080, 320:640] = cv2.resize(self.diag6, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(self.diagScreen[840:1080, 320:640], 'diag6', (30, 60), self.font, 1, (255, 0, 0), 2)

        self.diagScreen[840:1080, 640:960] = cv2.resize(self.diag7, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(self.diagScreen[840:1080, 640:960], 'diag7', (30, 60), self.font, 1, (255, 0, 0), 2)

        self.diagScreen[720:840, 0:1280] = middlepanel

        self.diagScreen[840:1080, 960:1280] = cv2.resize(self.diag8, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(self.diagScreen[840:1080, 960:1280], 'diag8', (30, 60), self.font, 1, (255, 0, 0), 2)

        self.diagScreen[600:1080, 1280:1920] = cv2.resize(self.diag9, (640, 480), interpolation=cv2.INTER_AREA) * 4
        cv2.putText(self.diagScreen[600:1080, 1280:1920], 'diag9', (30, 60), self.font, 1, (255, 0, 0), 2)

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist)

    def warp(self, img, dbgcode=0):
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        if dbgcode == 1:
            f, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 9))
            f.tight_layout()
            ax0.imshow(img)
            ax0.set_title("Before Warp")
            ax0.plot(self.src[0][0], self.src[0][1], '.', markersize=12)
            ax0.plot(self.src[1][0], self.src[1][1], '.', markersize=12)
            ax0.plot(self.src[2][0], self.src[2][1], '.', markersize=12)
            ax0.plot(self.src[3][0], self.src[3][1], '.', markersize=12)
            lleft = plt.Line2D((self.src[0, 0], self.src[1, 0]), (self.src[0, 1], self.src[1, 1]))
            ax0.add_line(lleft)
            lright = plt.Line2D((self.src[2, 0], self.src[3, 0]), (self.src[2, 1], self.src[3, 1]))
            ax0.add_line(lright)
            ax1.imshow(warped)
            ax1.set_title("After Warp")
            plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
            plt.show()
        return warped

    def unwarp(self, img):
        Minv = cv2.getPerspectiveTransform(self.dst, self.src)
        warped = cv2.warpPerspective(img, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        return warped

    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        elif orient == 'y':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        else:
            raise ValueError('wrong orientation')
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        sbinary = np.zeros_like(scaled_sobel)
        scaled_sobel[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return sbinary

    def mag_thresh(self, img, sobel_kernel=3, thresh=(0, 255)):
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        mag = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))
        scaled_mag = np.uint8(255 * mag / np.max(mag))
        binary_output = np.zeros_like(scaled_mag)
        binary_output[(scaled_mag >= thresh[0]) & (scaled_mag <= thresh[1])] = 1
        return binary_output

    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobelx = np.abs(sobelx)
        abs_sobely = np.abs(sobely)
        direction = np.arctan2(abs_sobely, abs_sobelx)
        binary_output = np.zeros_like(gray)
        binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
        return binary_output

    def canny(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)

    def color_mask(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        H = hls[:, :, 0]
        L = hls[:, :, 1]
        S = hls[:, :, 2]
        S = (S / np.max(S) * 255).astype(np.uint8)
        hls_binary = np.zeros_like(H)
        hls_binary[((H <= 24) & (H >= 18)) & (S >= 100)] = 1

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_binary = np.zeros_like(gray)
        gray_binary[gray >= 200] = 1

        color_binary = np.zeros_like(gray)
        color_binary[(gray_binary == 1) | (hls_binary == 1)] = 1
        return color_binary

    def thresholding(self, img):
        color_binary = self.color_mask(img)
        ksize = 7

        gradx = self.abs_sobel_thresh(color_binary, orient='x', sobel_kernel=ksize, thresh=(30, 255))
        grady = self.abs_sobel_thresh(color_binary, orient='y', sobel_kernel=ksize, thresh=(30, 255))
        color_gradx_grady = np.dstack((np.zeros_like(gradx), (255 * gradx / np.max(gradx)).astype(np.uint8), \
                                       (255 * grady / np.max(grady)).astype(np.uint8)))
        gradx_and_grady = np.zeros_like(gradx)
        gradx_and_grady[(gradx == 1) & (grady == 1)] = 1

        mag_binary = self.mag_thresh(color_binary, sobel_kernel=ksize, thresh=(30, 255))
        dir_binary = self.dir_threshold(color_binary, sobel_kernel=ksize, thresh=(0.7, 1.3))
        color_mag_dir = np.dstack((np.zeros_like(mag_binary), (255 * mag_binary / np.max(mag_binary)).astype(np.uint8), \
                                   (255 * dir_binary / np.max(dir_binary)).astype(np.uint8)))
        mad_and_dir = np.zeros_like(mag_binary)
        mad_and_dir[(mag_binary == 1) & (dir_binary == 1)] = 1

        combined_thresholding = np.zeros_like(dir_binary)
        combined_thresholding[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        return combined_thresholding, color_binary

    def sliding_find_cor(self, img, xstart, y, sum_th, coordinates):
        sums = []
        x_non_zero_start_flag = 0
        x_non_zero_start_point = 0
        for x in range(int(xstart - 3 * self.box_width), int(xstart + 3 * self.box_width),
                       self.box_step_horizontal):
            if x < 0:
                continue
            elif ((x + self.box_width) >= img.shape[1]):
                break
            else:
                if x_non_zero_start_flag == 0:
                    x_non_zero_start_point = x
                    x_non_zero_start_flag = 1
                sums.append(np.sum(img[(y - self.box_height):y, x:x + self.box_width]))
        max_sum = np.max(sums)
        y_res = y - 0.5 * self.box_height
        if (max_sum > sum_th):
            x_res = np.argmax(sums) + x_non_zero_start_point + self.box_width * 0.5
        else:
            return
            #if (len(coordinates) == 0):
            #    x_res = xstart
            #else:
            #    x_res = coordinates[-1][1]
        if x_res < 0:
            x_res = x_res
            pass
        coordinates.append((y_res, x_res))

    def sliding(self, img, sum_th):
        histogram = np.sum(img[int(img.shape[0] / 2):, :], axis=0)
        if np.max(histogram[:int(img.shape[1] / 2)]) > 5 and np.max(histogram[int(img.shape[1] / 2):]) > 5:
            left_start = np.argmax(histogram[:int(img.shape[1] / 2)])
            right_start = np.argmax(histogram[int(img.shape[1] / 2):]) + img.shape[1] / 2
            self.last_left_start = left_start
            self.last_right_start = right_start
        else:
            left_start = self.last_left_start
            right_start = self.last_right_start
        coorinates_left = []
        coorinates_right = []
        for y in range(img.shape[0], 0, -self.box_step_vertical):
            if (y - self.box_height) >= 0:
                self.sliding_find_cor(img, left_start, y, sum_th, coorinates_left)
                self.sliding_find_cor(img, right_start, y, sum_th, coorinates_right)
                if len(coorinates_left)>0:
                    left_start = coorinates_left[-1][1]
                if len(coorinates_right) > 0:
                    right_start = coorinates_right[-1][1]
        return coorinates_left, coorinates_right

    def draw_box(self, img, coordinate):
        topLeft = (int(coordinate[1] - self.box_width / 2), int(coordinate[0] - self.box_height / 2))
        bottomLeft = (int(coordinate[1] - self.box_width / 2), int(coordinate[0] + self.box_height / 2))
        topRight = (int(coordinate[1] + self.box_width / 2), int(coordinate[0] - self.box_height / 2))
        bottomRight = (int(coordinate[1] + self.box_width / 2), int(coordinate[0] + self.box_height / 2))
        C = (0, 255, 0)
        cv2.line(img, topLeft, bottomLeft, C, 2)
        cv2.line(img, bottomLeft, bottomRight, C, 2)
        cv2.line(img, bottomRight, topRight, C, 2)
        cv2.line(img, topLeft, topRight, C, 2)

    def draw_sliding_result(self, img, coordinates_left, coordinates_right):
        img = cv2.cvtColor(img * 255, cv2.COLOR_GRAY2RGB)
        for coodinate in coordinates_left:
            self.draw_box(img, coodinate)
        for coodinate in coordinates_right:
            self.draw_box(img, coodinate)
        return img

    def gen_one_box_in_box_image(self, box_image, coordinate):
        topLeft = (int(coordinate[1] - self.box_width / 2), int(coordinate[0] - self.box_height / 2))
        bottomLeft = (int(coordinate[1] - self.box_width / 2), int(coordinate[0] + self.box_height / 2))
        topRight = (int(coordinate[1] + self.box_width / 2), int(coordinate[0] - self.box_height / 2))
        bottomRight = (int(coordinate[1] + self.box_width / 2), int(coordinate[0] + self.box_height / 2))
        box = np.ones((self.box_height, self.box_width))
        box_image[topLeft[1]:bottomLeft[1], topLeft[0]:topRight[0]] = 1
        return

    def box_image_gen(self, box_img_shape, coodinates_left, coordinates_right):
        box_image_left = np.zeros(box_img_shape)
        box_image_right = np.zeros(box_img_shape)
        for coordinate in coodinates_left:
            self.gen_one_box_in_box_image(box_image_left, coordinate)
        for coordinate in coordinates_right:
            self.gen_one_box_in_box_image(box_image_right, coordinate)
        return box_image_left, box_image_right

    def poly_fit(self, filtered_by_box_image):
        yx = np.where(filtered_by_box_image == 1)
        y = yx[0]
        x = yx[1]
        try:
            fit = np.polyfit(y, x, 2)
        except TypeError:
            cv2.imwrite("debug_orig_img_poly_fit_filter_box.jpg", cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR))
            quit()
        y = np.linspace(0, filtered_by_box_image.shape[0]-1, num=100)
        fitx = fit[0] * y ** 2 + fit[1] * y + fit[2]
        return y, fitx
    
    def remove_out_of_bound_pts(self,warped_shape,y_left, fitx_left, y_right, fitx_right):
        assert len(y_left)==len(fitx_left)
        assert len(y_right) == len(fitx_right)
        y_left_bounded=[]
        fitx_left_bounded=[]
        y_right_bounded=[] 
        fitx_right_bounded=[]
        for i in range(len(y_left)):
            if fitx_left[i]>=0 and fitx_left[i]<warped_shape[1]:
                y_left_bounded.append(y_left[i])
                fitx_left_bounded.append(fitx_left[i])
        for i in range(len(y_right)):
            if fitx_right[i]>=0 and fitx_right[i]<warped_shape[1]:
                y_right_bounded.append(y_right[i])
                fitx_right_bounded.append(fitx_right[i])
        return y_left_bounded, fitx_left_bounded, y_right_bounded, fitx_right_bounded
                
    def poly_fit_unwarped(self,img_shape, y_left, x_left,y_right, x_right):
        try:
            fit_left = np.polyfit(y_left, x_left, 2)
            fit_right = np.polyfit(y_right, x_right, 2)
        except TypeError:
            cv2.imwrite("debug_orig_img_poly_fit_find_polyfit_edge.jpg", cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR))
            quit()
        fity_left = np.linspace(np.min(y_left),img_shape[0]-25, num=100)
        fitx_left = fit_left[0] * fity_left ** 2 + fit_left[1] * fity_left + fit_left[2]
        fity_right = np.linspace(np.min(y_right),img_shape[0]-25, num=100)
        fitx_right = fit_right[0] * fity_right ** 2 + fit_right[1] * fity_right + fit_right[2]
        return fity_left,fitx_left,fity_right,fitx_right

    def gen_lines_from_pts(self,img_shape,y, x):
        ret_img = np.zeros(img_shape)
        assert len(y) == len(x)
        pts = list(zip(y, x))
        for i in pts:
            ret_img[i] = 255
        return ret_img
    
    def project(self, warped, y_left, fitx_left, y_right, fitx_right, undist):
        # Remove out of bound points
        y_left_bounded, fitx_left_bounded, y_right_bounded, fitx_right_bounded = \
            self.remove_out_of_bound_pts(warped.shape,y_left, fitx_left, y_right, fitx_right)
        # Extend the lines to bottom of images
        left_line_only_img = self.gen_lines_from_pts(warped.shape, y_left_bounded, fitx_left_bounded)
        right_line_only_img = self.gen_lines_from_pts(warped.shape, y_right_bounded, fitx_right_bounded)
        left_line_only_img_unwarped = self.unwarp(left_line_only_img)
        right_line_only_img_unwarped = self.unwarp(right_line_only_img)
        yx = np.where(left_line_only_img_unwarped > 0)
        y_left_unwarped, x_left_unwarped = yx[0], yx[1]
        yx = np.where(right_line_only_img_unwarped > 0)
        y_right_unwarped, x_right_unwarped = yx[0], yx[1]
        fity_left_unwarped, fitx_left_unwarped, fity_right_unwarped, fitx_right_unwarped = \
            self.poly_fit_unwarped(warped.shape,y_left_unwarped, x_left_unwarped,y_right_unwarped, x_right_unwarped)
        # Apply poly fill
        poly_fill_gray = np.zeros_like(warped).astype(np.uint8)
        poly_fill_color = np.dstack((poly_fill_gray, poly_fill_gray, poly_fill_gray))
        pts_left = np.array([np.transpose(np.vstack([fitx_left_unwarped, fity_left_unwarped]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([fitx_right_unwarped, fity_right_unwarped])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(poly_fill_color, np.int_([pts]), (0, 255, 0))
        result = cv2.addWeighted(undist, 1, poly_fill_color, 0.3, 0)
        return result,poly_fill_color

    def curvrad(self, y_left, fitx_left, y_right, fitx_right):
        ym_per_pix = 3 / self.dashed_line_length_in_pixel
        xm_per_pix = 3.7 / self.lane_width_in_pixel
        y_eval = np.max(y_left) * ym_per_pix
        left_fit_cr = np.polyfit(y_left * ym_per_pix, fitx_left * xm_per_pix, 2)
        right_fit_cr = np.polyfit(y_right * ym_per_pix, fitx_right * xm_per_pix, 2)
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) \
                        / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) \
                         / np.absolute(2 * right_fit_cr[0])
        return left_curverad, right_curverad

    def estimate_center_offset(self, y_left, fitx_left, y_right, fitx_right, warped_shape):
        xm_per_pix = 3.7 / self.lane_width_in_pixel
        car_center_pix = int(warped_shape[1] / 2)
        lane_center_pix = abs(fitx_left[-1] - fitx_right[-1])
        offset_pix = car_center_pix - lane_center_pix
        offset_meter = (offset_pix + 268) * xm_per_pix
        return offset_meter

    def pipline(self, img):
        self.img = img
        undist = self.undistort(img)
        undist = cv2.resize(undist, (1280, 738))
        combined_thresholing, color_binary = self.thresholding(undist)
        warped = self.warp(color_binary)
        coorindates_left, coordinates_right = self.sliding(warped, 0.15*self.box_height*self.box_width)
        sliding_result = self.draw_sliding_result(warped, coorindates_left, coordinates_right)
        box_image_left, box_image_right = self.box_image_gen(warped.shape, coorindates_left, coordinates_right)
        filtered_by_box_image_left = np.multiply(warped, box_image_left)
        filtered_by_box_image_right = np.multiply(warped, box_image_right)
        y_left, fitx_left = self.poly_fit(filtered_by_box_image_left)
        y_right, fitx_right = self.poly_fit(filtered_by_box_image_right)
        projection, warped_polyfill = self.project(warped, y_left, fitx_left, y_right, fitx_right, undist)
        left_curvrad, right_curvrad = self.curvrad(y_left, fitx_left, y_right, fitx_right)
        offset_meter = self.estimate_center_offset(y_left, fitx_left, y_right, fitx_right, warped.shape)
        self.offsets_meter.append(offset_meter)
        self.left_curverads.append(left_curvrad)
        self.right_curverads.append(right_curvrad)
        self.resetDiag()
        self.mainDiagScreen = projection
        self.diag1 = self.warp(undist)
        self.diag2 = warped
        self.diag3 = color_binary
        self.diag4 = combined_thresholing
        self.diag5 = sliding_result
        self.diag6 = filtered_by_box_image_left
        self.diag7 = filtered_by_box_image_right
        self.diag8 = warped_polyfill
        self.frameNum = self.frameNum + 1
        self.diagScreenUpdate()
        return self.diagScreen

    def toVideo(self, input_video_file_name, output_video_file_name):
        clipInput = VideoFileClip(input_video_file_name)
        clipOutput = clipInput.fl_image(self.pipline)
        clipOutput.write_videofile(output_video_file_name, audio=False)
