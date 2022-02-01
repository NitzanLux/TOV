import cv2
from typing import List, Dict
import numpy as np
from os.path import join

CALIBRATION_FILE_FORMAT = "mat%d.npy"

CALIBRATION_MATRIX_FOLDER = "calibration_matrixes"
MIN_MATCH_COUNT_NUMBER_FOR_IMAGE_COMPERATION = 8
MIN_MATCH_COUNT_FOR_HOMOGRAPHY = 8
MAXIMAL_ITTERATION_TRIALS_FOR_CALIBRATIONS =300

CH_BO_W = 7
CH_BO_H = 9


class Camera:
    def __init__(self, camera_id: int,calibration_needed):
        camera = cv2.VideoCapture(camera_id)
        self.id=camera_id
        if not camera.isOpened():
            assert False, "camera is n/a"
        self.camera = camera
        frame = self.get_frame()
        if frame is None:
            assert False, "camera is n/a"
        self.h, self.w = frame.shape[0], frame.shape[1]
        # self.calibrate_camera(calibration_needed)

    def __call__(self):
        return self.camera

    def isOpened(self):
        return self.camera.isOpened()

    def release(self):
        self.camera.release()

    def get_frame(self):
        if self.camera is not None:
            ret, frame = self.camera.read()
        else:
            ret, frame = self.camera.read()
        return frame if ret else None

    def get_imeges(self):
        images = []
        while (True):
            f = self.get_frame()
            if f is None:
                break
            cv2.imshow('frame', f)
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (CH_BO_W, CH_BO_H), None)
            if ret:
                images.append(f)
                print(len(images))
            if len(images)>15:
                print("next")
                break
            cv2.waitKey(4)
        cv2.destroyAllWindows()
        return images

    def calibrate_camera(self,is_needed):
        path = join(CALIBRATION_MATRIX_FOLDER, CALIBRATION_FILE_FORMAT % self.id)
        if not is_needed:
            try:
                self.calibration_matrix = np.load(path)
                return
            except FileNotFoundError:
                pass
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((CH_BO_W * CH_BO_H, 3), np.float32)
        objp[:, :2] = np.mgrid[0:CH_BO_W, 0:CH_BO_H].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = self.get_imeges()
        gray = None
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (CH_BO_W, CH_BO_H), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (CH_BO_W, CH_BO_H), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(50)
        cv2.destroyAllWindows()
        _,self.calibration_matrix ,_,_,_= cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        np.save(path,self.calibration_matrix)
    def sift(self, dest_cam,match_count_number, _trial_number=0):
        if _trial_number > MAXIMAL_ITTERATION_TRIALS_FOR_CALIBRATIONS:
            return None, None
        img1 = self.get_frame()
        img2 = dest_cam.get_frame()
        new_imges = np.hstack((img1,img2))
        cv2.imshow('new_imges',new_imges)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        # Initiate SIFT detector
        sift = cv2.xfeatures2d_SIFT()
        sift = sift.create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        if len(good) >= match_count_number:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            return src_pts, dst_pts
        else:
            print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT_FOR_HOMOGRAPHY))
            matchesMask = None
            return self.sift(dest_cam,match_count_number, _trial_number + 1)  # try another time

    def __gt__(self, other):
        """
        greater means ferther then me.
        :param other:
        :return:
        """
        return self.sift_scale(other)

    def sift_scale(self, other):
        M = self.sift_and_affine(other)
        return np.linalg.norm(M[:,0])> 1
        # src_pts, dest_pts = self.sift(other, MIN_MATCH_COUNT_NUMBER_FOR_IMAGE_COMPERATION)
        # src_pts, dest_pts = np.array(src_pts), np.array(dest_pts)
        # if src_pts is None:
        #     return True
        # src_pts_std = src_pts.std(axis=0)
        # dest_pts_std = dest_pts.std(axis=0)
        # return np.linalg.norm(src_pts_std) <np.linalg.norm(dest_pts_std)

    def sift_and_homography(self, other):
        dest_pts,src_pts =None,None
        while(src_pts is None):
            src_pts, dest_pts = self.sift(other, MIN_MATCH_COUNT_FOR_HOMOGRAPHY)
        return Camera.get_homography(src_pts, dest_pts)

    def sift_and_affine(self, other):
        dest_pts,src_pts =None,None
        while(src_pts is None):
            src_pts, dest_pts = self.sift(other, MIN_MATCH_COUNT_FOR_HOMOGRAPHY)
        return Camera.get_affine(src_pts, dest_pts)
    def __and__(self, other):
        src_pts, dest_pts = self.sift(other, MIN_MATCH_COUNT_NUMBER_FOR_IMAGE_COMPERATION)
        return src_pts is not None


    @staticmethod
    def get_affine(src_pts, dst_pts):
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts)
        matchesMask = mask.ravel().tolist()
        return M

    @staticmethod
    def get_homography(src_pts, dst_pts):
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        return M