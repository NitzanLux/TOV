import cv2
import numpy as np
from typing import List
from camera_menger import Camera_meneger

MAXIMAL_ITTERATION_TRIALS_FOR_CALIBRATIONS = 100
MAX_SCALE_PRECENTAGE = 100
FOCUS_FACTOR = 20
ZOOM_SCALE_FACTOR = 10

KEY = 1
MIN_MATCH_COUNT = 5


class Process_frame:
    def __init__(self, current_key_pointer: List[int], calibration_needed):
        self.current_key_pointer = current_key_pointer
        self.camera_menger: Camera_meneger = Camera_meneger(calibration_needed)
        if len(self.camera_menger) == 0:
            assert False, "No cameras has been found"
        self.scale = 100
        self.focus = 0
        self.key = 400
        higtes = [c.h for c in self.camera_menger]
        widthes = [c.w for c in self.camera_menger]
        self.w, self.h = max(widthes), max(higtes)
        self.maximal_scale_for_moving = []
        self.affin_factor = np.array([0, 0, 0])
        self.center_smallest()  # needed to be befor find center points
        self.center_points = []
        self.find_center_points()
        self.find_maximal_zoom()
        # self.load_NN()
    # def load_NN(self):
    #     pass
    def release(self):
        self.camera_menger.release_all()

    def is_open(self):
        return self.camera_menger.is_open()

    def get_frame(self):
        frame = self.camera_menger.get_frame()
        return self.process_frame(frame) if frame is not None else None

    def process_frame(self, frame):
        camera_index = self.camera_menger.current_idx
        frame = self.transform(frame)
        current_key = self.current_key_pointer[0]
        if current_key == ord('+'):
            print(self.scale)
            if not camera_index == len(self.camera_menger) - 1 and self.scale > self.maximal_scale_for_moving[
                camera_index]:
                self.camera_menger += 1
            self.scale += ZOOM_SCALE_FACTOR
            # else:
            print("+ : %f" % self.scale)
        elif current_key == ord('-'):
            if camera_index > 0 and self.maximal_scale_for_moving[camera_index - 1] > self.scale - ZOOM_SCALE_FACTOR:
                self.camera_menger -= 1
            self.scale -= ZOOM_SCALE_FACTOR
            # else:
            print("- : %f" % self.scale)
        elif current_key == ord("y"):
            self.camera_menger += 1
            print(self.camera_menger.get_index())
            print("a")
        return frame

    def center_smallest(self):
        w, h = self.camera_menger[-1].w, self.camera_menger[-1].h
        M = self.camera_menger.transformation[-1]
        temp_vec = np.array([w // 2, h // 2, 1])
        out_vec = M @ temp_vec
        diff = np.array([self.w // 2, self.h // 2, 1]) - out_vec
        self.affin_factor = diff

    def find_center_points(self):
        self.center_points = []
        for i in range(len(self.camera_menger)):
            new_M = self.process_transformation(i)
            center_vec = np.linalg.inv(new_M) @ (np.array([self.w // 2, self.h // 2, 1]))
            self.center_points.append(center_vec)

    def find_maximal_zoom(self):
        self.maximal_scale_for_moving = []
        i = 0
        s = self.scale
        while (i < len(self.camera_menger)-1):
            while(True):
                w = self.camera_menger[i + 1].w
                M = self.process_transformation(i + 1, s)
                l_vec = np.array([0, 0, 1])
                r_vec = np.array([w, 0, 1])
                lp = M @ l_vec
                rp = M @ r_vec
                if abs(lp[0] - rp[0]) > self.w:
                    s-=ZOOM_SCALE_FACTOR
                    self.maximal_scale_for_moving.append(s)
                    print("   %d   ----->>>>    %d"%(i,s))
                    break
                s += ZOOM_SCALE_FACTOR
            i += 1
        self.maximal_scale_for_moving.append(-1)

    def scale_transformation(self, M, i, s=None):
        s = self.scale if s is None else s
        if s == 100:
            return M
        new_M = M * np.array([s / 100, s / 100, 1])
        diff = np.array([self.w // 2, self.h // 2, 1]) - new_M @ self.center_points[i]
        new_M[:, 2] += diff
        return new_M

    def shift_smallest_to_center(self, M):
        new_M = np.copy(M)
        new_M[:, 2] += self.affin_factor
        return new_M

    def process_transformation(self, i, s=None):
        M = self.camera_menger.transformation[i]
        new_M = self.shift_smallest_to_center(M)
        new_M = self.scale_transformation(new_M, i, s)
        return new_M

    def transform(self, img):
        i = self.camera_menger.get_index()
        new_M = self.process_transformation(i)
        new_M = new_M[:2, :]
        img = cv2.warpAffine(img, new_M, (self.w, self.h))
        return img

    # def auto_focus(self): todo figure out
    #     focus = 0  # min: 0, max: 255, increment:5
    #     self.cam_pointer[self.camera_index].set(cv2.CAP_PROP_AUTOFOCUS, True)


    def get_frame_height_and_width(self):
        return self.w, self.h
