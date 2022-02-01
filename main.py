import numpy as np
import cv2
from typing import List
from process_frame import *
from time import time
from threading import Thread
import sys
import cv2
from queue import Queue


def camera_loop(delay: int = 4, save_video=False, calibration_needed=False):
    current_key_pointer = [-1]
    frame_processor = Process_frame(current_key_pointer, calibration_needed)
    frame_width, frame_height = frame_processor.get_frame_height_and_width()
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi', fourcc, 1, (frame_width, frame_height)) if save_video else None
    kount = 0
    fps = 0
    buffer = []
    factor = 1
    start = time()

    while (frame_processor.is_open()):
        frame = frame_processor.get_frame()
        if frame is not None:
            kount += 1
            buffer.append(frame) if save_video else None
            cv2.imshow('frame', frame)
            if kount >= delay:
                if kount == delay:
                    end = time()
                    fps = kount / (end - start)
                    out = cv2.VideoWriter('output.mkv', fourcc, fps, (frame_width, frame_height))
                out.write(buffer[kount - delay]) if save_video else None
            current_key_pointer[0] = cv2.waitKey(1)
            if current_key_pointer[0] & 0xFF == ord('q'):
                [out.write(buffer[i]) for i in range(kount - delay, kount)] if save_video else None
                break
        else:
            break
    print("Frames Per Second = ", fps)
    frame_processor.release()
    out.release() if save_video else None
    cv2.destroyAllWindows()

if __name__ == '__main__':
    camera_loop(4)
