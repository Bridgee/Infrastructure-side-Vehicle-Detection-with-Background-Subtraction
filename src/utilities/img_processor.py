import numpy as np
import cv2

class ImgProcessor():
    def __init__(self):
        self.w = 1280
        self.h = 960
        self.f = 331.04228163
        self.K = np.eye(3)

        self.K[0, 0] = self.f
        self.K[1, 1] = self.f
        self.K[0, 2] = self.w / 2
        self.K[1, 2] = self.h / 2
        self.D = np.array([0., 0., 0., 0.])

        self.Knew = self.K.copy()
        self.Knew[(0,1), (0,1)] = 0.28 * self.Knew[(0,1), (0,1)]

    def get_raw(self, frame):
        return frame

    def get_undist(self, frame):
        return cv2.fisheye.undistortImage(frame, self.K, D = self.D, Knew = self.Knew)

    def BGR2RGB(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def RGB2BGR(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def img_loader(self, name):
        return cv2.imread(name)

    def img_resizer(self, frame, scale_x = 0.7, scale_y = 0.7):
        return cv2.resize(frame, dsize = None, fx = scale_x, fy = scale_y)

    def img_saver(self, frame, path, time):
        cv2.imwrite(path + time + '.png', frame)