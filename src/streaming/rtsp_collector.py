"""
This is the tkinter GUI for the real-time detection algorithm
"""

__author__ = 'Zhouqiao Zhao'

import cv2

class RtspCollector():
    def __init__(self, address):
        print(address)
        self.address = address
        self.vcap = cv2.VideoCapture(address)
        if not self.vcap.isOpened():
            raise ValueError("Unable to open video source", self.address)
        self.width = self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def __del__(self):
        if self.vcap.isOpened():
            self.vcap.release()

    def cv2_streaming(self):
        while True:
            ret, frame = self.vcap.read()
            cv2.imshow('RTSP Stream', frame)
            k = cv2.waitKey(30)

            if k == 27:
                break
            elif k == 32:
                while cv2.waitKey(0) != 32:
                    cv2.waitKey(0)

    def get_frame(self):
        if self.vcap.isOpened():
            ret, frame = self.vcap.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

if __name__ == '__main__':
    address = 'rtsp://169.235.68.132:9000/1'
    cur_rtsp_collector = RtspCollector(address)
    print('Esc to quit, Space to pause/play')
    cur_rtsp_collector.cv2_streaming()
    cv2.destroyAllWindows()