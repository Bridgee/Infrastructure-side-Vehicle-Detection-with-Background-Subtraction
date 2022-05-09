from __future__ import print_function
# from pybgs import bgs
import os.path
import numpy as np
np.set_printoptions(suppress=True)
import json
from socket import *
import csv
import cv2
print(cv2.__version__)
from cv2 import imread, absdiff, cvtColor, threshold, morphologyEx, dilate, bitwise_not,bitwise_and, add, pow,addWeighted, imwrite,convertScaleAbs,findContours,contourArea,fillConvexPoly,drawContours,bitwise_or
import matplotlib.pyplot as plt
import sys
import time
import random
import os
import queue
import threading
from multiprocessing import Process, Pool, shared_memory
from multiprocessing.managers import SharedMemoryManager
from math import radians, degrees, cos, sin, asin, sqrt, tan, atan2
# sys.path.append('C:\\bgslibrary\\build_win64')
import pybgs as bgs

path = './temp2'
files = sorted(os.listdir(path), key=len)
files.sort(key = lambda x: int(x[:-4]))
files_num = len(files)

random_files = files.copy()
random.shuffle(random_files)

bg = cv2.imread('bg_arco_new.png')
bg_t = cv2.imread('bg_test.png')
mask = cv2.imread('mask.png', flags = cv2.IMREAD_GRAYSCALE)
mask_flow = cv2.imread('mask_flow.png', flags = cv2.IMREAD_GRAYSCALE)
mask_t = cv2.erode(mask, kernel5, iterations = 1)
mask_center = cv2.imread('mask_center.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_center = threshold(mask_center, 10, 255, cv2.THRESH_BINARY)
# left down, from center to edge
mask_ldf = cv2.imread('mask_ldf.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_ldf = threshold(mask_ldf, 10, 255, cv2.THRESH_BINARY)
# left down, towards center
mask_ldt = cv2.imread('mask_ldt.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_ldt = threshold(mask_ldt, 10, 255, cv2.THRESH_BINARY)
# left down, towards center, left trun
mask_ldt_l = cv2.imread('mask_ldt_left.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_ldt_l = threshold(mask_ldt_l, 10, 255, cv2.THRESH_BINARY)

# left upper, from center to edge
mask_luf = cv2.imread('mask_luf.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_luf = threshold(mask_luf, 10, 255, cv2.THRESH_BINARY)
# left upper, towards center
mask_lut = cv2.imread('mask_lut.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_lut = threshold(mask_lut, 10, 255, cv2.THRESH_BINARY)
# left upper, towards center, left trun
mask_lut_l = cv2.imread('mask_lut_left.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_lut_l = threshold(mask_lut_l, 10, 255, cv2.THRESH_BINARY)

# right down, from center to edge
mask_rdf = cv2.imread('mask_rdf.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_rdf = threshold(mask_rdf, 10, 255, cv2.THRESH_BINARY)
# right down, towards center
mask_rdt = cv2.imread('mask_rdt.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_rdt = threshold(mask_rdt, 10, 255, cv2.THRESH_BINARY)
# right down, towards center, left trun
mask_rdt_l = cv2.imread('mask_rdt_left.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_rdt_l = threshold(mask_rdt_l, 10, 255, cv2.THRESH_BINARY)

# right upper, from center to edge
mask_ruf = cv2.imread('mask_ruf.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_ruf = threshold(mask_ruf, 10, 255, cv2.THRESH_BINARY)
# right upper, towards center
mask_rut = cv2.imread('mask_rut.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_rut = threshold(mask_rut, 10, 255, cv2.THRESH_BINARY)
# right upper, towards center, left trun
mask_rut_l = cv2.imread('mask_rut_left.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_rut_l = threshold(mask_rut_l, 10, 255, cv2.THRESH_BINARY)

mask_tuple = (mask_center,mask_ldf,mask_ldt,mask_ldt_l,
              mask_luf,mask_lut,mask_lut_l,
              mask_rdf,mask_rdt,mask_rdt_l,
              mask_ruf,mask_rut,mask_rut_l)
mask_name = ['mask_center.png',
             'mask_ldf.png', 'mask_ldt.png', 'mask_ldt_left.png',
             'mask_luf.png', 'mask_lut.png', 'mask_lut_left.png',
             'mask_rdf.png', 'mask_rdt.png', 'mask_rdt_left.png',
             'mask_ruf.png', 'mask_rut.png', 'mask_rut_left.png']

'''
OpticalFlow class
kernal function (OpticalFlow.kernal())
OpticalFlow.flow_contour: optical flow contour
OpticalFlow.mag : optical flow amplitude
'''
class OpticalFlow():
    def __init__(self,img):
        # self.init_flag = True
        self.mask_flow = cv2.imread('mask_flow.png', flags = cv2.IMREAD_GRAYSCALE)
        img_prev_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img_prev_gray = bitwise_and(img_prev_gray,img_prev_gray,mask = mask_flow)
    def kernal(self,img):
        # input: video_frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = bitwise_and(img_gray,img_gray,mask = self.mask_flow)

        flow = cv2.calcOpticalFlowFarneback(self.img_prev_gray, img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.img_prev_gray = img_gray.copy()
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mag[np.isinf(mag)] = 0 # sometimes flow elements of 0 is reagarded as a number close to 0 but not equal to 0 by Python, then the cartToPolar function will get an inf value.
        mag[np.isnan(mag)] = 0
        self.mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #  mag shape 960 1280
        _, flow_contour = threshold(self.mag, 40, 255, cv2.THRESH_BINARY)
        flow_contour_raw = flow_contour.copy()
        contours, _ = findContours(flow_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            if contourArea(contour)<100:
                fillConvexPoly(flow_contour, contour, (0,0,0))
        flow_contour = cv2.morphologyEx(flow_contour, cv2.MORPH_OPEN, kernel, iterations=1)
        flow_contour = cv2.dilate(flow_contour, kernel3, iterations=1)
        contours, _ = findContours(flow_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            fillConvexPoly(flow_contour, contour, (255,255,255))
    #         if contourArea(contour)>100:
    #             fillConvexPoly(flow_contour, contour, (255,255,255))
    #             drawContours(flow_contour, contour, -1, (255,255,255), cv2.FILLED)
    #         else:
    #             drawContours(flow_contour, contour, -1, (0,0,0), cv2.FILLED)
        contours, _ = findContours(flow_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if contourArea(contour)<100:
                fillConvexPoly(flow_contour, contour, (0,0,0))
        contours, _ = findContours(flow_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if contourArea(contour)<100:
                fillConvexPoly(flow_contour, contour, (0,0,0))
        self.contours, _ = findContours(flow_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.flow_contour = flow_contour.copy()
        # print(np.shape(self.flow_contour))
'''
Renew class
bgs library renew (Renew.bgs_renew())
SIFT comparing for choosing best bakcground (Renew.compare_sfit())
kernal function includes these two above functions
Renew.bgs_output: background generated from bgs bgslibrary
Renew.output: background after choosing the most similar background
'''
class Renew():
    def __init__(self, mask_name):
        super(renew, self).__init__()
        self.name = mask_name[:-4]
        img_mask = cv2.imread(mask_name, flags = cv2.IMREAD_GRAYSCALE)
        _, self.mask = threshold(img_mask, 10, 255, cv2.THRESH_BINARY)
        self.flow_region_mean = 0
        self.flow_region_mean_prev = 0
        self.sift = cv2.SIFT_create()
        self.bgs = bgs.MixtureOfGaussianV2()

        FLANN_INDEX_KDTREE=0
        indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        searchParams = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams, searchParams)
        bg = cv2.imread('bg_arco_new.png')
        bg = bitwise_and(bg,bg, mask = self.mask) # template background
        _, self.des1 = self.sift.detectAndCompute(bg, None) # sift descriptor of the template background
        self.score = 0
        self.best_score = 0
        self.best_index = 0
        self.output = np.zeros((960,960))
        self.init_flag = True

    def subapply(self, input_frame):
        update_region = bitwise_and(input_frame,input_frame,mask = self.mask)
        _ = self.apply(update_region)

    def getbg(self):
        output = self.getBackgroundModel()
        return output

    def kernal(self, input_frame, flow_contour, mag,index_frame):
        self.bgs_renew(input_frame, flow_contour, mag)
        self.compare_sfit(index_frame)


    def bgs_renew(self, input_frame, flow_contour, mag):
        # input : video frame, optical flow contour, optical flow amplitude

        flow_region = bitwise_and(flow_contour,flow_contour,mask = self.mask)
        flow_region_contours, _ = findContours(flow_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.flow_region_mean = cv2.mean(cv2.bitwise_and(mag,mag,mask = flow_region),  mask=flow_region)[0]
        # flow_region_mean.append(flow_mean)
        area = 0
        for contour in flow_region_contours:
            area += contourArea(contour)
        # print(idx_frame ,',',i,',', area,',', flow_region_mean[i])
        if self.init_flag:
            update_region = bitwise_and(input_frame,input_frame,mask = self.mask)
            _ = self.bgs.apply(update_region)
            self.init_flag = False
        if area>3000 and self.flow_region_mean>self.flow_region_mean_prev and self.flow_region_mean>50 :
            # print('update', idx_frame ,',',i,',', area,',', flow_region_mean[i])
            update_region = bitwise_and(input_frame,input_frame,mask = self.mask)
            _ = self.apply(update_region)
        self.bgs_output = self.bgs.getBackgroundModel()

    def compare_sfit(self, index_frame):
        # input : frame index
        _, des2 = self.sift.detectAndCompute(self.bgs_output, None)
        matches = self.flann.knnMatch(self.des1, des2, k=2)
        matchNum = [m for (m, n) in matches if m.distance <0.8* n.distance]
        self.score = len(matchNum) / len(matches)
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_index = index_frame
            self.output = self.bgs_output.copy() # generated background

if __name__ == "__main__":
    idx_frame = 0
    img = cv2.imread(os.path.join(path,files[idx_frame])) # read the first image

    renew_instance = Renew('mask_rut.png') # renew module of bgslibrary and sift
    OF_instance = OpticalFlow(img) #

    start_time = time.time()


    while 1:
        idx_frame += 1
        # if idx_frame%2==0: continue
        if idx_frame>=100: break # files_num-1

        # output_region = ()
        img = cv2.imread(os.path.join(path,files[idx_frame]))

        OF_instance.kernal(img)

        renew_instance.kernal(img, OF_instance.flow_contour, OF_instance.mag, idx_frame)

        print('################ Frame:', idx_frame, '################',files[idx_frame])
        # print(renew_instance.best_score)
        # print(renew_instance.best_index)
