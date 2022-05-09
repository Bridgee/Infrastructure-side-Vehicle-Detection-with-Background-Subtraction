from __future__ import print_function
# from pybgs import bgs
import os.path
import numpy as np
np.set_printoptions(suppress=True)
import json
from socket import *
import csv
import cv2
# print(cv2.__version__)
from cv2 import imread, absdiff, cvtColor, threshold, morphologyEx, dilate, bitwise_not,bitwise_and, add, pow,addWeighted, imwrite,convertScaleAbs,findContours,contourArea,fillConvexPoly,drawContours,bitwise_or
import matplotlib.pyplot as plt
import sys
import time
import random
import os
import queue
import threading
from multiprocessing import Process, Pool, shared_memory, Semaphore, Event
from multiprocessing.managers import SharedMemoryManager
from math import radians, degrees, cos, sin, asin, sqrt, tan, atan2
from config import *
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

img_sm = shared_memory.SharedMemory(create = True, size=bg.nbytes)
img_buffer = np.ndarray(bg.shape, dtype=bg.dtype, buffer=img_sm.buf)

region_sm = shared_memory.SharedMemory(create = True, size=bg.nbytes)
region_buffer = np.ndarray(bg.shape, dtype=bg.dtype, buffer=region_sm.buf)

contour_sm = shared_memory.SharedMemory(create = True, size=mask.nbytes)
contour_buffer = np.ndarray(mask.shape, dtype=np.uint8, buffer=contour_sm.buf)

mag_sm = shared_memory.SharedMemory(create = True, size=mask.nbytes)
mag_buffer = np.ndarray(mask.shape, dtype=np.uint8, buffer=mag_sm.buf)



mask_tuple = (mask_center,mask_ldf,mask_ldt,mask_ldt_l,
              mask_luf,mask_lut,mask_lut_l,
              mask_rdf,mask_rdt,mask_rdt_l,
              mask_ruf,mask_rut,mask_rut_l)
mask_name = ['mask_center.png',
             'mask_ldf.png', 'mask_ldt.png', 'mask_ldt_left.png',
             'mask_luf.png', 'mask_lut.png', 'mask_lut_left.png',
             'mask_rdf.png', 'mask_rdt.png', 'mask_rdt_left.png',
             'mask_ruf.png', 'mask_rut.png', 'mask_rut_left.png']

crop_position = [[370,100],[480,0],[480,0],[480,0],
                [0,0],[100,0],[100,0],
                [480,300],[480,400],[480,400],
                [50,380],[0,300],[0,300]]
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
        self.flow_contour = np.zeros((960,960))
        self.mag = np.zeros((960,960))

    def kernal(self,img):
        # input: video_frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = bitwise_and(img_gray,img_gray,mask = self.mask_flow)

        flow = cv2.calcOpticalFlowFarneback(self.img_prev_gray, img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.img_prev_gray = img_gray.copy()
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        # sometimes flow elements of 0 is reagarded as a number close to 0
        # but not equal to 0 by Python, then the cartToPolar function will get an inf value.
        mag[np.isinf(mag)] = 0
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
        # super(Renew, self).__init__()
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
        if area>3000 and \
        self.flow_region_mean > self.flow_region_mean_prev \
        and self.flow_region_mean>50 :
            # print('update', idx_frame ,',',i,',', area,',', flow_region_mean[i])
            update_region = bitwise_and(input_frame,input_frame,mask = self.mask)
            _ = self.bgs.apply(update_region)
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

def run_kernal(instance,*args): #
    start_time = time.time()
    print('ok')

    instance.kernal(*args)
    print(instance.time_usage)
    serial_time = time.time()
    print('time usage',serial_time-start_time)

def run_renew(mask_name, crop_p ,img_sm, region_buffer, contour_sm, mag_sm, flow_e, read_e, renew_e, shutdown_e):
    print('Process (%s) start' % os.getpid(), mask_name)
    img_mask = cv2.imread(mask_name, flags = cv2.IMREAD_GRAYSCALE)
    _, region_mask = threshold(img_mask, 10, 255, cv2.THRESH_BINARY)
    bgs_renew = bgs.MixtureOfGaussianV2()
    flow_region_mean = 0
    flow_region_mean_prev = 0

    sift = cv2.SIFT_create()
    FLANN_INDEX_KDTREE=0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    bg = cv2.imread('bg_arco_new.png')
    bg = bitwise_and(bg,bg, mask = region_mask)
    bg = bg[crop_p[0] : crop_p[0]+480, crop_p[1]:crop_p[1]+480]
    # print('bg',np.shape(bg), bg.dtype)
    _, des1 = sift.detectAndCompute(bg, None)
    score = 0
    best_score = 0
    best_index = 0
    output = np.zeros((960,960), dtype=np.uint8)
    init_flag = True


    #####################################################################################
    img = np.ndarray((960,960,3), dtype=np.uint8, buffer=img_sm.buf)
    flow_contour = np.ndarray((960,960), dtype=np.uint8, buffer=contour_sm.buf)
    mag = np.ndarray((960,960), dtype=np.uint8, buffer=mag_sm.buf)


    #####################################################################################
    contours, _ = findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area = 0
    for contour in contours:
        area += contourArea(contour)
    if area/4>3000:
        area_threshold = 3000
    else:
        area_threshold = area/4

    #####################################################################################
    renew_e.set()
    # time.sleep(0.1)
    count = 0
    flow_e.wait()



    while 1:
        # print('0')
        child_start_time = time.time()
        if shutdown_e.is_set():
            break


        # print("1")
        # renew_e.clear()
        flow_e.wait() # wait for current frame, optical flow contour and amplitude

        wait_time = time.time() - child_start_time
        renew_e.clear()
        read_e.clear() # start renew and lock parent process

        ##############################################################################

        # print("2")
        img_cp = img.copy()
        flow_contour_cp = flow_contour.copy()
        mag_cp= mag.copy()

        

        read_e.set() # resume parent process

        ##############################################################################

        # print('3')
        flow_region = bitwise_and(flow_contour_cp, flow_contour_cp, mask = region_mask)
        flow_region_contours, _ = findContours(flow_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        flow_region_mean = cv2.mean(cv2.bitwise_and(mag_cp,mag_cp,mask = flow_region),  mask=flow_region)[0]
        area = 0
        for contour in flow_region_contours:
            area += contourArea(contour)
        flow_time = time.time() - child_start_time - wait_time

        # print("3")
        if init_flag:
            update_region = bitwise_and(img_cp,img_cp, mask= region_mask)
            update_region = update_region[crop_p[0]:crop_p[0]+480, crop_p[1]:crop_p[1]+480]
             #print('initialization')
            _ = bgs_renew.apply(update_region)
            init_flag = False
            continue

        # print("4")
        if area>area_threshold and flow_region_mean>flow_region_mean_prev and flow_region_mean>50 :
            # print('update', idx_frame ,',',i,',', area,',', flow_region_mean[i])
            update_region = bitwise_and(img_cp,img_cp,mask = region_mask)
            update_region = update_region[crop_p[0]:crop_p[0]+480, crop_p[1]:crop_p[1]+480]
            _ = bgs_renew.apply(update_region)
        renew_time = time.time() - child_start_time - wait_time - flow_time

        bgs_output = bgs_renew.getBackgroundModel()
        # cv2.imwrite('./gmm_mt_bgs/'+mask_name[5:-4]+'_'+str(count)+'_'+str(round(area,1))+'_'+str(round(flow_region_mean,3))+'.png', bgs_output)
        
        # print("5")
        bgs_time = time.time() - child_start_time - wait_time - flow_time - renew_time
        # print('bgs output',np.shape(bgs_output), bgs_output.dtype)
        _, des2 = sift.detectAndCompute(bgs_output, None)
        sift_time = time.time() - child_start_time - wait_time - flow_time - renew_time - bgs_time
        # print(np.shape(des1), des1.dtype)
        # print(np.shape(des2), des2.dtype)
        
        # print("6")
        matches = flann.knnMatch(des1, des2, k=2)
        matchNum = [m for (m, n) in matches if m.distance <0.8* n.distance]
        score = len(matchNum) / len(matches)
        
        # print("7")
        if score > best_score:
            best_score = score
            # best_index = index_frame
            output = bgs_output.copy()
            # region_buffer[:] = output[:]
            # child_end_time = time.time()
            # cv2.imwrite('./gmm_mt/'+mask_name[5:-4]+'_'+str(count)+'.png', output)
        # print("8")
        total_time = time.time() - child_start_time
        count += 1

        # flow_e.clear()
        if count%50 == 0:
            print( mask_name[:-4],'count', count ,'wait_time',wait_time,'flow_time',flow_time,'renew_time',
                renew_time,'bgs_time',bgs_time,'sift_time',sift_time,'total_time',total_time)
        
        ##############################################################################
        # print('4')
        # print("9")
        renew_e.set()
        flow_e.clear()
        # print(mask_name,'count',count,'finish',renew_e.is_set())

    # plt.imshow(cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB))
    # plt.show()

if __name__ == "__main__":
    idx_frame = 0
    img = cv2.imread(os.path.join(path,files[idx_frame])) # read the first image

    #renew_instance = Renew('mask_rut.png') # renew module of bgslibrary and sift
    OF_instance = OpticalFlow(img) #

    # flow_event = Event()
    flow_event_list = [0] *13
    for i in range(13):
        flow_event_list[i] = Event()

    renew_event_list = [0] *13
    for i in range(13):
        renew_event_list[i] = Event()

    readimg_list = [0] *13
    for i in range(13):
        readimg_list[i] = Event()

    shutdown_event = Event()

    # p = Process(target= run_renew ,args=(mask_name[0], img_sm,region_buffer, contour_sm, mag_sm,flow_event, renew_event, shutdown_event))
    # p.start()

    # p = Pool(13)
    # for i in range(13):
    #     p.apply_async(run_renew ,args=(mask_name[i], img_sm, region_buffer, contour_sm, mag_sm,flow_event, renew_event_list[i], shutdown_event,))
    # p.close()

    for i in range(13):
        p = Process(target = run_renew, 
                    args=(mask_name[i], crop_position[i], img_sm, region_buffer, contour_sm, mag_sm,
                          flow_event_list[i],readimg_list[i], renew_event_list[i], shutdown_event,)
                   )
        p.start()

    print('finish initialization')
    # for i in range(13):
    #     flow_event_list[i].set()
    start_time = time.time()
    while 1:
        # for i in range(13):
        #     flow_event_list[i].clear()# suspend all child processes
        idx_frame += 1
        if idx_frame%50==0:
            print('################ Frame:', idx_frame, '################',files[idx_frame])

        if idx_frame>=400:
            shutdown_event.set()
            for i in range(13):
                flow_event_list[i].set()
            # kill all child processes
            break
        
        # Main Job -- optical flow
        img = cv2.imread(os.path.join(path,files[idx_frame]))
        img_buffer[:] = img[:]
        OF_instance.kernal(img)
        contour_buffer[:] = OF_instance.flow_contour[:]
        cv2.imwrite('./gmm_mt/'+str(idx_frame)+'OF.png', OF_instance.flow_contour)
        mag_buffer[:] = OF_instance.mag[:]

        # print('dump all imgs to shared memory')
        for i in range(13):
            flow_event_list[i].set()
        # print('enable child process to read')

        for i in range(13):
            readimg_list[i].wait() # wait for child process to read
         #print('finish reading')
        for i in range(13):
            renew_event_list[i].wait() # wait for child process renew
        # print('wait renew')
    end_time = time.time()
    print('total time:',end_time - start_time)

    img_sm.close()
    region_sm.close()
    contour_sm.close()
    mag_sm.close()

    img_sm.unlink()
    region_sm.unlink()
    contour_sm.unlink()
    mag_sm.unlink()
