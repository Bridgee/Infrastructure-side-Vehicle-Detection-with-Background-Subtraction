"""
This is the tkinter GUI for the real-time detection algorithm.
04_10_2022
"""

__author__ = ['Zhouqiao Zhao', 'Jiahe Cao']

import sys, os
sys.path.append(os.getcwd() + '/src')
import tkinter as tk
import PIL.Image, PIL.ImageTk
from tkinter import filedialog as fd
import time
from datetime import datetime
import threading
from multiprocessing import Process, shared_memory, Event, Manager
import numpy as np
import cv2
import pybgs as bgs

from utilities import img_processor
from streaming import rtsp_collector

kernel = np.ones((1, 1), np.uint8)
kernel2 = np.ones((2, 2), np.uint8)
kernel3 = np.ones((3, 3), np.uint8)
kernel4 = np.ones((4, 4), np.uint8)
kernel5 = np.ones((5, 5), np.uint8)
kernel6 = np.ones((6, 6), np.uint8)
kernel7 = np.ones((7, 7), np.uint8)
kernel8 = np.ones((8, 8), np.uint8)
kernel9 = np.ones((9, 9), np.uint8)

# Center
mask_center = cv2.imread('./data/masks/mask_center.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_center = cv2.threshold(mask_center, 10, 255, cv2.THRESH_BINARY)

# left down, from center to edge
mask_ldf = cv2.imread('./data/masks/mask_ldf.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_ldf = cv2.threshold(mask_ldf, 10, 255, cv2.THRESH_BINARY)

# left down, towards center
mask_ldt = cv2.imread('./data/masks/mask_ldt.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_ldt = cv2.threshold(mask_ldt, 10, 255, cv2.THRESH_BINARY)

# left down, towards center, left trun
mask_ldt_l = cv2.imread('./data/masks/mask_ldt_left.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_ldt_l = cv2.threshold(mask_ldt_l, 10, 255, cv2.THRESH_BINARY)

# left upper, from center to edge
mask_luf = cv2.imread('./data/masks/mask_luf.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_luf = cv2.threshold(mask_luf, 10, 255, cv2.THRESH_BINARY)

# left upper, towards center
mask_lut = cv2.imread('./data/masks/mask_lut.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_lut = cv2.threshold(mask_lut, 10, 255, cv2.THRESH_BINARY)

# left upper, towards center, left trun
mask_lut_l = cv2.imread('./data/masks/mask_lut_left.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_lut_l = cv2.threshold(mask_lut_l, 10, 255, cv2.THRESH_BINARY)

# right down, from center to edge
mask_rdf = cv2.imread('./data/masks/mask_rdf.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_rdf = cv2.threshold(mask_rdf, 10, 255, cv2.THRESH_BINARY)

# right down, towards center
mask_rdt = cv2.imread('./data/masks/mask_rdt.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_rdt = cv2.threshold(mask_rdt, 10, 255, cv2.THRESH_BINARY)

# right down, towards center, left trun
mask_rdt_l = cv2.imread('./data/masks/mask_rdt_left.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_rdt_l = cv2.threshold(mask_rdt_l, 10, 255, cv2.THRESH_BINARY)

# right upper, from center to edge
mask_ruf = cv2.imread('./data/masks/mask_ruf.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_ruf = cv2.threshold(mask_ruf, 10, 255, cv2.THRESH_BINARY)
# right upper, towards center
mask_rut = cv2.imread('./data/masks/mask_rut.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_rut = cv2.threshold(mask_rut, 10, 255, cv2.THRESH_BINARY)

# right upper, towards center, left trun
mask_rut_l = cv2.imread('./data/masks/mask_rut_left.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_rut_l = cv2.threshold(mask_rut_l, 10, 255, cv2.THRESH_BINARY)

mask_tuple = (mask_center, mask_ldf, mask_ldt, mask_ldt_l,
              mask_luf, mask_lut, mask_lut_l,
              mask_rdf, mask_rdt, mask_rdt_l,
              mask_ruf, mask_rut, mask_rut_l)

crop_position = [[370,100],[480,0],[480,0],[480,0],
                [0,0],[100,0],[100,0],
                [480,300],[480,400],[480,400],
                [50,380],[0,300],[0,300]]

area_threshold = [3000, 2895.0, 3000, 2106.357, 3000, 0, 
                  3000, 3000, 1000, 1000, 3000, 3000, 3000]

######################## BgOpticalFlowTrigger #########################
class BgOpticalFlowTrigger():
    def __init__(self, img, mask_flow, zone_mask_tuple):
        self.mask_flow = mask_flow
        self.zone_mask_tuple = zone_mask_tuple
        img_prev_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img_prev_gray = cv2.bitwise_and(img_prev_gray, img_prev_gray, mask = mask_flow)
        self.flow_region_mean_prev = [0 for _ in range(13)]
        
    def update(self, img):
        # Mask incoming frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.bitwise_and(img_gray, img_gray, mask = self.mask_flow)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(self.img_prev_gray, 
                                            img_gray, 
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Update previous frame
        self.img_prev_gray = img_gray.copy()
        
        # Process magnitude of Optical flow
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag[np.isinf(mag)] = 0 # sometimes flow elements of 0 is reagarded as a number close to 0 but not equal to 0 by Python, then the cartToPolar function will get an inf value.
        mag[np.isnan(mag)] = 0
        self.mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U) #  mag shape: (960, 1280)
        _, flow_contour = cv2.threshold(self.mag, 40, 255, cv2.THRESH_BINARY)
        
        ################### Morphlogical Transformation #######################
        contours, _ = cv2.findContours(flow_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if cv2.contourArea(contour) < 100:
                cv2.fillConvexPoly(flow_contour, contour, (0, 0, 0))
                
        flow_contour = cv2.morphologyEx(flow_contour, cv2.MORPH_OPEN, kernel, iterations=1)
        flow_contour = cv2.dilate(flow_contour, kernel3, iterations=1)
        
        contours, _ = cv2.findContours(flow_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            cv2.fillConvexPoly(flow_contour, contour, (255,255,255))
            
        contours, _ = cv2.findContours(flow_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if cv2.contourArea(contour) < 100:
                cv2.fillConvexPoly(flow_contour, contour, (0,0,0))
                
        contours, _ = cv2.findContours(flow_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if cv2.contourArea(contour) < 100:
                cv2.fillConvexPoly(flow_contour, contour, (0,0,0))
        ################### Morphlogical Transformation #######################
        
        self.contours, _ = cv2.findContours(flow_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.flow_contour = flow_contour.copy()
        
    def is_zone_triggered(self, zone_id):
        cur_mask = self.zone_mask_tuple[zone_id].copy()
        flow_region = cv2.bitwise_and(self.flow_contour, self.flow_contour, mask = cur_mask)
        flow_region_contours, _ = cv2.findContours(flow_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        total_area = 0
        for contour in flow_region_contours:
            total_area += cv2.contourArea(contour)
            
        flow_region_mean = 0
        flow_region_mean = cv2.mean(cv2.bitwise_and(self.mag, self.mag, mask = flow_region),
                                         mask = flow_region)[0]
        
        # if total_area > area_threshold[zone_id] and flow_region_mean > 50 and flow_region_mean > self.flow_region_mean_prev[zone_id]:
        if total_area > area_threshold[zone_id] and flow_region_mean > 10:
            self.flow_region_mean_prev[zone_id] = flow_region_mean
            return True
        else:
            self.flow_region_mean_prev[zone_id] = flow_region_mean
            return False

######################## ZoneInitializer #########################
class ZoneInitializer():
    def __init__(self, bg_img, zone_mask_tuple, zone_id):
        # Parameters
        self.zone_id = zone_id
        self.score = 0
        self.best_score = 0
        self.update_flag = False
        self.init_flag = True

        # Zone mask
        cur_zone_mask = zone_mask_tuple[zone_id].copy()
        _, self.zone_mask = cv2.threshold(cur_zone_mask, 10, 255, cv2.THRESH_BINARY)
        self.eroded_mask = cv2.erode(self.zone_mask, kernel2, iterations = 1)
        self.eroded_mask = self.eroded_mask[crop_position[self.zone_id][0] : crop_position[self.zone_id][0] + 480,
                                            crop_position[self.zone_id][1] : crop_position[self.zone_id][1] + 480]
        
        # Init SIFT, GMM
        self.sift = cv2.SIFT_create()
        self.bgs_model = bgs.MixtureOfGaussianV2()

        # Init Matcher
        FLANN_INDEX_KDTREE = 0
        indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        searchParams = dict(checks = 50)
        self.flann = cv2.FlannBasedMatcher(indexParams, searchParams)

        # Load old bg for the matcher
        old_bg = cv2.bitwise_and(bg_img, bg_img, mask = self.zone_mask) # template background
        old_bg = old_bg[crop_position[self.zone_id][0] : crop_position[self.zone_id][0] + 480,
                        crop_position[self.zone_id][1] : crop_position[self.zone_id][1] + 480]
        _, self.des_prev_bg = self.sift.detectAndCompute(old_bg, mask = self.eroded_mask) # sift descriptor of the template background

    def get_cur_bgs(self):
        return self.bgs_model.getBackgroundModel()
    
    def get_bg(self):
        if self.update_flag == False:
            return self.get_cur_bgs()
        else:
            return self.bg_output

    def bgs_model_update(self, frame):
        print('Zone', self.zone_id , 'GMM model updated')
        update_region = cv2.bitwise_and(frame, frame, mask = self.zone_mask)
        update_region = update_region[crop_position[self.zone_id][0] : crop_position[self.zone_id][0] + 480,
                                      crop_position[self.zone_id][1] : crop_position[self.zone_id][1] + 480]
        self.bgs_model.apply(update_region)
        self.init_flag = False

    def compare_sift(self):
        # Calculate sift descriptor from current bgs model
        _, des_cur_bg = self.sift.detectAndCompute(self.get_cur_bgs(), mask = self.eroded_mask)
        
        # Calculate match score
        matches = self.flann.knnMatch(self.des_prev_bg, des_cur_bg, k = 2)
        matchNum = [m for (m, n) in matches if m.distance < 0.8 * n.distance]
        self.score = len(matchNum) / len(matches)
        
        # Update bg_output
        if self.score >= self.best_score:
            print('Zone', self.zone_id , 'SIFT satisfied********')
            self.best_score = self.score
            self.bg_output = self.get_cur_bgs() # generated background
            self.update_flag = True
    
    def bg_update(self, frame, of_trigger = True, sift_flg = True):
        if self.init_flag or of_trigger:
            self.bgs_model_update(frame) # update bgs model when triggered
    
        if sift_flg:
            self.compare_sift() # only update BG when have higher score
        else:
            self.bg_output = self.get_cur_bgs() # always update BG
            self.update_flag = True

######################## Zone Init Thread #########################
def zone_initializing_process(cur_frame, trigger, old_bg_img, mask_tuple, zone_id, of_event, zone_init_event, shutdown_event, bg_queue):
    # Initialization
    print('ZoneInit ' + str(zone_id), ': started')
    zone_init = ZoneInitializer(old_bg_img, mask_tuple, zone_id)

    # BG update
    while True:
        if shutdown_event.is_set():
            break

        # print('ZoneInit ' + str(zone_id), ': wait for OpticalFlow')
        of_event.wait()
        zone_init_event.clear()

        ###### ZoneInit Task ######
        zone_init.bg_update(cur_frame, trigger[zone_id])
        ###### ZoneInit Task ######

        # print('ZoneInit ' + str(zone_id), ': finished!!!!!!!')
        zone_init_event.set()
        of_event.clear()
    
    bg_queue.put([zone_init.get_bg(), zone_id])
    print('ZoneInit ' + str(zone_id), 'Done, best match score:', zone_init.best_score)

class AppGUI():
    def __init__(self):
        # ImgProcessor
        self.img_proc = img_processor.ImgProcessor()
        self.frame_receive = False
        self.tmp_img = None
        self.current_raw = None
        self.current_undist = None

        # Canvas Update Parameters
        self.gui_update_time = 200
        self.after_id = None
        self.tmp_img = None

        # Streaming Info
        self.streaming_max_delay = 0.5
        self.streaming_start_time = 0
        self.update_frame_cnt = 0
        self.total_frame_cnt = 0
        self.total_fps = 0
        self.ten_sec_frame_queue = []
        self.ten_sec_fps = 0

        # BG Parameters
        print(os.getcwd())
        self.current_bg_name = './data/background/bg_default.png'
        self.current_bg = self.img_proc.BGR2RGB(self.img_proc.img_loader(self.current_bg_name))
        
        ############################# GUI #############################
        # Main Window
        self.window = tk.Tk()
        self.window.title('GridSmart Detection')
        self.window.geometry('1280x960')
        self.window.resizable(False, False)

        # RTSP Connection
        self.address_label = tk.Label(self.window, text = 'RTSP Address:', font = ('Arial', 15))
        self.address_label.place(relx = 0.01, rely = 0.06, relheight = 0.05, relwidth = 0.11, anchor = tk.NW)

        self.address_var = tk.StringVar()
        self.address_var.set('rtsp://169.235.68.132:9000/1')
        self.address_entry = tk.Entry(self.window, textvariable = self.address_var, show = None, font = ('Arial', 15))
        self.address_entry.place(relx = 0.15, rely = 0.06, relheight = 0.05, relwidth = 0.32, anchor = tk.NW)

        self.streaming_ctr_flg = False
        self.streaming_ctr_btn = tk.Button(self.window, text = 'Start Streaming', command = self.streaming_ctr_fun)
        self.streaming_ctr_btn.place(relx = 0.52, rely = 0.06, relheight = 0.05, relwidth = 0.14, anchor = tk.NW)

        self.streaming_save_flg = False
        self.streaming_save_btn = tk.Button(self.window, text = 'Save Streaming', command = self.streaming_save_fun)
        self.streaming_save_btn.place(relx = 0.67, rely = 0.06, relheight = 0.05, relwidth = 0.14, anchor = tk.NW)

        self.streaming_info_label = tk.Label(self.window, text = 'Streaming Info:', font = ('Arial', 8))
        self.streaming_info_label.place(relx = 0.82, rely = 0.06, relheight = 0.05, relwidth = 0.085, anchor = tk.NW)
        self.total_fps_label = tk.Label(self.window, text = 'Total FPS: ' + str(self.total_fps), font = ('Arial', 8))
        self.total_fps_label.place(relx = 0.9, rely = 0.062, relheight = 0.022, relwidth = 0.1, anchor = tk.NW)
        self.ten_sec_fps_label = tk.Label(self.window, text = '10s FPS: ' + str(self.ten_sec_fps), font = ('Arial', 8))
        self.ten_sec_fps_label.place(relx = 0.9, rely = 0.078, relheight = 0.022, relwidth = 0.1, anchor = tk.NW)

        # Canvas
        self.canvas = tk.Canvas(self.window, height = 640, width = 480, bg = 'gray')
        self.canvas.place(relx = 0.26, rely = 0.2, relheight = 0.7, relwidth = 0.7, anchor = tk.NW)

        # BG Initialization
        self.zone_id_var = tk.StringVar()
        self.zone_id_var.set('ZoneID')
        self.zone_id_entry = tk.Entry(self.window, textvariable = self.zone_id_var, show = None, font = ('Arial', 8))
        self.zone_id_entry.place(relx = 0.01, rely = 0.2, relheight = 0.05, relwidth = 0.05, anchor = tk.NW)
        self.zone_update_time_var = tk.StringVar()
        self.zone_update_time_var.set('Dur')
        self.zone_update_time_entry = tk.Entry(self.window, textvariable = self.zone_update_time_var, show = None, font = ('Arial', 8))
        self.zone_update_time_entry.place(relx = 0.06, rely = 0.2, relheight = 0.05, relwidth = 0.05, anchor = tk.NW)

        self.auto_initialization_btn = tk.Button(self.window, text = 'Auto Init', command = self.auto_init_fun)
        self.auto_initialization_btn.place(relx = 0.12, rely = 0.2, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.init_mode_of_label = tk.Label(self.window, text = 'OF', font = ('Arial', 7))
        self.init_mode_of_label.place(relx = 0.01, rely = 0.25, relheight = 0.02, relwidth = 0.02, anchor = tk.NW)
        self.init_mode_gmm_label = tk.Label(self.window, text = 'GMM', font = ('Arial', 7))
        self.init_mode_gmm_label.place(relx = 0.04, rely = 0.25, relheight = 0.02, relwidth = 0.02, anchor = tk.NW)
        self.init_mode_sift_label = tk.Label(self.window, text = 'SIFF', font = ('Arial', 7))
        self.init_mode_sift_label.place(relx = 0.07, rely = 0.25, relheight = 0.02, relwidth = 0.02, anchor = tk.NW)

        self.init_mode_of_var = tk.BooleanVar()
        self.init_mode_gmm_var = tk.BooleanVar()
        self.init_mode_sift_var = tk.BooleanVar()
        self.init_mode_of_check = tk.Checkbutton(self.window, variable = self.init_mode_of_var, onvalue = True, offvalue= False)
        self.init_mode_gmm_check = tk.Checkbutton(self.window, variable = self.init_mode_gmm_var, onvalue = True, offvalue= False)
        self.init_mode_sift_check = tk.Checkbutton(self.window, variable = self.init_mode_sift_var, onvalue = True, offvalue= False)
        self.init_mode_of_check.place(relx = 0.01, rely = 0.27, relheight = 0.02, relwidth = 0.02, anchor = tk.NW)
        self.init_mode_gmm_check.place(relx = 0.04, rely = 0.27, relheight = 0.02, relwidth = 0.02, anchor = tk.NW)
        self.init_mode_sift_check.place(relx = 0.07, rely = 0.27, relheight = 0.02, relwidth = 0.02, anchor = tk.NW)

        self.zone_bg_update_btn = tk.Button(self.window, text = 'Zone BG Update', command = self.manual_init_fun)
        self.zone_bg_update_btn.place(relx = 0.12, rely = 0.25, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.bg_load_btn = tk.Button(self.window, text = 'BG Load', command = self.load_bg)
        self.bg_load_btn.place(relx = 0.01, rely = 0.3, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.bg_maintain_btn = tk.Button(self.window, text = 'BG Maintaining', command = self.auto_init_fun)
        self.bg_maintain_btn.place(relx = 0.12, rely = 0.3, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        # Detection
        self.detection_param_var = tk.StringVar()
        self.detection_param_var.set('Detection Param')
        self.detection_param_entry = tk.Entry(self.window, textvariable = self.detection_param_var, show = None, font = ('Arial', 8))
        self.detection_param_entry.place(relx = 0.01, rely = 0.38, relheight = 0.05, relwidth = 0.21, anchor = tk.NW)

        self.detection_ctr_flg = False
        self.detection_ctr_btn = tk.Button(self.window, text = 'Start Detection', command = self.detection_ctr_fun)
        self.detection_ctr_btn.place(relx = 0.01, rely = 0.43, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)
        
        self.detection_save_flg = False
        self.detection_save_btn = tk.Button(self.window, text = 'Save Detection', command = self.detection_save_fun)
        self.detection_save_btn.place(relx = 0.12, rely = 0.43, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        # Visualization Selection
        self.VS_var = tk.IntVar()
        self.VS_var.set(1)

        self.VS_none = tk.Radiobutton(self.window, text = 'None', variable = self.VS_var, value = 0)
        self.VS_none.place(relx = 0.01, rely = 0.6, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.VS_raw = tk.Radiobutton(self.window, text = 'Raw', variable = self.VS_var, value = 1)
        self.VS_raw.place(relx = 0.12, rely = 0.6, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.VS_undist = tk.Radiobutton(self.window, text = 'Undist', variable = self.VS_var, value = 2)
        self.VS_undist.place(relx = 0.01, rely = 0.65, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.VS_detection = tk.Radiobutton(self.window, text = 'Detection', variable = self.VS_var, value = 3)
        self.VS_detection.place(relx = 0.12, rely = 0.65, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)
        
        self.VS_BG = tk.Radiobutton(self.window, text = 'Dynamic BG', variable = self.VS_var, value = 4)
        self.VS_BG.place(relx = 0.12, rely = 0.78, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.VS_zone_var = tk.StringVar()
        self.VS_zone_var.set(('All', 'Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 
                              'Zone 5', 'Zone 6', 'Zone 7', 'Zone 8', 'Zone 9', 'Zone 10', 'Zone 11', 'Zone 12', 'Zone 13'))
        self.VS_zone_listbox = tk.Listbox(self.window, listvariable = self.VS_zone_var)
        self.VS_zone_listbox.select_set(0)
        self.VS_zone_listbox.place(relx = 0.01, rely = 0.7, relheight = 0.2, relwidth = 0.1, anchor = tk.NW)

        # Communication
        self.socket_connection_flg = False
        self.socket_connection = tk.Button(self.window, text = 'Connect Server', command = self.socket_connection_fun)
        self.socket_connection.place(relx = 0.01, rely = 0.52, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.socket_send_flg = False
        self.socket_send = tk.Button(self.window, text = 'Send Data', command = self.socket_send_fun)
        self.socket_send.place(relx = 0.12, rely = 0.52, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

    def streaming_ctr_fun(self):
        if self.streaming_ctr_flg:
            print('Streaming manually stopped!')
            self.frame_receive = False
            self.streaming_ctr_flg = False
            self.streaming_ctr_btn.config(text = 'Start Streaming')
            self.streaming_save_flg = False
            self.streaming_save_btn.config(text = 'Save Streaming')
            self.detection_ctr_flg = False
            self.detection_ctr_btn.config(text = 'Start Detection')
            self.detection_save_flg = False
            self.detection_save_btn.config(text = 'Save Detection')
            self.address_entry.config({'background': 'white'})
            self.detection_param_entry.config({'background': 'white'})
            self.window.after_cancel(self.after_id)
            self.after_id = None
            self.total_frame_cnt = 0
            self.update_frame_cnt = 0
            self.total_fps = 0
            self.ten_sec_frame_queue = []
            self.ten_sec_fps = 0
            self.total_fps_label.config(text = 'Total FPS: 0.0')
            self.ten_sec_fps_label.config(text = '10s FPS: 0.0')
        else:
            print('RTSP connect...')
            self.streaming_ctr_flg = True
            self.streaming_ctr_btn.config(text = 'Stop Streaming')
            address = self.address_entry.get()
            self.cur_rtsp_collector = rtsp_collector.RtspCollector(address) ##### Init streaming collector #####
            self.frame_receive, _ = self.cur_rtsp_collector.get_frame()

            if self.frame_receive:
                print('Conneced!')
                self.address_entry.config({'background': 'green'})
                self.streaming_start_time = time.time()

                cur_streaming_thread = threading.Thread(target = self.streaming_thread, daemon=True)
                cur_streaming_thread.start()

                # set visualization canvas
                if self.after_id == None:
                    self.after_id = self.window.after(self.gui_update_time, self.canvas_update)
            else:
                print('Streaming fail!')
                self.address_entry.config({'background': 'red'})
                self.streaming_ctr_flg = False
                self.streaming_ctr_btn.config(text = 'Start Streaming')
                self.window.after_cancel(self.after_id)
                self.after_id = None
          
    def streaming_save_fun(self):
        if self.streaming_save_flg:
            print('Stop saving raw')
            self.streaming_save_flg = False
            self.streaming_save_btn.config(text = 'Save Streaming')
        else:
            if not self.streaming_ctr_flg:
                tk.messagebox.showerror('Error', 'Please start streaming first!')
            else:
                print('Save raw!')
                self.streaming_save_flg = True
                self.streaming_save_btn.config(text = 'Stop Saving Streaming')

    def streaming_thread(self):
        print('Streaming!')
        while self.frame_receive and self.streaming_ctr_flg:
            self.frame_receive, frame = self.cur_rtsp_collector.get_frame()
            
            if self.frame_receive:
                # Calculate FPS
                cur_time = time.time()
                self.total_frame_cnt += 1
                self.total_fps = self.total_frame_cnt / (cur_time - self.streaming_start_time)
                self.total_fps = round(self.total_fps, 1)
                while len(self.ten_sec_frame_queue) > 0 and cur_time - self.ten_sec_frame_queue[0] > 10:
                    self.ten_sec_frame_queue.pop(0)
                self.ten_sec_frame_queue.append(cur_time)
                if cur_time - self.streaming_start_time < 10:
                    self.ten_sec_fps = self.total_fps
                else:
                    self.ten_sec_fps = round(len(self.ten_sec_frame_queue) / 10, 1)

                # Frame skip to assure fps
                if cur_time - self.streaming_start_time < 10:
                    if self.total_frame_cnt > (cur_time - self.streaming_start_time - self.streaming_max_delay) * 10:
                        ####### Update Current Image #######
                        self.current_raw = self.img_proc.get_raw(frame)
                        self.current_undist = self.img_proc.get_undist(frame)
                        self.update_frame_cnt += 1

                        # Update Streaming Info on GUI    
                        self.total_fps_label.config(text = 'Total FPS: ' + str(self.total_fps))
                        self.ten_sec_fps_label.config(text = '10s FPS: ' + str(self.ten_sec_fps))

                        # Save Raw and Undist
                        if self.streaming_save_flg:
                            now = datetime.utcnow().strftime('%Y-%m-%d-%H_%M_%S.%f')[:-3]
                            self.img_proc.img_saver(self.img_proc.RGB2BGR(self.current_raw), '../data/raw/', now)
                            self.img_proc.img_saver(self.img_proc.RGB2BGR(self.current_undist), '../data/undist/', now)
                    else:
                        print('### Skip frame, updated frame in 10s:', self.total_frame_cnt)
                else:
                    if len(self.ten_sec_frame_queue)  > (10 - self.streaming_max_delay) * 10:
                        ####### Update Current Image #######
                        self.current_raw = self.img_proc.get_raw(frame)
                        self.current_undist = self.img_proc.get_undist(frame)
                        self.update_frame_cnt += 1

                        # Update Streaming Info on GUI    
                        self.total_fps_label.config(text = 'Total FPS: ' + str(self.total_fps))
                        self.ten_sec_fps_label.config(text = '10s FPS: ' + str(self.ten_sec_fps))

                        # Save Raw and Undist
                        if self.streaming_save_flg:
                            now = datetime.utcnow().strftime('%Y-%m-%d-%H_%M_%S.%f')[:-3]
                            self.img_proc.img_saver(self.img_proc.RGB2BGR(self.current_raw), '../data/raw/', now)
                            self.img_proc.img_saver(self.img_proc.RGB2BGR(self.current_undist), '../data/undist/', now)
                    else:
                        print('### Skip frame, updated frame in 10s:', len(self.ten_sec_frame_queue))
            else:
                print('Streaming fail!')
                self.address_entry.config({'background': 'red'})
                self.streaming_ctr_flg = False
                self.streaming_ctr_btn.config(text = 'Start Streaming')
                self.total_frame_cnt = 0
                self.update_frame_cnt = 0
                self.total_fps = 0
                self.ten_sec_frame_queue = []
                self.ten_sec_fps = 0

    def detection_ctr_fun(self):
        if self.detection_ctr_flg:
            print('Detection stopped!')
            self.detection_ctr_flg = False
            self.detection_ctr_btn.config(text = 'Start Detection')
            self.detection_save_flg = False
            self.detection_save_btn.config(text = 'Save Detection')
            self.detection_param_entry.config({'background': 'white'})
        else:
            if not self.streaming_ctr_flg:
                tk.messagebox.showerror('Error', 'Please start streaming first!')
            else:
                print('Start detection...')
                self.detection_ctr_flg = True
                self.detection_ctr_btn.config(text = 'Stop Detection')
                self.detection_param_entry.config({'background': 'green'})

                cur_detection_thread = threading.Thread(target = self.detection_thread, daemon=True)
                cur_detection_thread.start()

    def detection_save_fun(self):
        if self.detection_save_flg:
            self.detection_save_flg = False
            self.detection_save_btn.config(text = 'Save Detection')
        else:
            if not self.detection_ctr_flg:
                tk.messagebox.showerror('Error', 'Please start detection first!')
            else:
                self.detection_save_flg = True
                self.detection_save_btn.config(text = 'Stop Saving')

    def detection_thread(self):
        print('Detecting!')
        update_frame_cnt_last = -1
        while self.detection_ctr_flg:
            if self.update_frame_cnt > update_frame_cnt_last:
                print('detecting frame:', self.update_frame_cnt)
                update_frame_cnt_last = self.update_frame_cnt
                time.sleep(0.06)
            else:
                pass
    
    def socket_connection_fun(self):
        pass

    def socket_send_fun(self):
        pass

    def socket_thread(self):
        pass

    def canvas_update(self):
        # None
        if self.VS_var.get() == 0:
            pass
        # Raw
        elif self.VS_var.get() == 1:
            self.tmp_img = self.cv2tk(self.current_raw)
            self.canvas.create_image(0, 0, image = self.tmp_img, anchor = tk.NW)
        # Undist
        elif self.VS_var.get() == 2:
            self.tmp_img = self.cv2tk(self.current_undist)
            self.canvas.create_image(0, 0, image = self.tmp_img, anchor = tk.NW)
        # Detection
        elif self.VS_var.get() == 3:
            self.tmp_img = self.cv2tk(self.current_undist)
            self.canvas.create_image(0, 0, image = self.tmp_img, anchor = tk.NW)
        # BG
        else:
            if self.VS_zone_listbox.curselection():
                cur_VS_zone = self.VS_zone_listbox.curselection()[0]
            else:
                cur_VS_zone = 0

            if cur_VS_zone == 0:
                self.tmp_img = self.cv2tk(self.current_bg)
                self.canvas.create_image(0, 0, image = self.tmp_img, anchor = tk.NW)
            else:
                self.tmp_img = self.cv2tk(cv2.bitwise_and(self.current_bg.copy(), 
                                                          self.current_bg.copy(), 
                                                          mask = mask_tuple[cur_VS_zone - 1]))
                self.canvas.create_image(0, 0, image = self.tmp_img, anchor = tk.NW)

        self.after_id = self.window.after(self.gui_update_time, self.canvas_update)

    def cv2tk(self, cv_img):
        cv_img_resized = self.img_proc.img_resizer(cv_img)
        return PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img_resized))

    def load_bg(self):
        filetype = (('JPG', '*.jpg'), ('PNG', '*.png'), )
        self.current_bg_name = fd.askopenfilename(title = 'Load a background', initialdir='../data/', filetypes = filetype)
        print('Load:')
        print(self.current_bg_name)

    def auto_init_fun(self):
        if not self.streaming_ctr_flg:
            tk.messagebox.showerror('Error', 'Please start streaming first!')
        else:
            running_time = self.zone_update_time_entry.get()
            if running_time.isnumeric():
                running_time = int(running_time)
            else:
                running_time = 10

            print('Auto Init', running_time, 'sec...')

            auto_init_main = threading.Thread(target = self.auto_init_main_thread, args = (running_time,), daemon=True)
            auto_init_main.start()
            self.zone_update_time_entry.config({'background': 'red'})

    def auto_init_main_thread(self, running_time):
        print('Auto Init Main Thread Started...')
        start_time = time.time()

        old_bg_img = cv2.imread('./data/background/bg_history.png')
        mask_flow = cv2.imread('./data/masks/mask_flow.png', flags = cv2.IMREAD_GRAYSCALE)

        number_of_zone = 13
        frame_idx = 0
        frame_num = 1000

        ##### Shared Memory #####
        cur_frame_shared_memory = shared_memory.SharedMemory(create = True, size = 960 * 960 * 3)
        cur_frame_buffer = np.ndarray((960, 960, 3), dtype = 'uint8', buffer = cur_frame_shared_memory.buf)

        trigger_shared_memory = shared_memory.ShareableList([True for _ in range(number_of_zone)])
        
        ##### BG Queue #####
        manager = Manager()
        bg_queue = manager.Queue()

        ##### Events #####
        of_event = Event()
        zone_init_events = [Event() for _ in range(number_of_zone)]
        shutdown_event = Event()

        ##### Start Multi-process #####
        bg_init_process_list = [Process(target = zone_initializing_process, 
                                        args = (cur_frame_buffer, trigger_shared_memory,
                                                old_bg_img, mask_tuple, zone_id, 
                                                of_event, zone_init_events[zone_id], shutdown_event, bg_queue))
                                for zone_id in range(number_of_zone)]
        for p in bg_init_process_list:
            p.start()

        time.sleep(0.1)

        cur_frame = self.current_undist[:,160:-160]

        of_trigger = BgOpticalFlowTrigger(cur_frame, mask_flow, mask_tuple)

        while 1:
            frame_idx += 1
            # if frame_idx >= frame_num:
            if time.time() - start_time > running_time:
                of_event.set()
                shutdown_event.set()
                break

            print('\n######### Current frame:', frame_idx, 'Running time:', int(time.time() - start_time), ' ############')
            of_event.clear()

            ###### OpticalFlow Task ######
            cur_frame = self.current_undist[:,160:-160]
            cur_frame_buffer[:] = cur_frame[:]

            ########## Use OF or not ############
            # of_trigger.update(cur_frame)

            # for idx in range(number_of_zone):
            #     trigger_shared_memory[idx] = of_trigger.is_zone_triggered(idx)

            # print('Main Trigger:', trigger_shared_memory)
            

            # print('OpticalFlow: finished!!!!!!!')
            of_event.set()

            # print('Main: Wait for ZoneInit')
            for zone_init_event in zone_init_events:
                zone_init_event.clear()
            for zone_init_event in zone_init_events:
                zone_init_event.wait()

        for p in bg_init_process_list:
            p.join()

        bg_result_list = [bg_queue.get() for _ in range(number_of_zone)]
        bg_final = np.zeros((960,960,3), dtype = 'uint8')
        for i in range(number_of_zone):
            zero_img = np.zeros((960,960,3), dtype = 'uint8')

            zero_img[crop_position[bg_result_list[i][1]][0] : crop_position[bg_result_list[i][1]][0] + 480,
                     crop_position[bg_result_list[i][1]][1] : crop_position[bg_result_list[i][1]][1] + 480] = bg_result_list[i][0]
            bg_final= cv2.add(bg_final,zero_img)
            del zero_img

        end_time = time.time()
        print('##### Sec/Frame:', (end_time - start_time) / frame_idx, '#####')

        now = datetime.utcnow().strftime('%Y-%m-%d-%H_%M_%S.%f')[:-3]
        self.img_proc.img_saver(self.img_proc.RGB2BGR(bg_final), './data/background/', now)
    
        ###### Close SharedMemory ######
        cur_frame_shared_memory.close()
        cur_frame_shared_memory.unlink()
        trigger_shared_memory.shm.close()
        trigger_shared_memory.shm.unlink()

        self.current_bg = bg_final
        self.zone_update_time_entry.config({'background': 'green'})
        time.sleep(0.5)
        self.zone_update_time_entry.config({'background': 'white'})
        print('All Initialization Done.')

    def manual_init_fun(self):
        if not self.streaming_ctr_flg:
            tk.messagebox.showerror('Error', 'Please start streaming first!')
        else:
            running_time = self.zone_update_time_entry.get()
            if running_time.isnumeric():
                running_time = int(running_time)
            else:
                running_time = 10

            zone_id = self.zone_id_entry.get()
            if zone_id.isnumeric() and int(zone_id) < 13:
                zone_id = int(zone_id)
            else:
                zone_id = 0

            print('Zone', zone_id, 'Init', running_time, 'sec...')

            auto_init_main = threading.Thread(target = self.manual_init_main_thread, args = (running_time, zone_id), daemon=True)
            auto_init_main.start()
            self.zone_id_entry.config({'background': 'red'})
            self.zone_update_time_entry.config({'background': 'red'})

    def manual_init_main_thread(self, running_time, zone_id):
        print('Manual Init Main Thread Started...')
        start_time = time.time()

        old_bg_img = cv2.imread('./data/background/bg_history.png')
        mask_flow = cv2.imread('./data/masks/mask_flow.png', flags = cv2.IMREAD_GRAYSCALE)

        frame_idx = 0
        frame_num = 1000

        ##### Shared Memory #####
        cur_frame_shared_memory = shared_memory.SharedMemory(create = True, size = 960 * 960 * 3)
        cur_frame_buffer = np.ndarray((960, 960, 3), dtype = 'uint8', buffer = cur_frame_shared_memory.buf)

        trigger_shared_memory = shared_memory.ShareableList([True, ])
        
        ##### BG Queue #####
        manager = Manager()
        bg_queue = manager.Queue()

        ##### Events #####
        of_event = Event()
        zone_init_events = [Event(),]
        shutdown_event = Event()

        ##### Start Multi-process #####
        bg_init_process_list = [Process(target = zone_initializing_process, 
                                        args = (cur_frame_buffer, trigger_shared_memory,
                                                old_bg_img, mask_tuple, zone_id, 
                                                of_event, zone_init_events[zone_id], shutdown_event, bg_queue)),]

        for p in bg_init_process_list:
            p.start()

        time.sleep(0.1)

        cur_frame = self.current_undist[:,160:-160]

        of_trigger = BgOpticalFlowTrigger(cur_frame, mask_flow, mask_tuple)

        while 1:
            frame_idx += 1
            # if frame_idx >= frame_num:
            if time.time() - start_time > running_time:
                of_event.set()
                shutdown_event.set()
                break

            print('\n######### Current frame:', frame_idx, 'Running time:', int(time.time() - start_time), ' ############')
            of_event.clear()

            ###### OpticalFlow Task ######
            cur_frame = self.current_undist[:,160:-160]
            cur_frame_buffer[:] = cur_frame[:]

            ########## Use OF or not ############
            # of_trigger.update(cur_frame)

            # for idx in range(number_of_zone):
            #     trigger_shared_memory[idx] = of_trigger.is_zone_triggered(idx)

            # print('Main Trigger:', trigger_shared_memory)
            

            # print('OpticalFlow: finished!!!!!!!')
            of_event.set()

            # print('Main: Wait for ZoneInit')
            for zone_init_event in zone_init_events:
                zone_init_event.clear()
            for zone_init_event in zone_init_events:
                zone_init_event.wait()

        for p in bg_init_process_list:
            p.join()

        bg_result_list = [bg_queue.get(),]
        bg_final = np.zeros((960,960,3), dtype = 'uint8')

        zero_img = np.zeros((960,960,3), dtype = 'uint8')

        zero_img[crop_position[bg_result_list[0][1]][0] : crop_position[bg_result_list[0][1]][0] + 480,
                 crop_position[bg_result_list[0][1]][1] : crop_position[bg_result_list[0][1]][1] + 480] = bg_result_list[0][0]
        bg_final= cv2.add(bg_final,zero_img)
        del zero_img

        end_time = time.time()

        print('##### Sec/Frame:', (end_time - start_time) / frame_idx, '#####')

        now = datetime.utcnow().strftime('%Y-%m-%d-%H_%M_%S.%f')[:-3]
        self.img_proc.img_saver(self.img_proc.RGB2BGR(bg_final), './data/background/', now)
    
        ###### Close SharedMemory ######
        cur_frame_shared_memory.close()
        cur_frame_shared_memory.unlink()
        trigger_shared_memory.shm.close()
        trigger_shared_memory.shm.unlink()

        self.current_bg = bg_final
        self.zone_update_time_entry.config({'background': 'green'})
        self.zone_id_entry.config({'background': 'green'})
        time.sleep(0.5)
        self.zone_update_time_entry.config({'background': 'white'})
        self.zone_id_entry.config({'background': 'white'})
        print('All Initialization Done.')

    def gui_run(self):
        self.window.mainloop()

if __name__ == '__main__':
    cur_app_GUI = AppGUI()
    cur_app_GUI.gui_run()