"""
04_10_2022
This is the tkinter GUI for the real-time detection algorithm.
05_09_2022
The major functions include:
1. Auto background initialization
2. Manual zone-background initialization
3. RTSP streaming and auto frame skip
4. Dynamic background maintainnance
5. Background subtraction-based object detection
6. Optical-flow-based moving angle detection
7. UDP-based result broadcasting
"""

__author__ = ['Zhouqiao Zhao', 'Jiahe Cao']

import sys, os
sys.path.append(os.getcwd() + '/src')
from socket import *
import tkinter as tk
import PIL.Image, PIL.ImageTk
from tkinter import filedialog as fd
from tkinter import messagebox
import time
from datetime import datetime
import threading
from multiprocessing import Process, shared_memory, Event, Manager
import numpy as np
import cv2
import queue
import math
from math import atan2, pi
import csv
import json
import pybgs as bgs

from streaming import rtsp_collector
from config import *

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
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
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
        
        # if total_area > area_threshold[zone_id] and flow_region_mean > 10:
        if total_area > area_threshold[zone_id] and flow_region_mean > 25 and flow_region_mean > (self.flow_region_mean_prev[zone_id]+3):
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
        self.best_var = np.inf
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
        self.old_bg = old_bg[crop_position[self.zone_id][0] : crop_position[self.zone_id][0] + 480,
                             crop_position[self.zone_id][1] : crop_position[self.zone_id][1] + 480]
        _, self.des_prev_bg = self.sift.detectAndCompute(self.old_bg, mask = self.eroded_mask) # sift descriptor of the template background

    def get_cur_bgs(self):
        return self.bgs_model.getBackgroundModel()
    
    def get_bg(self):
        if self.update_flag == False:
            return self.get_cur_bgs()
        else:
            return self.bg_output

    def bgs_model_update(self, frame):
        update_region = cv2.bitwise_and(frame, frame, mask = self.zone_mask)
        update_region = update_region[crop_position[self.zone_id][0] : crop_position[self.zone_id][0] + 480,
                                      crop_position[self.zone_id][1] : crop_position[self.zone_id][1] + 480]
        self.bgs_model.apply(update_region)
        print('Zone', self.zone_id , 'GMM model updated')
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
        
        self.best_score = self.best_score * 0.9999
    
    def compare_var(self):
        img_dif = cv2.absdiff(self.get_cur_bgs(), self.old_bg)
        img_dif = cv2.pow(img_dif.astype(np.int32), 2)
        self.var = np.sum(img_dif)

        # print('Zone', self.zone_id ,'variance', self.var, self.best_var)
        if self.var <= self.best_var:
            print('Zone', self.zone_id , 'Variation satisfied********')
            self.best_var = self.var
            self.bg_output = self.get_cur_bgs() # generated background
            self.update_flag = True

    def bg_update(self, frame, trigger_times, of_trigger = True, of_flg = True, var_flg = True, sift_flg = True):
        if self.init_flag or of_trigger or not of_flg:
            self.bgs_model_update(frame)
    
        if var_flg:
            self.compare_var()
        elif sift_flg:
            if trigger_times > 50:
                self.compare_sift()
        else:
            print('Zone', self.zone_id , '********')
            self.bg_output = self.get_cur_bgs() # always update BG
            self.update_flag = True

######################## Zone Init Thread #########################
def zone_initializing_process(cur_frame, trigger, of_on, var_on, sift_on, old_bg_img, mask_tuple, zone_id, of_event, zone_init_event, shutdown_event, bg_queue):
    # Initialization
    print('ZoneInit ' + str(zone_id), ': started')
    zone_init = ZoneInitializer(old_bg_img, mask_tuple, zone_id)
    count = 0
    trigger_times = 0

    # BG update
    while True:
        if shutdown_event.is_set():
            break

        of_event.wait()
        zone_init_event.clear()

        ###### ZoneInit Task ######
        if trigger[zone_id]:
            trigger_times += 1

        zone_init.bg_update(cur_frame, trigger_times, of_trigger = trigger[zone_id], of_flg = of_on, var_flg = var_on, sift_flg = sift_on)

        # cv2.imwrite('./data/bgsimg/'+str(zone_id)+'_'+str(count)+'_'+str(zone_init.var)+'.png', zone_init.get_cur_bgs())
        count += 1
        ###### ZoneInit Task ######

        zone_init_event.set()
        of_event.clear()

    bg_queue.put([zone_init.get_bg(), zone_id])
    
    print('ZoneInit ' + str(zone_id), 'Done, best match score:', zone_init.best_score)

class AppGUI():
    def __init__(self):
        # Img
        self.frame_receive = False
        self.current_raw = None
        self.current_undist = None
        self.current_detection = None
        self.tmp_img = None
        self.mirror_tmp_img = None

        # Canvas Update Parameters
        self.gui_update_time = 100
        self.after_id = None

        # Streaming Info
        self.streaming_max_delay = 0.5
        self.streaming_start_time = 0
        self.update_frame_cnt = 0
        self.total_frame_cnt = 0
        self.total_fps = 0
        self.ten_sec_frame_queue = []
        self.ten_sec_fps = 0

        # BG Parameters
        self.current_bg = cv2.cvtColor(cv2.imread('./data/background/bg_default.png'), cv2.COLOR_BGR2RGB)
        self.mirror_map = cv2.cvtColor(cv2.imread('./data/mirror_map.png'), cv2.COLOR_BGR2RGB)
        self.current_mirror = self.mirror_map.copy()
        self.detection_file = None
        
        ############################# GUI #############################
        # Main Window
        self.window = tk.Tk()
        self.window.title('GridSmart Detection')
        self.window.geometry('1400x960')
        self.window.resizable(False, False)

        # Logo
        self.logo_img = PIL.Image.open('./data/logo.png')
        self.logo_img = self.logo_img.resize((int(self.logo_img.width / 2), int(self.logo_img.height / 2)))
        self.logo_img = PIL.ImageTk.PhotoImage(self.logo_img)
        self.logo_canvas = tk.Canvas(self.window, bg = 'gray')
        self.logo_canvas.place(relx = 0.715, rely = 0.905, relheight = 0.09, relwidth = 0.28, anchor = tk.NW)
        self.logo_canvas.create_image(0, 0, image = self.logo_img, anchor = tk.NW)

        # RTSP Connection
        self.address_label = tk.Label(self.window, text = 'RTSP Address:', font = ('Arial', 15))
        self.address_label.place(relx = 0.02, rely = 0.02, relheight = 0.05, relwidth = 0.11, anchor = tk.NW)

        self.address_var = tk.StringVar()
        self.address_var.set('rtsp://169.235.68.132:9000/1')
        self.address_entry = tk.Entry(self.window, textvariable = self.address_var, show = None, font = ('Arial', 15))
        self.address_entry.place(relx = 0.15, rely = 0.02, relheight = 0.05, relwidth = 0.32, anchor = tk.NW)

        self.streaming_ctr_flg = False
        self.streaming_ctr_btn = tk.Button(self.window, text = 'Start Streaming', command = self.streaming_ctr_fun)
        self.streaming_ctr_btn.place(relx = 0.52, rely = 0.02, relheight = 0.05, relwidth = 0.14, anchor = tk.NW)

        self.streaming_save_flg = False
        self.streaming_save_btn = tk.Button(self.window, text = 'Save Streaming', command = self.streaming_save_fun)
        self.streaming_save_btn.place(relx = 0.67, rely = 0.02, relheight = 0.05, relwidth = 0.14, anchor = tk.NW)

        self.streaming_info_label = tk.Label(self.window, text = 'Streaming Info:', font = ('Arial', 8))
        self.streaming_info_label.place(relx = 0.82, rely = 0.02, relheight = 0.05, relwidth = 0.085, anchor = tk.NW)
        self.total_fps_label = tk.Label(self.window, text = 'Total FPS: ' + str(self.total_fps), font = ('Arial', 8))
        self.total_fps_label.place(relx = 0.9, rely = 0.022, relheight = 0.022, relwidth = 0.1, anchor = tk.NW)
        self.ten_sec_fps_label = tk.Label(self.window, text = '10s FPS: ' + str(self.ten_sec_fps), font = ('Arial', 8))
        self.ten_sec_fps_label.place(relx = 0.9, rely = 0.038, relheight = 0.022, relwidth = 0.1, anchor = tk.NW)

        # Main Canvas
        self.main_label = tk.Label(self.window, text = 'Real world:', font = ('Arial', 14))
        self.main_label.place(relx = 0.01, rely = 0.25, relheight = 0.05, relwidth = 0.08, anchor = tk.NW)

        self.canvas = tk.Canvas(self.window, bg = 'gray')
        self.canvas.place(relx = 0.01, rely = 0.3, relheight = 0.6, relwidth = 0.55, anchor = tk.NW)

        # Mirror Environment Canvas
        self.mirror_label = tk.Label(self.window, text = 'CMM:', font = ('Arial', 14))
        self.mirror_label.place(relx = 0.57, rely = 0.25, relheight = 0.05, relwidth = 0.04, anchor = tk.NW)

        self.mirror_canvas = tk.Canvas(self.window, bg = 'gray')
        self.mirror_canvas.place(relx = 0.57, rely = 0.3, relheight = 0.6, relwidth = 0.38, anchor = tk.NW)

        ########### BG Initialization ###########
        self.zone_id_var = tk.StringVar()
        self.zone_id_var.set('Zone ID')
        self.zone_id_entry = tk.Entry(self.window, textvariable = self.zone_id_var, show = None, font = ('Arial', 8))
        self.zone_id_entry.place(relx = 0.01, rely = 0.1, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.zone_update_time_var = tk.StringVar()
        self.zone_update_time_var.set('Duration')
        self.zone_update_time_entry = tk.Entry(self.window, textvariable = self.zone_update_time_var, show = None, font = ('Arial', 8))
        self.zone_update_time_entry.place(relx = 0.01, rely = 0.15, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.auto_initialization_btn = tk.Button(self.window, text = 'Auto Init', command = self.auto_init_fun)
        self.auto_initialization_btn.place(relx = 0.12, rely = 0.1, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.init_mode_of_label = tk.Label(self.window, text = 'OF', font = ('Arial', 7))
        self.init_mode_of_label.place(relx = 0.01, rely = 0.2, relheight = 0.02, relwidth = 0.02, anchor = tk.NW)
        self.init_mode_var_label = tk.Label(self.window, text = 'VAR', font = ('Arial', 7))
        self.init_mode_var_label.place(relx = 0.04, rely = 0.2, relheight = 0.02, relwidth = 0.02, anchor = tk.NW)
        self.init_mode_sift_label = tk.Label(self.window, text = 'SIFT', font = ('Arial', 7))
        self.init_mode_sift_label.place(relx = 0.07, rely = 0.2, relheight = 0.02, relwidth = 0.02, anchor = tk.NW)

        self.init_mode_of_var = tk.BooleanVar()
        self.init_mode_of_var.set(True)
        self.init_mode_of_check = tk.Checkbutton(self.window, variable = self.init_mode_of_var, onvalue = True, offvalue= False)
        self.init_mode_of_check.place(relx = 0.01, rely = 0.22, relheight = 0.02, relwidth = 0.02, anchor = tk.NW)

        self.init_mode_var_var = tk.BooleanVar()
        self.init_mode_var_var.set(True)
        self.init_mode_var_check = tk.Checkbutton(self.window, variable = self.init_mode_var_var, onvalue = True, offvalue= False)
        self.init_mode_var_check.place(relx = 0.04, rely = 0.22, relheight = 0.02, relwidth = 0.02, anchor = tk.NW)

        self.init_mode_sift_var = tk.BooleanVar()
        self.init_mode_sift_check = tk.Checkbutton(self.window, variable = self.init_mode_sift_var, onvalue = True, offvalue= False)
        self.init_mode_sift_check.place(relx = 0.07, rely = 0.22, relheight = 0.02, relwidth = 0.02, anchor = tk.NW)

        self.zone_bg_update_btn = tk.Button(self.window, text = 'Zone BG Update', command = self.manual_init_fun)
        self.zone_bg_update_btn.place(relx = 0.12, rely = 0.15, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.bg_load_btn = tk.Button(self.window, text = 'BG Load', command = self.bg_load)
        self.bg_load_btn.place(relx = 0.12, rely = 0.20, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.VS_zone_var = tk.StringVar()
        self.VS_zone_var.set(('All', 'Zone 0', 'Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 
                              'Zone 5', 'Zone 6', 'Zone 7', 'Zone 8', 'Zone 9', 'Zone 10', 'Zone 11', 'Zone 12'))
        self.VS_zone_listbox = tk.Listbox(self.window, listvariable = self.VS_zone_var, selectmode = 'multiple', font = ('Arial', 6))
        self.VS_zone_listbox.select_set(0)
        self.VS_zone_listbox.place(relx = 0.23, rely = 0.1, relheight = 0.15, relwidth = 0.1, anchor = tk.NW)

        self.bar_canvas_1 = tk.Canvas(self.window, bg = 'black')
        self.bar_canvas_1.place(relx = 0.36, rely = 0.1, relheight = 0.15, relwidth = 0.005, anchor = tk.NW)

        ########### BG Maintain ###########
        self.bg_maintain_flg = False
        self.bg_maintain_btn = tk.Button(self.window, text = 'BG Maintaining', command = self.bg_maintain_fun)
        self.bg_maintain_btn.place(relx = 0.4, rely = 0.1, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)
        
        self.save_bg_freq_var = tk.StringVar()
        self.save_bg_freq_var.set('Save 10 f/update')
        self.save_bg_freq_entry = tk.Entry(self.window, textvariable = self.save_bg_freq_var, show = None, font = ('Arial', 7))
        self.save_bg_freq_entry.place(relx = 0.4, rely = 0.15, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.bg_save_flg = False
        self.bg_save_btn = tk.Button(self.window, text = 'Save BG', command = self.bg_save_fun)
        self.bg_save_btn.place(relx = 0.4, rely = 0.20, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.bar_canvas_2 = tk.Canvas(self.window, bg = 'black')
        self.bar_canvas_2.place(relx = 0.53, rely = 0.1, relheight = 0.15, relwidth = 0.005, anchor = tk.NW)

        ########### Detection ###########
        self.detection_ctr_flg = False
        self.detection_ctr_btn = tk.Button(self.window, text = 'Start Detection', command = self.detection_ctr_fun)
        self.detection_ctr_btn.place(relx = 0.57, rely = 0.1, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.save_dt_freq_var = tk.StringVar()
        self.save_dt_freq_var.set('Save 10 f/update')
        self.save_dt_freq_entry = tk.Entry(self.window, textvariable = self.save_dt_freq_var, show = None, font = ('Arial', 7))
        self.save_dt_freq_entry.place(relx = 0.57, rely = 0.15, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.save_dt_img_flg = False
        self.save_dt_img_btn = tk.Button(self.window, text = 'Save Detect Img', command = self.detection_img_save_fun)
        self.save_dt_img_btn.place(relx = 0.68, rely = 0.15, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.detection_save_flg = False
        self.detection_save_btn = tk.Button(self.window, text = 'Save Detect CSV', command = self.detection_save_fun)
        self.detection_save_btn.place(relx = 0.57, rely = 0.2, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.bar_canvas_2 = tk.Canvas(self.window, bg = 'black')
        self.bar_canvas_2.place(relx = 0.81, rely = 0.1, relheight = 0.15, relwidth = 0.005, anchor = tk.NW)

        ########### Visualization Selection ###########
        self.VS_var = tk.IntVar()
        self.VS_var.set(1)

        self.VS_raw = tk.Radiobutton(self.window, text = 'Raw', variable = self.VS_var, value = 1)
        self.VS_raw.place(relx = 0.01, rely = 0.9, relheight = 0.03, relwidth = 0.1, anchor = tk.NW)

        self.VS_undist = tk.Radiobutton(self.window, text = 'Undist', variable = self.VS_var, value = 2)
        self.VS_undist.place(relx = 0.11, rely = 0.9, relheight = 0.03, relwidth = 0.1, anchor = tk.NW)

        self.VS_detection = tk.Radiobutton(self.window, text = 'Detection', variable = self.VS_var, value = 3)
        self.VS_detection.place(relx = 0.21, rely = 0.9, relheight = 0.03, relwidth = 0.1, anchor = tk.NW)
        
        self.VS_BG = tk.Radiobutton(self.window, text = 'Dynamic BG', variable = self.VS_var, value = 4)
        self.VS_BG.place(relx = 0.31, rely = 0.9, relheight = 0.03, relwidth = 0.1, anchor = tk.NW)

        # Communication
        self.socket_connection_flg = False
        self.socket_connection = tk.Button(self.window, text = 'Connect Server', command = self.socket_connection_fun)
        self.socket_connection.place(relx = 0.85, rely = 0.12, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.socket_send_flg = False
        self.socket_send = tk.Button(self.window, text = 'Send Data', command = self.socket_send_fun)
        self.socket_send.place(relx = 0.85, rely = 0.17, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

    def streaming_ctr_fun(self):
        if self.streaming_ctr_flg:
            print('Streaming manually stopped!')
            self.frame_receive = False
            self.streaming_ctr_flg = False
            self.streaming_ctr_btn.config(text = 'Start Streaming')

            self.streaming_save_flg = False
            self.streaming_save_btn.config(text = 'Save Streaming')

            self.bg_maintain_flg = False
            self.bg_maintain_btn.config(text = 'BG Maintaining')

            self.bg_save_flg = False
            self.bg_save_btn.config(text = 'Save BG')

            self.detection_ctr_flg = False
            self.detection_ctr_btn.config(text = 'Start Detection')

            self.detection_save_flg = False
            self.detection_save_btn.config(text = 'Save Detect CSV')

            self.save_dt_img_flg = False
            self.save_dt_img_btn.config(text = 'Save Detect Img')

            self.address_entry.config({'background': 'white'})

            if self.detection_file is not None: self.detection_file.close()

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
            print('Stop saving streaming data')
            self.streaming_save_flg = False
            self.streaming_save_btn.config(text = 'Save Streaming')
        else:
            if not self.streaming_ctr_flg:
                messagebox.showerror('Error', 'Please start streaming first!')
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
                        self.current_raw = frame.copy()
                        self.current_undist = cv2.fisheye.undistortImage(frame, K = K, D = D, Knew = Knew)
                        self.update_frame_cnt += 1

                        # Update Streaming Info on GUI    
                        self.total_fps_label.config(text = 'Total FPS: ' + str(self.total_fps))
                        self.ten_sec_fps_label.config(text = '10s FPS: ' + str(self.ten_sec_fps))

                        # Save Raw and Undist
                        if self.streaming_save_flg:
                            now = datetime.now().strftime('%Y-%m-%d_%H_%M_%S.%f')[:-3]
                            cv2.imwrite('./data/raw/' + now + '.png',
                                        cv2.cvtColor(self.current_raw, cv2.COLOR_RGB2BGR))
                            cv2.imwrite('./data/undist/' + now + '.png',
                                        cv2.cvtColor(self.current_undist, cv2.COLOR_RGB2BGR))
                    else:
                        print('### Skip frame, updated frame in 10s:', self.total_frame_cnt)
                else:
                    if len(self.ten_sec_frame_queue)  > (10 - self.streaming_max_delay) * 10:
                        ####### Update Current Image #######
                        self.current_raw = frame.copy()
                        self.current_undist = cv2.fisheye.undistortImage(frame, K = K, D = D, Knew = Knew)
                        self.update_frame_cnt += 1

                        # Update Streaming Info on GUI    
                        self.total_fps_label.config(text = 'Total FPS: ' + str(self.total_fps))
                        self.ten_sec_fps_label.config(text = '10s FPS: ' + str(self.ten_sec_fps))

                        # Save Raw and Undist
                        if self.streaming_save_flg:
                            now = datetime.now().strftime('%Y-%m-%d_%H_%M_%S.%f')[:-3]
                            cv2.imwrite('./data/raw/' + now + '.png',
                                        cv2.cvtColor(self.current_raw, cv2.COLOR_RGB2BGR))
                            cv2.imwrite('./data/undist/' + now + '.png',
                                        cv2.cvtColor(self.current_undist, cv2.COLOR_RGB2BGR))
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

    def auto_init_fun(self):
        if not self.streaming_ctr_flg:
            messagebox.showerror('Error', 'Please start streaming first!')
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

        number_of_zone = 13
        frame_idx = 0

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
                                                self.init_mode_of_var.get(), self.init_mode_var_var.get(), self.init_mode_sift_var.get(),
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
            if self.init_mode_of_var.get():
                of_trigger.update(cur_frame)

                for idx in range(number_of_zone):
                    trigger_shared_memory[idx] = of_trigger.is_zone_triggered(idx)

                # print('Trigger: ', trigger_shared_memory)

            of_event.set()

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
            bg_final = cv2.add(bg_final, zero_img)
            del zero_img

        end_time = time.time()
        print('##### Sec/Frame:', (end_time - start_time) / frame_idx, '#####')
    
        ###### Close SharedMemory ######
        cur_frame_shared_memory.close()
        cur_frame_shared_memory.unlink()
        trigger_shared_memory.shm.close()
        trigger_shared_memory.shm.unlink()

        self.current_bg = bg_final
        if not self.detection_ctr_flg:
            self.current_mirror = self.mirror_map.copy()

        now = datetime.now().strftime('%Y-%m-%d_%H_%M_%S.%f')[:-3]
        cv2.imwrite('./data/background/' + now + '_autoinit.png',
                    cv2.cvtColor(self.current_bg, cv2.COLOR_RGB2BGR))

        self.zone_update_time_entry.config({'background': 'green'})
        time.sleep(1)
        self.zone_update_time_entry.config({'background': 'white'})
        print('All Initialization Done.')

    def manual_init_fun(self):
        if not self.streaming_ctr_flg:
            messagebox.showerror('Error', 'Please start streaming first!')
        else:
            running_time = self.zone_update_time_entry.get()
            if running_time.isnumeric():
                running_time = int(running_time)
            else:
                running_time = 10

            zone_id_text = self.zone_id_entry.get()
            if zone_id_text[0] == '(' and zone_id_text[-1] == ')':
                zone_id = tuple(map(int, zone_id_text[1:-1].split(', ')))
            else:
                zone_id = (0, )

            print('Zone', zone_id, 'Init', running_time, 'sec...')

            auto_init_main = threading.Thread(target = self.manual_init_main_thread, args = (running_time, zone_id), daemon=True)
            auto_init_main.start()
            self.zone_id_entry.config({'background': 'red'})
            self.zone_update_time_entry.config({'background': 'red'})

    def manual_init_main_thread(self, running_time, zone_ids):
        print('Manual Init Main Thread Started...')
        start_time = time.time()

        number_of_zone = len(zone_ids)
        frame_idx = 0

        ##### Shared Memory #####
        cur_frame_shared_memory = shared_memory.SharedMemory(create = True, size = 960 * 960 * 3)
        cur_frame_buffer = np.ndarray((960, 960, 3), dtype = 'uint8', buffer = cur_frame_shared_memory.buf)

        trigger_shared_memory = shared_memory.ShareableList([True for _ in range(13)])
        
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
                                                self.init_mode_of_var.get(), self.init_mode_var_var.get(), self.init_mode_sift_var.get(),
                                                old_bg_img, mask_tuple, zone_id[1], 
                                                of_event, zone_init_events[zone_id[0]], shutdown_event, bg_queue))
                                for zone_id in enumerate(zone_ids)]

        for p in bg_init_process_list:
            p.start()

        time.sleep(0.1)

        cur_frame = self.current_undist[:,160:-160]

        of_trigger = BgOpticalFlowTrigger(cur_frame, mask_flow, mask_tuple)

        while 1:
            frame_idx += 1
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
            if self.init_mode_of_var.get():
                of_trigger.update(cur_frame)

                for zone_id in zone_ids:
                    trigger_shared_memory[zone_id] = of_trigger.is_zone_triggered(zone_id)

                # print('Main Trigger:', trigger_shared_memory)

            of_event.set()

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
            bg_final = cv2.add(bg_final, zero_img)
            del zero_img

        end_time = time.time()

        print('##### Sec/Frame:', (end_time - start_time) / frame_idx, '#####')
    
        ###### Close SharedMemory ######
        cur_frame_shared_memory.close()
        cur_frame_shared_memory.unlink()
        trigger_shared_memory.shm.close()
        trigger_shared_memory.shm.unlink()

        current_bg_no_zone_mask = np.zeros((960,960), dtype = 'uint8')
        for zone_id in range(13):
            if zone_id in zone_ids:
                continue
            else:
                current_bg_no_zone_mask = cv2.add(current_bg_no_zone_mask, mask_tuple[zone_id])
        current_bg_no_zone =cv2.bitwise_and(self.current_bg.copy(), 
                                            self.current_bg.copy(), 
                                            mask = current_bg_no_zone_mask)
        
        self.current_bg = cv2.add(current_bg_no_zone, bg_final)
        if not self.detection_ctr_flg:
            self.current_mirror = self.mirror_map.copy()

        now = datetime.now().strftime('%Y-%m-%d_%H_%M_%S.%f')[:-3]
        cv2.imwrite('./data/background/' + now + '_zoneinit.png',
                    cv2.cvtColor(self.current_bg, cv2.COLOR_RGB2BGR))

        self.zone_update_time_entry.config({'background': 'green'})
        self.zone_id_entry.config({'background': 'green'})
        time.sleep(1)
        self.zone_update_time_entry.config({'background': 'white'})
        self.zone_id_entry.config({'background': 'white'})
        print('Zone Initialization Done.')

    def bg_load(self):
        filetype = (('PNG', '*.png'), ('JPG', '*.jpg'), )
        current_bg_name = fd.askopenfilename(title = 'Load a background', initialdir='./data/background/', filetypes = filetype)
        self.current_bg = cv2.cvtColor(cv2.imread(current_bg_name), cv2.COLOR_BGR2RGB)
        print('Load:')
        print(current_bg_name)

    def bg_maintain_fun(self):
        if self.bg_maintain_flg:
            print('BG maintain stopped!')
            self.bg_maintain_flg = False
            self.bg_maintain_btn.config(text = 'BG Maintaining')
        else:
            if not self.streaming_ctr_flg:
                messagebox.showerror('Error', 'Please start streaming first!')
            else:
                print('Start BG maintain...')
                self.bg_maintain_flg = True
                self.bg_maintain_btn.config(text = 'Stop BG Maintain')

                cur_bg_maintain_thread = threading.Thread(target = self.bg_maintain_thread, daemon=True)
                cur_bg_maintain_thread.start()

    def bg_maintain_thread(self):
        print('BG Maintaining!')
        update_frame_cnt_last = -1

        # Initialization
        algorithm = bgs.MixtureOfGaussianV2()
        algorithm_2 = bgs.MixtureOfGaussianV2()
        algorithm_test = bgs.MixtureOfGaussianV2()

        bg_buffer = self.current_bg.copy()
        bg_buffer_2 = self.current_bg.copy()
        bg_buffer_test = self.current_bg.copy()

        _ = algorithm.apply(self.current_bg)
        _ = algorithm_2.apply(self.current_bg)
        _ = algorithm_test.apply(self.current_bg)

        dirty_mask_queue = queue.Queue(60)

        bg_maintain_cnt = -1
        ratio = 0.2

        while self.bg_maintain_flg:
            # if frame update
            if self.update_frame_cnt > update_frame_cnt_last:
                bg_maintain_cnt += 1
                # print('Maintaining frame:', self.update_frame_cnt, bg_maintain_cnt)
                update_frame_cnt_last = self.update_frame_cnt

                ########## Maintain Task ##########
                current_undist_sample = self.current_undist.copy()
                current_undist_sample = current_undist_sample[:, 160 : -160]
                current_undist_sample = cv2.bitwise_and(current_undist_sample, current_undist_sample, mask = mask_all)
                img_diff = cv2.absdiff(bg_buffer_2, current_undist_sample)

                # now = datetime.now().strftime('%Y-%m-%d_%H_%M_%S.%f')[:-3]
                # cv2.imwrite('./debug/' + now + '.png',
                #             cv2.cvtColor(img_diff, cv2.COLOR_RGB2BGR))

                ##### diff -> normal -> gray -> morph -> contour -> not -> & mask
                img_diff = img_diff / 16
                img_diff_gray = cv2.add(cv2.pow(img_diff[:, :, 0], 2) / 3,
                                        cv2.pow(img_diff[:, :, 1], 2) / 3,
                                        cv2.pow(img_diff[:, :, 2], 2) / 3)

                _, img_bg_opened = cv2.threshold(img_diff_gray, 0.5, 255, cv2.THRESH_BINARY)
                img_bg_opened = cv2.convertScaleAbs(img_bg_opened)
                img_bg_opened = cv2.morphologyEx(img_bg_opened, cv2.MORPH_OPEN, kernel3, iterations = 1)
                img_bg_opened = cv2.dilate(img_bg_opened, kernel2, iterations=1)
                contours, _ = cv2.findContours(img_bg_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                for contour in contours:
                    if cv2.contourArea(contour)>300:
                        cv2.fillConvexPoly(img_bg_opened, contour, (255, 255, 255))
                        cv2.drawContours(img_bg_opened, contour, -1, (255, 255, 255), cv2.FILLED)
                    else:
                        cv2.drawContours(img_bg_opened, contour, -1, (0, 0, 0), cv2.FILLED)

                img_bg_mask = cv2.bitwise_not(img_bg_opened)
                frame_bg = cv2.bitwise_and(mask_all, img_bg_mask)

                # now = datetime.now().strftime('%Y-%m-%d_%H_%M_%S.%f')[:-3]
                # cv2.imwrite('./debug/' + now + '.png',
                #             cv2.cvtColor(frame_bg, cv2.COLOR_RGB2BGR))

                current_frame_renew = cv2.bitwise_and(current_undist_sample, current_undist_sample, mask = frame_bg)

                # now = datetime.now().strftime('%Y-%m-%d_%H_%M_%S.%f')[:-3]
                # cv2.imwrite('./debug/' + now + '.png',
                #             cv2.cvtColor(current_frame_renew, cv2.COLOR_RGB2BGR))

                bg_renew = cv2.addWeighted(current_frame_renew, 
                                           ratio, 
                                           cv2.bitwise_and(bg_buffer, bg_buffer, mask = frame_bg) , 
                                           (1 - ratio), 
                                           0)

                # now = datetime.now().strftime('%Y-%m-%d_%H_%M_%S.%f')[:-3]
                # cv2.imwrite('./debug/' + now + '.png',
                #             cv2.cvtColor(bg_renew, cv2.COLOR_RGB2BGR))

                if bg_maintain_cnt % 2 == 0:
                    bg_keep = cv2.bitwise_and(bg_buffer, bg_buffer, mask = img_bg_opened)
                else:
                    bg_keep = cv2.bitwise_and(bg_buffer_test, bg_buffer_test, mask = img_bg_opened)

                bg_buffer = cv2.add(bg_renew, bg_keep)

                _ = algorithm.apply(bg_buffer) # bg_buffer
                img_bgmodel = algorithm.getBackgroundModel()

                bgs_diff = cv2.absdiff(bg_buffer_test, bg_buffer)

                bgs_diff_gray = cv2.cvtColor(bgs_diff, cv2.COLOR_BGR2GRAY)

                _, bgs_diff_opened = cv2.threshold(bgs_diff_gray, 5, 255, cv2.THRESH_BINARY)

                bgs_diff_opened = cv2.morphologyEx(bgs_diff_opened, cv2.MORPH_OPEN, kernel3, iterations=1)

                bgs_mask = cv2.bitwise_not(bgs_diff_opened)

                frame_bgs = cv2.bitwise_and(mask_all, bgs_mask)

                bg_2_renew =  cv2.addWeighted(cv2.bitwise_and(img_bgmodel,img_bgmodel,mask = frame_bgs), 
                                              0.3,
                                              cv2.bitwise_and(bg_buffer, bg_buffer,mask = frame_bgs), 
                                              0.7, 
                                              0)

                bg_2_keep = cv2.bitwise_and(bg_buffer,bg_buffer,mask = bgs_diff_opened)

                bg_buffer_2 = cv2.add(bg_2_renew, bg_2_keep)

                _ = algorithm_2.apply(bg_buffer_2)

                bg_buffer_2 = algorithm_2.getBackgroundModel()

                ##### correct dirty area which has long-last mask
                if bg_maintain_cnt % 10 == 0:
                    dirty_mask_queue.put(img_bg_opened)
                    if bg_maintain_cnt % 600 == 0:
                        dirty_mask = dirty_mask_queue.get()
                        while not dirty_mask_queue.empty():
                            temp = dirty_mask_queue.get()
                            dirty_mask = cv2.bitwise_and(dirty_mask, temp)

                        bg_buffer_2 = cv2.add(cv2.bitwise_and(bg_buffer_test, bg_buffer_test, mask = dirty_mask), 
                                              cv2.bitwise_and(bg_buffer_2, bg_buffer_2, mask = cv2.bitwise_not(dirty_mask)))

                ##### Output #####
                save_bg_freq = self.save_bg_freq_entry.get()
                if save_bg_freq.isnumeric():
                    save_bg_freq = int(save_bg_freq)
                else:
                    save_bg_freq = 10

                if self.bg_save_flg and bg_maintain_cnt % save_bg_freq == 0:
                    now = datetime.now().strftime('%Y-%m-%d_%H_%M_%S.%f')[:-3]
                    cv2.imwrite('./data/background/' + now + '.png',
                                cv2.cvtColor(bg_buffer_2, cv2.COLOR_RGB2BGR))

                self.current_bg = bg_buffer_2
                if not self.detection_ctr_flg:
                    self.current_mirror = self.mirror_map.copy()
                ##### Output #####

                if bg_maintain_cnt % 50 == 0:
                    _ = algorithm_2.apply(bg_buffer_test)

                if bg_maintain_cnt % 300 == 0:
                    print('################ Maintainance Frame: ', bg_maintain_cnt, '################')
                    _ = algorithm_test.apply(current_undist_sample)
                    bg_buffer_test = algorithm_test.getBackgroundModel()

                ########## Maintain Task ##########
            else:
                pass

    def bg_save_fun(self):
        if self.bg_save_flg:
            print('Stop saving bg')
            self.bg_save_flg = False
            self.bg_save_btn.config(text = 'Save BG')
        else:
            if not self.bg_maintain_flg:
                messagebox.showerror('Error', 'Please start bg maintain first!')
            else:
                print('Save bg!')
                self.bg_save_flg = True
                self.bg_save_btn.config(text = 'Stop Saving bg')

    def detection_ctr_fun(self):
        if self.detection_ctr_flg:
            print('Detection stopped!')
            self.detection_ctr_flg = False
            self.detection_ctr_btn.config(text = 'Start Detection')
            self.detection_save_flg = False
            self.detection_save_btn.config(text = 'Save Detect CSV')
        else:
            if not self.streaming_ctr_flg:
                messagebox.showerror('Error', 'Please start streaming first!')
            else:
                print('Start detection...')
                self.detection_ctr_flg = True
                self.detection_ctr_btn.config(text = 'Stop Detection')

                cur_detection_thread = threading.Thread(target = self.detection_thread, daemon=True)
                cur_detection_thread.start()

    def detection_thread(self):
        print('Detecting!')
        update_frame_cnt_last = -1

        # Initialization
        trajecotry_queue = queue.Queue(10)
        trajecotry_number = 10
        bbox_list_last = np.array([], 'float32')
        detection_cnt = -1
        init_flag = True
        width = 960
        height = 960
        now = datetime.now().strftime('%Y-%m-%d_%H_%M_%S.%f')[:-3]
        self.detection_file = open('./data/results/detection_'+ now + '.csv', 'w')
        detection_writer = csv.writer(self.detection_file)
        detection_writer.writerow(['Time (Receive)', 'Time (Detected)', 'Detection'])
        of_resize = 4

        while self.detection_ctr_flg:
            if self.update_frame_cnt > update_frame_cnt_last:
                detection_cnt += 1
                # print('detecting frame:', self.update_frame_cnt, detection_cnt)
                update_frame_cnt_last = self.update_frame_cnt
                
                ########## Detection Task ##########
                bbox_list_ori = []
                bbox_list = []
                bbox_list_rt = []

                current_undist_sample = self.current_undist.copy()
                current_undist_sample = current_undist_sample[:, 160 : -160]
                current_undist_sample = cv2.bitwise_and(current_undist_sample, current_undist_sample, mask = mask_all)
                current_time_sample = time.time()
                current_detection_mask = np.zeros((960, 960), np.uint8)

                current_bg_sample = self.current_bg.copy()
                # img_showof = self.current_bg.copy()
                img_diff = cv2.absdiff(current_bg_sample, current_undist_sample)

                if init_flag:
                    init_flag = False
                    img_prev_gray = cv2.cvtColor(current_bg_sample, cv2.COLOR_BGR2GRAY)
                    img_prev_gray = cv2.resize(img_prev_gray, 
                                               (int(width / of_resize),int(height / of_resize)),
                                               interpolation = cv2.INTER_AREA)
                    continue
                else:
                    img_undist_low = cv2.resize(current_undist_sample,
                                                (int(width / of_resize),int(height / of_resize)),
                                                interpolation=cv2.INTER_AREA)
                    img_showof = img_undist_low.copy()

                img_diff_gray = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)
                img_gray = cv2.cvtColor(img_undist_low, cv2.COLOR_BGR2GRAY)

                ##### Optical Flow
                flow = cv2.calcOpticalFlowFarneback(img_prev_gray, img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                img_prev_gray = img_gray
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
                mag[np.isinf(mag)] = 0
                mag[np.isnan(mag)] = 0
                mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #  mag shape 960 1280

                mag = cv2.resize(mag, (int(width), int(height)), interpolation = cv2.INTER_AREA)
                _, mag_th = cv2.threshold(mag, 20, 255, cv2.THRESH_BINARY)
                ang = cv2.resize(ang, (int(width), int(height)), interpolation = cv2.INTER_AREA)
                # cv2.imwrite('./data/ang/'+str(detection_cnt)+'_ang'+'.png',ang)
                # y, x = np.mgrid[10/ 2:960:10, 10/ 2:960:10].reshape(2,-1).astype(int)
                # fx,fy =flow[y,x].T
                # lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
                # lines = np.int32(lines)
                # line = []
                # for l in lines:
                #     if l[0][0] - l[1][0] >1 or l[0][1] - l[1][1]>1:
                #         line.append(l)
                # cv2.polylines(img_showof, line, 0, (0,255,255))
                # cv2.imwrite('./data/ang/'+str(detection_cnt)+'.png', img_showof)

                
                ##### Vehicle Contour
                _, img_bin = cv2.threshold(img_diff_gray, 20, 255, cv2.THRESH_BINARY)
                img_opened = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel, iterations=3)
                img_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=3)
                img_dilated = cv2.dilate(img_opened, None, iterations=1)

                contours, _ = cv2.findContours(img_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                for contour in contours:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    if cv2.contourArea(contour) / (1 + math.sqrt((x - 480)**2 + (y - 480)**2) / (2 * 480**2)) > 500:
                        hull = cv2.convexHull(contour, returnPoints = False)
                        try:
                            defects = cv2.convexityDefects(contour, hull)
                        except: continue
                        if defects is None: continue
                        concave_point_list = []
                        concave_point_dis_list = []

                        for i in range(defects.shape[0]):
                            s,e,f,d = defects[i,0]
                            start = tuple(contour[s][0])
                            end = tuple(contour[e][0])
                            far = tuple(contour[f][0])

                            if d > 2500:
                                concave_point_list.append(far)
                                concave_point_dis_list.append(d)

                            if np.size(concave_point_dis_list)>1:
                                index = np.argsort(concave_point_dis_list)
                                cv2.line(img_opened, concave_point_list[index[-1]], concave_point_list[index[-2]], (0,0,0), 3)

                contours, _ = cv2.findContours(img_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                img_contours = current_undist_sample.copy()
                img_contours_mirror = self.mirror_map.copy()

                cntr_cnt = 0
                
                for contour in contours:
                    if cv2.contourArea(contour) / (1 + math.sqrt((x - 480)**2 + (y - 480)**2) / (2 * 480**2)) > 300:
                        # Get bbox
                        (x, y, w, h) = cv2.boundingRect(contour)
                        min_rect = cv2.minAreaRect(contour) # (center), (w, h), rotation
                        min_box_points = cv2.boxPoints(min_rect)
                        min_box_points = min_box_points.astype(int)

                        hull = cv2.convexHull(contour, returnPoints = False)

                        if cv2.contourArea(contour) / (1 + math.sqrt((x - 480)**2 + (y - 480)**2)/(2*480**2)) > 3000 and\
                           (np.max(min_rect[1]) / np.min(min_rect[1]) > 1.5):
                            sub_opened = img_opened[y:y+h+1, x:x+w+1] # Vehicle sub area
                            ero_iter = 3
                            sub_contours = [1, ] # contour after erode

                            # Erode until seperate
                            while len(sub_contours) < 2:
                                sub_eroded = cv2.erode(sub_opened, kernel, iterations = ero_iter)
                                sub_contours, _ = cv2.findContours(sub_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                                ero_iter += 1
                                if ero_iter == 5:
                                    break

                            # Check subcontour after erosion
                            for sub_contour in sub_contours:
                                if cv2.contourArea(sub_contour)/(1 + math.sqrt((x-480)**2+(y-480)**2)/(2*480**2)) > 300:
                                    # print('sub contour', sub_contour)
                                    ang_mask = np.zeros(ang.shape, np.uint8)
                                    sub_contour_t = sub_contour + [x,y]
                                    cv2.drawContours(ang_mask, [sub_contour_t], -1, 255,  thickness=-1)
                                    # cv2.imwrite('./debug/ang/1.png',ang_mask)
                                    ang_mean_raw = cv2.mean(cv2.bitwise_and(ang,ang,mask = ang_mask),  mask=ang_mask)[0]
                                    # ang_mean_img = cv2.bitwise_and(ang,ang,mask = ang_mask)
                                    # ang_mean_raw = ang_mean_img.max()
                                    ang_mean = ang_mean_raw
                                    # Get Bbox
                                    (sub_x, sub_y, sub_w, sub_h) = cv2.boundingRect(sub_contour)
                                    min_rect = cv2.minAreaRect(sub_contour) # center, w, h, rotation
                                    mag_mean = cv2.mean(cv2.bitwise_and(mag,mag,mask = ang_mask),  mask=ang_mask)[0]
                                    ori_ang = min_rect[2]
                                    ang_mean += 90
                                    ang_sent = ang_mean_raw

                                    center_x = sub_x + int(sub_w/2) + x
                                    center_y = sub_y + int(sub_h/2) + y
                                    _, cal_angle = divide_orientation([center_x, center_y])
                                    if not cal_angle:
                                        cal_angle = ang_mean_raw
                                    ang_mean = cal_angle + 90

                                    if mag_mean>20: # otherwise static vehicle
                                        temp = (min_rect[0], (np.min(min_rect[1]), np.max(min_rect[1])), ang_mean)
                                        min_rect = temp
                                    # elif bool(len(bbox_list_last)):# use nearest point angular in the last frame
                                    #     distance_last_frame = np.power(bbox_list_last[:][:2] - np.array([[min_rect[0][0], min_rect[0][1]]]).T, 2)
                                    #     nearest_point_id = np.argmin(np.sum(distance_last_frame,axis = 1))
                                    #     ang_mean = bbox_list_last[nearest_point_id,-1]
                                    #     temp = (min_rect[0], (np.min(min_rect[1]), np.max(min_rect[1])), ang_mean)
                                    #     # temp = (min_rect[0], min_rect[1], ang_mean)
                                    #     min_rect = temp
                                    else: # if there is no bblist_last
                                        if min_rect[1][0] > np.min(min_rect[1]):
                                            ang_mean = min_rect[-1]+90
                                        temp = (min_rect[0], (np.min(min_rect[1]), np.max(min_rect[1])), ang_mean)
                                        min_rect = temp

                                    min_box_points = cv2.boxPoints(min_rect)
                                    min_box_points = min_box_points.astype(int) + [x, y]

                                    hull = cv2.convexHull(sub_contour, returnPoints = False)

                                    new_x = sub_x + x
                                    new_y = sub_y + y



                                    ###### Draw Bbox ######
                                    cv2.drawContours(img_contours, [min_box_points], -1, (0, 255, 0), 3) # Min bbox
                                    cv2.fillConvexPoly(current_detection_mask, min_box_points, (255, ))
                                    cv2.circle(img_contours, (center_x, center_y), 5, (255, 0, 0), -1)
                                    cv2.circle(img_contours_mirror, (center_x, center_y), 5, (0, 255, 0), -1)
                                    _, center_flag = divide_orientation([center_x, center_y]) # if the vehicles are within center area, display the angles
                                    
                                    # if not center_flag:
                                    #     text = ['***' + str(cntr_cnt)+', '+str(round(min_rect[-1],2))+', '+str(round(ang_mean_raw,2))]
                                    #     cv2.putText(img_contours, text[0], (sub_x + x, sub_y + y), cv2.FONT_HERSHEY_PLAIN, 1.6, (0, 0, 255), 2)

                                    bbox_list_ori.append([new_x, new_y, sub_w, sub_h])
                                    lat, lon = find_Lat_Lon(center_x, center_y)
                                    bbox_list_rt.append([round(lat,6), round(lon,6), cal_angle, mag_mean])
                                    distance = np.power(480-center_x,2) + np.power(480-center_y,2)
                                    bbox_list.append([center_x, center_y, min_rect[1][0], min_rect[1][1], mag_mean, min_rect[-1], distance])
                                    cntr_cnt += 1
                        else:
                            ang_mask = np.zeros(ang.shape, np.uint8)
                            mag_mask = np.zeros(ang.shape, np.uint8)
                            cv2.drawContours(ang_mask, [contour], -1, 255, thickness=-1)
                            ang_mask = cv2.bitwise_and(ang_mask,ang_mask,mask = mag_th)
                            
                            ang_mean_img = cv2.bitwise_and(ang,ang,mask = ang_mask)
                            ang_mean_raw = cv2.mean(ang_mean_img,  mask=ang_mask)[0] # *180/np.pi
                            # ang_mean_raw = ang_mean_img.max()
                            mag_mean = cv2.mean(cv2.bitwise_and(mag,mag,mask = ang_mask),  mask=ang_mask)[0]
                            ori_ang = min_rect[2]
                            ang_mean = ang_mean_raw
                            ang_mean += 90
                            ang_sent = ang_mean_raw
                            center_x = x + int(w/2)
                            center_y = y + int(h/2)
                            _, cal_angle = divide_orientation([center_x, center_y])
                            if not cal_angle:
                                # cv2.imwrite('./data/ang/'+str(detection_cnt)+'_'+str(cntr_cnt)+'.png',ang_mean_img)
                                # cal_angle = ang_mean_raw
                                if min_rect[1][0] > np.min(min_rect[1]):
                                    cal_angle = min_rect[-1]
                                else:
                                    cal_angle = min_rect[-1] - 90

                                if bool(len(bbox_list_last)):
                                    distance_last_frame = np.power(bbox_list_last[:][:2] - np.array([[center_x, center_y]]).T, 2)
                                    nearest_point_id = np.argmin(np.sum(distance_last_frame,axis = 1))
                                    last_ang = bbox_list_last[nearest_point_id,-1]
                                    # cal_angle = (atan2(bbox_list_last[nearest_point_id,1]-center_y, bbox_list_last[nearest_point_id,0]-center_x)*180/pi+360)%360
                                    # if abs(last_ang - cal_angle) > 50:
                                    #     cal_angle = last_ang
                                    
                                
                            ang_mean = cal_angle + 90

                            # temp = (min_rect[0], (np.min(min_rect[1]), np.max(min_rect[1])), ang_mean)
                            # min_rect = temp
                            
                            if mag_mean > 20: # otherwise static vehicle
                                temp = (min_rect[0], (np.min(min_rect[1]), np.max(min_rect[1])), ang_mean)
                                mov_flag = 1
                                min_rect = temp
                            # elif bool(len(bbox_list_last)):# use nearest point angular in the last frame
                            #     distance_last_frame = np.power(bbox_list_last[:][:2] - np.array([[min_rect[0][0], min_rect[0][1]]]).T, 2)
                            #     nearest_point_id = np.argmin(np.sum(distance_last_frame,axis = 1))
                            #     ang_mean = bbox_list_last[nearest_point_id,-1]
                                
                            #     temp = (min_rect[0], (np.min(min_rect[1]), np.max(min_rect[1])), ang_mean)
                            #     # temp = (min_rect[0], min_rect[1], ang_mean)
                            #     min_rect = temp
                            else: # if there is no bblist_last
                                if min_rect[1][0] > np.min(min_rect[1]):
                                    ang_mean = min_rect[-1] + 90
                                mov_flag = 0
                                temp = (min_rect[0], (np.min(min_rect[1]), np.max(min_rect[1])), ang_mean)
                                min_rect = temp
                            



                            min_box_points = cv2.boxPoints(min_rect)
                            min_box_points = min_box_points.astype(int)

                            hull = cv2.convexHull(contour, returnPoints = False)
       


                            ###### Draw Bbox ######
                            cv2.drawContours(img_contours, [min_box_points], -1, (0, 255, 0), 3) # Min bbox
                            cv2.fillConvexPoly(current_detection_mask, min_box_points, (255, ))
                            cv2.circle(img_contours, (center_x, center_y), 5, (255, 0, 0), -1) # Position
                            cv2.circle(img_contours_mirror, (center_x, center_y), 5, (0, 255, 0), -1)
                            _, center_flag = divide_orientation([center_x, center_y]) # if the vehicles are within center area, display the angles
                            
                            # if not center_flag:
                            #     text = [str(mov_flag)+', '+str(round(min_rect[-1],2))+', '+str(round(cal_angle,2)) ] # +str(bbox_list_last[nearest_point_id,1]-center_y)+','+str(bbox_list_last[nearest_point_id,0]-center_x)
                            #     cv2.putText(img_contours, text[0], (x, y), cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 0, 0), 2)

                            bbox_list_ori.append([x, y, w, h])
                            lat, lon = find_Lat_Lon(center_x, center_y)
                            bbox_list_rt.append([round(lat,6), round(lon,6), cal_angle, mag_mean]) # Lat, Lon, Yaw, Speed
                            distance = np.power(480-center_x,2) + np.power(480-center_y,2)
                            bbox_list.append([center_x, center_y, min_rect[1][0], min_rect[1][1], mag_mean, min_rect[-1],distance])
                            cntr_cnt += 1
                
                current_time_finish = time.time()
                
                # sort the bound_box according to the distance to center
                bbox_list = np.array(bbox_list)
                bbox_index = np.argsort(bbox_list[:,-1])
                bbox_list = bbox_list[bbox_index]
                bbox_list_last = bbox_list.copy()

                bbox_list_rt = np.array(bbox_list_rt)
                bbox_list_rt = bbox_list_rt[bbox_index]
                bbox_list_rt_last = bbox_list_rt.copy()

                if self.detection_save_flg:
                    detection_writer.writerow([current_time_sample, current_time_finish, bbox_list_last])

                save_dt_img_freq = self.save_dt_freq_entry.get()
                if save_dt_img_freq.isnumeric():
                    save_dt_img_freq = int(save_dt_img_freq)
                else:
                    save_dt_img_freq = 10

                if self.save_dt_img_flg and detection_cnt % save_dt_img_freq == 0:
                    now = datetime.now().strftime('%Y-%m-%d_%H_%M_%S.%f')[:-3]
                    # cv2.imwrite('./data/detection_bbox/' + now + '.png',
                    #             cv2.cvtColor(img_contours, cv2.COLOR_RGB2BGR))
                    current_detection_objects = cv2.bitwise_and(current_undist_sample.copy(), 
                                                                current_undist_sample.copy(), 
                                                                mask = current_detection_mask)
                    cv2.imwrite('./data/detection_objects/' + now + '.png',
                                cv2.cvtColor(current_detection_objects, cv2.COLOR_RGB2BGR))
                
                if self.socket_send_flg:
                    # [Lat, Lon, Yaw, Speed]
                    # [{time, id, lon, lat, alt, x_size, y_size, z_side, yaw, t1, t2, t3, t4, t5}, {}, ...]
                    dict_data = []
                    for id, cur_sample_data in enumerate(bbox_list_rt_last):
                        current_sample_dict = dict()
                        current_sample_dict['time'] = current_time_sample
                        current_sample_dict['id'] = id
                        current_sample_dict['lon'] = cur_sample_data[1]
                        current_sample_dict['lat'] = cur_sample_data[0]
                        current_sample_dict['alt'] = 0
                        current_sample_dict['x_size'] = 120
                        current_sample_dict['y_size'] = 120
                        current_sample_dict['z_side'] = 0
                        current_sample_dict['yaw'] = cur_sample_data[2]
                        current_sample_dict['t1'] = 0
                        current_sample_dict['t2'] = 0
                        current_sample_dict['t3'] = 0
                        current_sample_dict['t4'] = 0
                        current_sample_dict['t5'] = time.time()

                        dict_data.append(current_sample_dict)

                    msg = json.dumps(dict_data)
                    msg = msg +  '\n'
                    self.sock_edge.sendall(msg.encode('utf-8'))
                    
                self.current_detection = img_contours.copy()
                self.current_mirror = img_contours_mirror.copy()
                del ang
                ########## Detection Task ##########
            else:
                pass

    def detection_save_fun(self):
        if self.detection_save_flg:
            print('Stop saving detection')
            self.detection_file.close()
            self.detection_save_flg = False
            self.detection_save_btn.config(text = 'Save Detect CSV')
        else:
            if not self.detection_ctr_flg:
                messagebox.showerror('Error', 'Please start detection first!')
            else:
                print('Save detection!')
                self.detection_save_flg = True
                self.detection_save_btn.config(text = 'Stop Saving')

    def detection_img_save_fun(self):
        if self.save_dt_img_flg:
            print('Stop saving')
            self.save_dt_img_flg = False
            self.save_dt_img_btn.config(text = 'Save Detect Img')
        else:
            if not self.detection_ctr_flg:
                messagebox.showerror('Error', 'Please start detection first!')
            else:
                print('Save detection img!')
                self.save_dt_img_flg = True
                self.save_dt_img_btn.config(text = 'Stop Saving')

    def socket_connection_fun(self):
        if self.socket_connection_flg:
            self.socket_connection_flg = False
        else:
            self.socket_connection_flg = True

            self.sock_edge = socket(AF_INET, SOCK_STREAM)
            print('Socket Created...')

            ADDR = (C_HOST, C_M1_PORT)
            self.sock_edge.connect(ADDR)

            print('Socket Connected to ' + ADDR[0] + ', at Port:' + str(ADDR[1]))

    def socket_send_fun(self):
        if self.socket_send_flg:
            print('Stop sending detection')
            self.socket_send_flg = False
            self.socket_send.config(text = 'Send Data')
        else:
            if not (self.detection_ctr_flg and self.socket_connection_flg):
                messagebox.showerror('Error', 'Please start detection and connect to server first!')
            else:
                print('Send detection!')
                self.socket_send_flg = True
                self.socket_send.config(text = 'Stop Sending')

    def canvas_update(self):
        # None
        if self.VS_var.get() == 0:
            pass
        # Raw
        elif self.VS_var.get() == 1:
            self.tmp_img = self.cv2tk(self.current_raw.copy())
            self.canvas.create_image(0, 0, image = self.tmp_img, anchor = tk.NW)
        # Undist
        elif self.VS_var.get() == 2:
            self.tmp_img = self.cv2tk(self.current_undist.copy())
            self.canvas.create_image(0, 0, image = self.tmp_img, anchor = tk.NW)
        # Detection
        elif self.VS_var.get() == 3:
            if self.detection_ctr_flg:
                if self.current_detection is None:
                    self.tmp_img = self.cv2tk(self.current_undist.copy())
                else:
                    self.tmp_img = self.cv2tk(self.current_detection.copy())
            else:
                self.tmp_img = self.cv2tk(self.current_undist.copy())
            self.canvas.create_image(0, 0, image = self.tmp_img, anchor = tk.NW)
        # BG
        else:
            if self.VS_zone_listbox.curselection():
                cur_VS_zones = self.VS_zone_listbox.curselection()
            else:
                cur_VS_zones = (0, )

            if cur_VS_zones[0] == 0:
                self.tmp_img = self.cv2tk(self.current_bg.copy())
                self.canvas.create_image(0, 0, image = self.tmp_img, anchor = tk.NW)
                self.zone_id_entry.delete(0, tk.END)
                self.zone_id_entry.insert(0, 'all')
            else:
                self.tmp_img = np.zeros((960, 960, 3), dtype = 'uint8')
                for cur_VS_zone in cur_VS_zones:                                        
                    self.tmp_img = cv2.add(self.tmp_img, 
                                        cv2.bitwise_and(self.current_bg.copy(), 
                                                        self.current_bg.copy(), 
                                                        mask = mask_tuple[cur_VS_zone - 1]))
                self.tmp_img = self.cv2tk(self.tmp_img)
                self.canvas.create_image(0, 0, image = self.tmp_img, anchor = tk.NW)
                self.zone_id_entry.delete(0, tk.END)
                cur_VS_zones = tuple([i - 1 for i in cur_VS_zones])
                self.zone_id_entry.insert(0, str(cur_VS_zones))

        # Mirror
        self.mirror_tmp_img = self.cv2tk(self.current_mirror.copy())
        self.mirror_canvas.create_image(0, 0, image = self.mirror_tmp_img, anchor = tk.NW)

        self.after_id = self.window.after(self.gui_update_time, self.canvas_update)

    def cv2tk(self, cv_img):
        cv_img_resized = cv2.resize(cv_img, dsize = None, fx = scale_x, fy = scale_y)
        return PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img_resized))

    def gui_run(self):
        self.window.mainloop()

if __name__ == '__main__':
    cur_app_GUI = AppGUI()
    cur_app_GUI.gui_run()