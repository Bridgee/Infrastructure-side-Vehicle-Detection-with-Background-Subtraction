import multiprocessing
import cv2
import time
import os
import numpy as np
np.set_printoptions(suppress=True)
from multiprocessing import Process, shared_memory, Event, Manager
from math import radians, degrees, cos, sin, asin, sqrt, tan, atan2
import pybgs as bgs
import matplotlib.pyplot as plt

######################## Parameters Setup #########################
kernel = np.ones((1, 1), np.uint8)
kernel2 = np.ones((2, 2), np.uint8)
kernel3 = np.ones((3, 3), np.uint8)
kernel4 = np.ones((4, 4), np.uint8)
kernel5 = np.ones((5, 5), np.uint8)
kernel6 = np.ones((6, 6), np.uint8)
kernel7 = np.ones((7, 7), np.uint8)
kernel8 = np.ones((8, 8), np.uint8)
kernel9 = np.ones((9, 9), np.uint8)

path = './src/test/test_data'
files = sorted(os.listdir(path), key=len)
files.sort(key = lambda x: int(x[:-4]))
files_num = len(files)

old_bg_img = cv2.imread('./data/background/bg_history.png')
mask_flow = cv2.imread('./data/masks/mask_flow.png', flags = cv2.IMREAD_GRAYSCALE)

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

area_threshold = [3000, 2895.0, 3000, 2106.357, 3000, 1615.25, 
                  3000, 3000, 3000, 3000, 3000, 3000, 3000]

######################## BgOpticalFlowTrigger #########################
class BgOpticalFlowTrigger():
    def __init__(self, img, mask_flow, zone_mask_tuple):
        self.mask_flow = mask_flow
        self.zone_mask_tuple = zone_mask_tuple
        img_prev_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img_prev_gray = cv2.bitwise_and(img_prev_gray, img_prev_gray, mask = mask_flow)
        self.flow_region_mean_prev = 0
        
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
        
        flow_region = cv2.bitwise_and(self.flow_contour, self.flow_contour, cur_mask)
        flow_region_contours, _ = cv2.findContours(flow_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        total_area = 0
        for contour in flow_region_contours:
            total_area += cv2.contourArea(contour)
            
        flow_region_mean = 0
        flow_region_mean = cv2.mean(cv2.bitwise_and(self.mag, self.mag, mask = flow_region),
                                         mask = flow_region)[0]
        
        # if total_area > area_threshold[zone_id] and flow_region_mean > 50 and flow_region_mean > self.flow_region_mean_prev:
        if total_area > area_threshold[zone_id] and flow_region_mean > 50:
            self.flow_region_mean_prev = flow_region_mean
            return True
        else:
            self.flow_region_mean_prev = flow_region_mean
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
        
        # Init SIFT, GMM
        self.sift = cv2.SIFT_create()
        self.bgs_model = bgs.MixtureOfGaussianV2()

        # Init Matcher
        FLANN_INDEX_KDTREE=0
        indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        searchParams = dict(checks = 50)
        self.flann = cv2.FlannBasedMatcher(indexParams, searchParams)

        # Load old bg for the matcher
        old_bg = cv2.bitwise_and(bg_img, bg_img, mask = self.zone_mask) # template background
        # old_bg = old_bg[crop_position[self.zone_id][0] : crop_position[self.zone_id][0] + 480,
        #                 crop_position[self.zone_id][1] : crop_position[self.zone_id][1] + 480]
        _, self.des_prev_bg = self.sift.detectAndCompute(old_bg, None) # sift descriptor of the template background

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
        # update_region = update_region[crop_position[self.zone_id][0] : crop_position[self.zone_id][0] + 480,
        #                               crop_position[self.zone_id][1] : crop_position[self.zone_id][1] + 480]
        self.bgs_model.apply(update_region)
        self.init_flag = False

    def compare_sift(self):
        # Calculate sift descriptor from current bgs model
        _, des_cur_bg = self.sift.detectAndCompute(self.get_cur_bgs(), None)
        
        # Calculate match score
        matches = self.flann.knnMatch(self.des_prev_bg, des_cur_bg, k = 2)
        matchNum = [m for (m, n) in matches if m.distance < 0.8 * n.distance]
        self.score = len(matchNum) / len(matches)
        
        # Update bg_output
        if self.score > self.best_score:
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
        # print('    ZoneInit ' + str(zone_id), 'calculating..., trigger:', trigger[zone_id])
        zone_init.bg_update(cur_frame, trigger[zone_id])
        ###### ZoneInit Task ######

        # print('ZoneInit ' + str(zone_id), ': finished!!!!!!!')
        zone_init_event.set()
        of_event.clear()

    if 0:
        plt.imshow(cv2.cvtColor(zone_init.get_bg(), cv2.COLOR_BGR2RGB))
        plt.title('Zone' + str(zone_id) + ', Update: ' + str(zone_init.update_flag))
        plt.show()
    
    bg_queue.put(zone_init.get_bg())
    print('ZoneInit ' + str(zone_id), 'Done, best match score:', zone_init.best_score)

def auto_init_run(running_time = 100):
    start_time = time.time()

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

    cur_frame = cv2.imread(os.path.join(path,files[frame_idx]))
    of_trigger = BgOpticalFlowTrigger(cur_frame, mask_flow, mask_tuple)

    while 1:
        frame_idx += 1
        # if frame_idx >= frame_num:
        if time.time() - start_time > running_time:
            of_event.set()
            shutdown_event.set()
            break

        print('\n######### Current frame:', frame_idx, files[frame_idx], ' ############')
        of_event.clear()

        ###### OpticalFlow Task ######
        cur_frame = cv2.imread(os.path.join(path, files[frame_idx]))
        cur_frame_buffer[:] = cur_frame[:]
        of_trigger.update(cur_frame)

        for idx in range(number_of_zone):
            trigger_shared_memory[idx] = of_trigger.is_zone_triggered(idx)
        # print('Main Trigger:', trigger_shared_memory)
        
        # Show current frame
        if 0:
            plt.figure()
            plt.subplot(121)
            plt.imshow(of_trigger.flow_contour, cmap = 'gray')

            plt.subplot(122)
            plt.imshow(cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB))
            plt.show()

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

    bg_final = bg_result_list[0]
    for zone_id in range(1, number_of_zone):
        bg_final += bg_result_list[zone_id]

    end_time = time.time()
    print('##### Sec/Frame:', (end_time - start_time) / frame_idx, '#####')

    if 0:
        plt.imshow(cv2.cvtColor(bg_final, cv2.COLOR_BGR2RGB))
        plt.show()

    print('All Done.')
    
    ###### Close SharedMemory ######
    cur_frame_shared_memory.close()
    cur_frame_shared_memory.unlink()
    trigger_shared_memory.shm.close()
    trigger_shared_memory.shm.unlink()

if __name__ == '__main__':
    auto_init_run()