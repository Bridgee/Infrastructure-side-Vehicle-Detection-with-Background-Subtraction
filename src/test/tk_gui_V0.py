"""
This is the tkinter GUI for the real-time detection algorithm
"""

__author__ = 'Zhouqiao Zhao'

import sys, os
sys.path.append(os.getcwd())
import tkinter as tk
import PIL.Image, PIL.ImageTk
from tkinter import filedialog as fd
from detection import img_processor
from streaming import rtsp_collector
import time
from datetime import datetime
import threading

class AppGUI():
    def __init__(self):
        self.window =tk.Tk()
        self.window.title('GridSmart Detection')
        self.window.geometry('1280x960')
        self.window.resizable(False, False)

        self.img_proc = img_processor.ImgProcessor()
        self.canvas_show = False
        self.delay = 10
        self.after_id = None

        self.tmp_img = None
        self.current_raw = None
        self.current_undist = None
        self.bg_img_name = '../data/background/bg_default.png'
        self.current_bg = self.img_proc.BGR2RGB(self.img_proc.img_loader(self.bg_img_name))
        
        # RTSP Connection
        self.address_label = tk.Label(self.window, text='RTSP Address', font = ('Arial', 15), width = 30, height = 5)
        self.address_label.place(relx = 0.01, rely = 0.05, relheight = 0.05, relwidth = 0.15, anchor = tk.NW)

        self.address_var = tk.StringVar()
        self.address_var.set('rtsp://169.235.68.132:9000/1')
        self.address_entry = tk.Entry(self.window, textvariable = self.address_var, show=None, font = ('Arial', 15))
        self.address_entry.place(relx = 0.15, rely = 0.05, relheight = 0.05, relwidth = 0.4, anchor = tk.NW)

        self.RTSP_connection = tk.Button(self.window, text = 'RTSP Connection', command = self.rtsp_connect)
        self.RTSP_connection.place(relx = 0.56, rely = 0.05, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.RTSP_streaming = tk.Button(self.window, text = 'RTSP Streaming', command = self.rtsp_streaming)
        self.RTSP_streaming.place(relx = 0.68, rely = 0.05, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.RTSP_streaming = tk.Button(self.window, text = 'RTSP Stop', command = self.rtsp_stop)
        self.RTSP_streaming.place(relx = 0.8, rely = 0.05, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        # BG Initialization
        self.auto_initialization = tk.Button(self.window, text = 'Auto Init', command = self.auto_init)
        self.auto_initialization.place(relx = 0.07, rely = 0.25, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.zone_id_var = tk.StringVar()
        self.zone_id_var.set('Zone ID')
        self.zone_id_entry = tk.Entry(self.window, textvariable = self.zone_id_var, show=None, font = ('Arial', 8))
        self.zone_id_entry.place(relx = 0.01, rely = 0.31, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.zone_update_time_var = tk.StringVar()
        self.zone_update_time_var.set('Update Duration')
        self.zone_update_time_entry = tk.Entry(self.window, textvariable = self.zone_update_time_var, show=None, font = ('Arial', 8))
        self.zone_update_time_entry.place(relx = 0.12, rely = 0.31, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.zone_bg_update = tk.Button(self.window, text = 'Zone BG Update', command = self.auto_init)
        self.zone_bg_update.place(relx = 0.07, rely = 0.38, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.bg_load = tk.Button(self.window, text = 'BG Load', command = self.load_bg)
        self.bg_load.place(relx = 0.07, rely = 0.45, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        # Canvas
        self.canvas = tk.Canvas(self.window, height = 640, width = 480, bg = 'gray')
        self.canvas.place(relx = 0.25, rely = 0.15, relheight = 0.5, relwidth = 0.5, anchor = tk.NW)

        # Visualization Selection
        self.radiobutton_var = tk.IntVar()
        self.radiobutton_var.set(1)

        self.radiobutton_1 = tk.Radiobutton(self.window, text = 'Raw', variable = self.radiobutton_var, value = 1) # (command = self.radiobutton_fun)
        self.radiobutton_1.place(relx = 0.07, rely = 0.65, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.radiobutton_2 = tk.Radiobutton(self.window, text = 'Undist', variable = self.radiobutton_var, value = 2)
        self.radiobutton_2.place(relx = 0.37, rely = 0.65, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.radiobutton_3 = tk.Radiobutton(self.window, text = 'Detection', variable = self.radiobutton_var, value = 3)
        self.radiobutton_3.place(relx = 0.57, rely = 0.65, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.radiobutton_4 = tk.Radiobutton(self.window, text = 'Dynamic BG', variable = self.radiobutton_var, value = 4)
        self.radiobutton_4.place(relx = 0.77, rely = 0.65, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        # Data Saver
        self.save_raw_button = tk.Button(self.window, text = 'Save Raw', command = self.save_raw)
        self.save_raw_button.place(relx = 0.07, rely = 0.7, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        self.save_undist_button = tk.Button(self.window, text = 'Save Undist', command = self.save_undist)
        self.save_undist_button.place(relx = 0.37, rely = 0.7, relheight = 0.05, relwidth = 0.1, anchor = tk.NW)

        # UTP Communication
        self.socket_connection = tk.Button(self.window, text = 'Communication Connection', command = self.auto_init)
        self.socket_connection.place(relx = 0.8, rely = 0.31, relheight = 0.05, relwidth = 0.15, anchor = tk.NW)

        self.socket_send = tk.Button(self.window, text = 'Send Data', command = self.auto_init)
        self.socket_send.place(relx = 0.8, rely = 0.38, relheight = 0.05, relwidth = 0.15, anchor = tk.NW)

        # Detection Summary
        self.detection_summary_label = tk.Label(self.window, text='Detection Summary', background = 'grey',font = ('Arial', 15), width = 800, height = 50)
        self.detection_summary_label.place(relx = 0.1, rely = 0.8, relheight = 0.15, relwidth = 0.8, anchor = tk.NW)

    def test_thread(self):
        for i in range(10):
            print('Current: ', str(i))
            time.sleep(1)

    def save_raw(self):
        now = datetime.utcnow().strftime('%Y-%m-%d-%H_%M_%S.%f')[:-3]
        self.img_proc.img_saver(self.img_proc.RGB2BGR(self.current_raw), '../data/raw/', now)
    
    def save_undist(self):
        now = datetime.utcnow().strftime('%Y-%m-%d-%H_%M_%S.%f')[:-3]
        self.img_proc.img_saver(self.img_proc.RGB2BGR(self.current_undist), '../data/undist/', now)

    def rtsp_connect(self):
        print('RTSP connect...')
        address = self.address_entry.get()
        self.cur_rtsp_collector = rtsp_collector.RtspCollector(address)
        ret, _ = self.cur_rtsp_collector.get_frame()
        if ret:
            self.address_entry.config({"background": "green"})
            print('Sucess')
        else:
            self.address_entry.config({"background": "red"})

    def rtsp_streaming(self):
        print('Start Streaming')
        self.canvas_show = True
        if self.after_id == None:
            self.after_id = self.window.after(self.delay, self.canvas_update)

    def canvas_update(self):
        if self.canvas_show:
            ret, frame = self.cur_rtsp_collector.get_frame()
            if ret:
                self.current_raw = self.img_proc.get_raw(frame)
                self.current_undist = self.img_proc.get_undist(frame)

            # Show Dynamic BG
            if self.radiobutton_var.get() == 4:
                self.tmp_img = self.cv2tk(self.current_bg)
            else:
                # Show Raw
                if self.radiobutton_var.get() == 1:
                    self.tmp_img = self.cv2tk(self.current_raw)
                # Show Undist
                elif self.radiobutton_var.get() == 2:
                    self.tmp_img = self.cv2tk(self.current_undist)
                # Show Detection
                else:
                    self.tmp_img = self.cv2tk(self.current_undist)
            self.canvas.create_image(0, 0, image = self.tmp_img, anchor = tk.NW)
            
        self.after_id = self.window.after(self.delay, self.canvas_update)

    def cv2tk(self, cv_img):
        cv_img_resized = self.img_proc.img_resizer(cv_img)
        return PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img_resized))

    def rtsp_stop(self):
        self.window.after_cancel(self.after_id)
        self.after_id = None

    def auto_init(self):
        test_T = threading.Thread(target = self.test_thread)
        test_T.start()

    def zone_bg_update(self, zone_id, duration):
        a = 1

    def load_bg(self):
        filetype = (('JPG', '*.jpg'), ('PNG', '*.png'), )
        filename = fd.askopenfilename(title = 'Load a background', initialdir='../', filetypes = filetype)
        self.bg_img_name = filename
        print(filename)

    def gui_run(self):
        self.window.mainloop()

if __name__ == '__main__':
    cur_app_GUI = AppGUI()
    cur_app_GUI.gui_run()