"""
Run this scipt to start the app.
"""

__author__ = 'Zhouqiao Zhao'

from gui import tk_gui

def run_gui():
    cur_gui = tk_gui.AppGUI()
    cur_gui.gui_run()

if __name__ == '__main__':
    run_gui()