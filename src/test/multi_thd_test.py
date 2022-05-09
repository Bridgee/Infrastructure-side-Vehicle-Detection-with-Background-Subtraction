import threading
import time

from sympy import arg
def run():
    global video
    video = 0

    streaming_thread = threading.Thread(target = thd1)
    streaming_thread.start()
    
    task_thread1 = threading.Thread(target = thd2, args=(1, ))
    task_thread1.start()

    task_thread2 = threading.Thread(target = thd2, args=(2, ))
    task_thread2.start()

    task_thread1.join()
    task_thread2.join()

def thd1():
    global video
    print('start streaming')

    while True:
        video += 1
        print('thd1:', video)
        time.sleep(1)

def thd2(zone_id):
    global video

    print('    AutoInit Start')
    time.sleep(zone_id)
    print(zone_id, video)
    print('    AutoInit Finish')

if __name__ == '__main__':
    run()