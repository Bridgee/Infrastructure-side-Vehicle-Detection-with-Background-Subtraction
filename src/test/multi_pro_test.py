from multiprocessing import Process, Pool
import os

class worker():
    def __init__(self, name, age) -> None:
        self.name = name
        self.age = age
    
    def work(self, time):
        print(self.name, 'is working for', time, 'hrs')

def main():
    mike = worker('mike', 30)

    cur_process = Process(target=work_wrapper, args=(mike, 5))
    cur_process.start()
    cur_process.join()
    print('Main age:', mike.age)

def task(num):
    print('Run task %s...' % (os.getpid()))

def work_wrapper(instance, *args):
    instance.work(*args)
    instance.age += 5
    print('Age:', instance.age)

if __name__ == '__main__':
    main()