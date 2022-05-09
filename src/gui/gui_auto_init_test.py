import sys, os
sys.path.append(os.getcwd() + '/src')
print(os.getcwd())
from utilities import img_processor
from bg_init import zone_init_fun

zone_init_fun.auto_init_run(10)