import numpy as np
import cv2
import math

######### Image processing parameters #########
w = 1280
h = 960
f = 331.04228163
K = np.eye(3)

K[0, 0] = f
K[1, 1] = f
K[0, 2] = w / 2
K[1, 2] = h / 2
D = np.array([0., 0., 0., 0.])

Knew = K.copy()
Knew[(0,1), (0,1)] = 0.28 * Knew[(0,1), (0,1)]

kernel = np.ones((1, 1), np.uint8)
kernel2 = np.ones((2, 2), np.uint8)
kernel3 = np.ones((3, 3), np.uint8)
kernel4 = np.ones((4, 4), np.uint8)
kernel5 = np.ones((5, 5), np.uint8)
kernel6 = np.ones((6, 6), np.uint8)
kernel7 = np.ones((7, 7), np.uint8)
kernel8 = np.ones((8, 8), np.uint8)
kernel9 = np.ones((9, 9), np.uint8)

######### Zone parameters #########
center_area = np.array([[[271,365]],[[544,551]],[[372,807]],[[103,609]]])

upperright_tocenter_area = np.array([[[271,365]],[[390,490]],[[744,0]],[[590,0]]])
upperright_backcenter_area = np.array([[[390,490]],[[544,551]],[[864,0]],[[744,0]]])

lowerright_tocenter_area = np.array([[[544,551]],[[440, 695]],[[770,959]],[[959,907]]])
lowerright_backcenter_area = np.array([[[440, 695]],[[770,959]],[[639,959]],[[372,807]]])

upperleft_backcenter_area = np.array([[[271,365]],[[204,478]],[[0,320]],[[0,197]]])
upperleft_tocenter_area = np.array([[[103,609]],[[204,478]],[[0,320]],[[0,456]]])

lowerleft_backcenter_area = np.array([[[103,609]],[[225,685]],[[38,959]],[[0,859]]])
lowerleft_tocenter_area = np.array([[[372,807]],[[225,685]],[[38,959]],[[219,959]]])

old_bg_img = cv2.imread('./data/background/bg_history.png')
mask_flow = cv2.imread('./data/masks/mask_flow.png', flags = cv2.IMREAD_GRAYSCALE)

# All
mask_all = cv2.imread('./data/masks/mask.png', flags = cv2.IMREAD_GRAYSCALE)
_, mask_center = cv2.threshold(mask_all, 10, 255, cv2.THRESH_BINARY)

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


######### Coordinate transformation parameters & functions #########
def LatLng_Dec2Rad(decNum):
    if type(decNum) == float:
        NumIntegral = int(decNum)
        NumDecimal = decNum - NumIntegral

        tmp = NumDecimal * 3600
        degree = NumIntegral
        minute = int(tmp // 60)
        second = round(tmp - minute * 60, 2)
        return degree, minute, second
    if type(decNum) == tuple:
        NumIntegral = int(decNum[0])
        NumDecimal = decNum[0] - NumIntegral
        tmp = NumDecimal * 3600
        degree0 = NumIntegral
        minute0 = int(tmp//60)
        second0 = round(tmp - minute0*60,2)

        NumIntegral = int(decNum[1])
        NumDecimal = decNum[1] - NumIntegral
        tmp = NumDecimal * 3600
        degree1 = NumIntegral
        minute1 = int(tmp//60)
        second1= round(tmp - minute1*60,2)
        return degree0, minute0, second0, degree1, minute1, second1

def LatLng_Rad2Dec(d, m, s):
    decNum = d + m / 60.0 + s / 3600.0
    return decNum

def divide_orientation(location):
    if cv2.pointPolygonTest(center_area, location, False) > 0:
        area = "center"
        angle = False
    elif cv2.pointPolygonTest(upperright_tocenter_area, location, False) > 0:
        area = "upperright_tocenter"
        angle = 123.69
    elif cv2.pointPolygonTest(upperright_backcenter_area, location, False) > 0:
        area = "upperright_backcenter"
        angle = 307.51
    elif cv2.pointPolygonTest(lowerright_tocenter_area, location, False) > 0:
        area = "lowerright_tocenter"
        angle = 217.66
    elif cv2.pointPolygonTest(lowerright_backcenter_area, location, False) > 0:
        area = "lowerright_backcenter"
        angle = 46.85
    elif cv2.pointPolygonTest(upperleft_tocenter_area, location, False) > 0:
        area = "upperleft_tocenter"
        angle = 46.85
    elif cv2.pointPolygonTest(upperleft_backcenter_area, location, False) > 0:
        area = "upperleft_backcenter"
        angle = 217.66
    elif cv2.pointPolygonTest(lowerleft_tocenter_area, location, False) > 0:
        area = "lowerleft_tocenter"
        angle = 307.51
    elif cv2.pointPolygonTest(lowerleft_backcenter_area, location, False) > 0:
        area = "lowerleft_backcenter"
        angle = 123.69
    else:
        area = "not on street"
        angle = False
    return area, angle

pixel_dis = math.sqrt((276 - 492) ** 2 + (402 - 560) ** 2)
r = pixel_dis / (LatLng_Rad2Dec(33, 58, 32.98) - LatLng_Rad2Dec(33, 58, 31.89))

def find_Lat_Lon(y, x):
    # start_time = time.time()
    alpha = 90 - 54.3221
    centerx = 480
    centery = 480
    center_lat = LatLng_Rad2Dec(33, 58, 32.11) # camera lat
    center_lon = -LatLng_Rad2Dec(117, 20, 22.77) # camera lon
    alpha = math.radians(alpha)
    lat_dis = abs(-math.tan(alpha) * x - y + centery + math.tan(alpha) * centerx) / math.sqrt(1 + math.tan(alpha) ** 2)
    if lat_dis<1e-13:
        lat_dis_t = 0
    else:
        lat_dis_t = lat_dis
    lon_dis = math.sqrt((x-centerx)**2 + (y-centery)**2 - lat_dis_t**2)

    angle = math.atan2(y-centery, x-centerx) + alpha
    # print(radians(angle))
    if 0 < angle and angle <= math.radians(90):
        lat_dis = -1 * lat_dis
        lon_dis = -1 * lon_dis
        # print("--")
    elif math.radians(90) < angle and angle <= math.radians(180):
        lat_dis = -1 * lat_dis
        # print("-+")
    elif math.radians(-90)<angle and angle<=0:
        lon_dis = -1 * lon_dis
        # print("+-")
    # else:
        # print("++")
    lat = lat_dis / r + center_lat
    # print(LatLng_Dec2Rad(lat))
    if center_lon<= 0:
        lon = 2*math.asin(math.sin(lon_dis/(2*r)) / math.cos(math.radians(lat)) ) + center_lon
    else:
        lon = -2*math.asin(math.sin(lon_dis/(2*r)) / math.cos(math.radians(lat)) ) + center_lon
    return lat, lon # west semisphere

######### GUI Canvas parameters #########
scale_x = 0.6
scale_y = 0.6

######### Communication parameters #########
C_HOST = '169.235.21.87' # ce-cert server
C_M1_PORT = 28778