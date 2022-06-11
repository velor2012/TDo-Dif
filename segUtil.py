#coding=utf-8
from cv2 import cv2 
import numpy as np
import numba as nb
from numba import njit, int32,float64,boolean,int64
def getBox(class_value,seg_result,expend=True):
    # img = cv2.imread('0_pred.png',0)
    # img2 = cv2.imread("2.png")
    # a = img==153 #pole
    seg_result_plus_1 = seg_result + 1 
    d = np.unique(seg_result)
    position = seg_result==class_value #pole
    a = np.max(position)
    a = np.min(position)
    #if use seg_result class_value == 0 will get wrong output
    pole = seg_result_plus_1*position
    pole = pole.astype(np.uint8)

    #OpenCV定义的结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15, 15))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(15, 15))


    if(expend):
        
        closed = cv2.morphologyEx(pole, cv2.MORPH_CLOSE, kernel)
        # #膨胀图像
        # dilated = cv2.dilate(pole,kernel)
        # #腐蚀图像
        # eroded = cv2.erode(dilated,kernel2)
        pole = closed
    #显示腐蚀后的图像

    _,binary_img = cv2.threshold(pole,0.9,255,cv2.THRESH_BINARY)
    # img3 = img2 * binary_img
    # cv2.imwrite("Eroded_Image.jpg",binary_img);

    contours,_ = cv2.findContours(binary_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img2,contours,-1,(0,0,255),3)

    # boundrects = [cv2.boundingRect(cnt) for cnt in contours]
    boundrects = [cv2.boundingRect(cnt) for cnt in contours]
    areas = [cv2.contourArea(cnt) for cnt in contours]


    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return np.array(boundrects),np.array(areas),binary_img

def cropBottom(img,percent=0.8,pred_arr=False):
    assert percent <= 1 and percent > 0
    if(pred_arr):
        h = img.shape[1]
        return img[:,:int(h*percent),:]
    h = img.shape[0]
    return img[:int(h*percent),:]

@njit(int64(int64[:], int64[:]))
def getMIOU(box1,box2):

    [x_i,y_i,w_i,h_i] = box1
    [x_j,y_j,w_j,h_j] = box2
    x_i_w = x_i + w_i
    y_i_h = y_i + h_i
    x_j_w = x_j + w_j
    y_j_h = y_j + h_j

    ## 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max(np.array([x_i, x_j],dtype=int64))
    yy1 = np.max(np.array([y_i, y_j],dtype=int64))
    xx2 = np.min(np.array([x_i_w, x_j_w],dtype=int64))
    yy2 = np.min(np.array([y_i_h, y_j_h],dtype=int64))

    inter_area = (np.max(np.array([0, xx2-xx1],dtype=int64))) * (np.max(np.array([0, yy2-yy1],dtype=int64)))
    return inter_area

# [x,y,w,h] = box
def filterBoxPair(box1,box2,origin_w , origin_h):
    [x,y,w,h] = box1
    [x2,y2,w2,h2] = box2
    if(np.abs(x-x2)+np.abs(y-y2) > 1/4 * origin_w):
        return False
    return True

def filterBox(boxes,areas,areaThreshold,allowContain=False,max_inter=0.5):
    # areaThreshold shape (2,) or a float number
    # when (2,0), areaThreshold equals [min_threshold,max_threshold]
    # when a float number, areaThreshold equals min_threshold
    valid_index = np.zeros(len(boxes),dtype=np.int32)
    threshold_is_array = False
    # print(type(areaThreshold))
    # if( type(areaThreshold) is np.ndarray or type(areaThreshold) is tuple or type(areaThreshold) is list):
    #     assert len(areaThreshold) == 2
    #     threshold_is_array = True
    for i in range(len(boxes)):
        area = boxes[i][2]*boxes[i][3]
        if(not threshold_is_array and area>areaThreshold):
            valid_index[i] = valid_index[i]+1
        elif(threshold_is_array and area>areaThreshold[0] and area < areaThreshold[1] ):
            valid_index[i] = valid_index[i]+1

    boxes_copy = boxes[valid_index==1]
    areas_copy = areas[valid_index==1]

    #filter IoU too large boxes
    valid_index_af_fi_area = np.ones(len(boxes_copy),dtype=np.int32)
    for i in range(len(boxes_copy)):
        for j in range(i+1,len(boxes_copy)):
            small_index = i
            big_index = j
            if(boxes_copy[j][2]*boxes_copy[j][3]<boxes_copy[i][2]*boxes_copy[i][3]):
                small_index = j
                big_index = i

            inter_area = getMIOU(boxes_copy[small_index],boxes_copy[big_index])
            # if(areas_copy[small_index])<0.001:
            #     print('f')
            if(inter_area/(boxes_copy[small_index][2]*boxes_copy[small_index][3])>max_inter):
                valid_index_af_fi_area[small_index] = 0
    boxes_copy = boxes_copy[valid_index_af_fi_area==1]
    areas_copy = areas_copy[valid_index_af_fi_area==1]
    other = {"valid_index":valid_index==1,"valid_index_af_fi_area":valid_index_af_fi_area==1}
    return boxes_copy,areas_copy,other

@njit
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
                