# class_surround_statistic -> [3,19,19] 分别是外轮廓上，外轮廓下，整个轮廓的类别分布
import numpy as np
from cv2 import cv2
import os
from collections import Counter
import collections
record_files = collections.defaultdict(list)
if not os.path.exists('statistic_citiscapes_results'):
    os.mkdir('statistic_citiscapes_results')
for k in range(3):
    type = 'up' if k == 0 else 'down' if k == 1 else 'all'
    for i in range(19):
        mfile = open("{}/record_surround_class_{}_{}.txt".format('statistic_citiscapes_results',type,i),"w+") 
        record_files[type].append(mfile)
temp = np.zeros(19)
def getStatistic(class_num,pred,class_position_statistic,class_surround_statistic):
    for class_id in range(class_num):
        class_position_statistic[class_id,pred == class_id] += 1
        mask = np.zeros_like(pred).astype(np.uint8)
        seg_result = pred
        seg_result_plus_1 = seg_result + 1 
        d = np.unique(seg_result)
        position = seg_result==class_id #pole
        #if use seg_result class_id == 0 will get wrong output
        pole = seg_result_plus_1*position
        pole = pole.astype(np.uint8)

        #OpenCV定义的结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10, 10))
            
        # #膨胀图像
        dilated = cv2.dilate(pole,kernel)

        _,binary_img = cv2.threshold(dilated,0.9,255,cv2.THRESH_BINARY)
        # img3 = img2 * binary_img
        # cv2.imwrite("Eroded_Image.jpg",binary_img);

        contours,_ = cv2.findContours(binary_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            mask_clone = mask.copy()
            cv2.drawContours(mask_clone,contours,i,1,3)
            m = mask_clone == 1
            pos = np.where(mask_clone == 1)
            mean_y,mean_x = np.mean(pos,axis=1)
            down_mask = m.copy()
            down_mask[:int(mean_y),:]=False
            up_mask = m.copy()
            up_mask[int(mean_y):,:]=False
            u = Counter(pred[up_mask])
            u_dict = dict(u)
            temp.fill(0)
            for k,value in u_dict.items():   #遍历打印obj所有元素
                if(k!=255):
                    class_surround_statistic[0,class_id,k] += value
                    temp[k] += value
            sss = ' '.join(str(i) for i in temp)
            record_files['up'][class_id].write(sss+'\n')

            d = Counter(pred[down_mask])
            d_dict = dict(d)
            temp.fill(0)
            for k,value in d_dict.items():   #遍历打印obj所有元素
                if(k!=255):
                    class_surround_statistic[1,class_id,k] += value
                    temp[k] += value
            record_files['down'][class_id].write(' '.join(str(i) for i in temp)+'\n')

            d = Counter(pred[m])
            d_dict = dict(d)
            temp.fill(0)
            for k,value in d_dict.items():   #遍历打印obj所有元素
                if(k!=255):
                    class_surround_statistic[2,class_id,k] += value
                    temp[k] += value
            record_files['all'][class_id].write(' '.join(str(i) for i in temp)+'\n')
            # u = Counter(pred[m])
            # mask[m] = 255
            # cv2.imwrite('test.png',mask)
            # print('dd')
    return 0