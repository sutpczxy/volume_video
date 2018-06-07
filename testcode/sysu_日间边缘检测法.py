# coding=utf-8
# 导入cv模块
from __future__ import division
import cv2
import sys
import os
import numpy as np
#import libbgs
#画图
#import matplotlib.pyplot as plt
#import matplotlib
#from matplotlib.font_manager import FontProperties
#import seaborn as sns
#font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)
#plt.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['axes.unicode_minus'] = False


# reload(sys)
# sys.setdefaultencoding('utf-8')


#def detection_area_plot(det_area):
#    fig = plt.figure()
#    X = np.arange(0,det_area.shape[0])
#    plt.bar(X, det_area, 0.4, color="green")
#    plt.xlabel("X-axis")
#    plt.ylabel("Y-axis")
#    plt.title("bar chart")
#    plt.show()


def which_lane(x,area_cordinate):
    if x>=area_cordinate[1][1]:
        if x>=area_cordinate[2][1]:
            if x>=area_cordinate[3][1]:
                return 4
            else:
                return 3
        else:
            return 2
    else:
        return 1


class detect_window(object):
    def __init__(self,x1=0,y1=0,x2=0,y2=0,status=0,conti =0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.status = status
        self.conti = conti


def generate_active_window(detect_area,active_window):
    for value in active_window:
        detect_area[value.x1:value.x2] = 0
    return detect_area

class Sysu_bg_subtraction(object):
    def __init__(self,capture):
        self.capture = capture

    def get_bimg(self,frame,width,height):
        img_list = np.ndarray([width * height, frame], dtype=int)
        try:
            img_list.astype("uint8")
        except:
            pass
        frame_count = 1
        process_flag = 0
        ten_per = int(frame/10)
        while frame_count<=frame:
            flag, img = self.capture.read()
            #flag,img = capture.read()
            if flag:
                img = cv2.resize(img,(width,height))
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                #img_list[:, (frame_count - 1)] = bg_generate(img, width, height)
                img_list[:,frame_count-1] = self.__bg_generate(img,width,height)
            else:
                break
            frame_count += 1
            process_flag +=1
            if process_flag == ten_per:
                print ("processing %s%%" %(int (frame_count*100/frame)))
                process_flag =0

        if frame_count == frame+1:
            bg_img = img_list.mean(axis=1)
            bg_img = bg_img.reshape([height,width])
            bg_img = bg_img.astype("uint8")
        return bg_img

    def img_update(self,img,b_img,learning_rate,fore_point):
        #学习率计算
        lr = 1/(learning_rate)
        where_are_inf = np.isinf(lr)
        lr[where_are_inf] = 0

        #只更新背景点#
        #where_are_255 = np.where(fore_point ==255)
        #lr[where_are_255] = 0.004
        #背景模型 (1/lr)*img+(1-(1/lr))*b_img
        next_bg_img = lr*img + (1-lr)*b_img
        return next_bg_img.astype("uint8")

    def __bg_generate(self,img,width,height):
        img = sobel(img)
        img = img.reshape([width * height])
        return img


def sobel(img,gray_flag=0):
    if gray_flag == 1:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst

def array_blur(area,step = 2):
    sort_list = []
    for index in range(len(area)-step):
        for step_count in range(step+1):
            sort_list.append(area[index+step_count])
        area[index] = max(sort_list)
        sort_list = []
    return area

def get_area_length(roi):
    roi_area = roi.sum(axis=0)
    return np.count_nonzero(roi_area)


#参数设置
area_status = {1:0,2:0,3:0,4:0}
vehicle_count = {1:0,2:0,3:0,4:0}
vehicle_type_count = {"小车":0,"大车":0}

#area_cordinate = {1:[0,55],2:[55,105],3:[105,140],4:[140,165]}
area_cordinate = {1:[0,55],2:[55,90],3:[90,125],4:[125,170]}

area_up_y = 107
#area_up_y = 100
area_down_y = 113
area_x_left = 0
area_x_right = 170

pixel_thresh = 0
conti_thresh = 25
right_lane_thresh = 10
#conti_thresh = 20

resize_width = 200
resize_height = 113
frame_count = 0
history = 300
image_no =0
active_window = []
clear_list = []

image_list = []
max_pixel = []


## BGS Library algorithms
#bgs = libbgs.SuBSENSE()
#video_file = "rtsp://admin:sutpc654321@10.10.150.100:554/h264/ch1/sub/av_stream"
#video_file = "rtsp://admin:sutpc654321@10.10.150.100:554/h264/ch1/sub/av_stream"
#video_file = "C:\\Users\\lenovo\\Desktop\\10.10.150.100_01_20171211111225497.mp4"
video_file = "7-31roadvideo.mp4"
#video_file = "dataset\\night\\10.10.5.199_01_20171130180002897_1.mp4"
# video_file = u"C:\\Users\\lenovo\\Desktop\\智慧路灯测试数据集\\1223_daybreak_3.mp4"
#video_file = u"C:\\Users\\lenovo\\Desktop\\智慧路灯测试数据集\\1225_moring_10.mp4"
#video_file = u"F:\\工作\\2017年下半年工作\\智慧路灯图像识别\\程序\\flow_calculate\\智慧路灯测试数据集\\1222_afternoon_15.mp4"
#video_file = u"F:\\工作\\2017年下半年工作\\智慧路灯图像识别\\程序\\flow_calculate\\智慧路灯测试数据集\\1222_night_20.mp4"
#video_file = "rtsp://admin:sutpc654321@10.10.150.100:554/h264/ch1/sub/av_stream"
# video_file = "10.10.5.199_01_20171130180002897_3.mp4"
capture = cv2.VideoCapture(video_file)
bg_model = Sysu_bg_subtraction(capture)
bg_img = bg_model.get_bimg(history,resize_width,resize_height)
#cv2.imshow("1",bg_img)
#cv2.waitKey(0)
try:
    while True:
    #while frame_count<=15000:
        flag, frame = capture.read()
        frame_count+= 1
        # print (frame_count)
        #前景相减
    
        process_frame = cv2.resize(frame, (resize_width, resize_height))
        sobel_fore = sobel(process_frame,1)
        sobel_dif = cv2.absdiff(sobel_fore,bg_img)
        #th_dif = cv2.threshold(sobel_dif,80,255,cv2.THRESH_BINARY)[1]
        th_para = cv2.threshold(sobel_dif, 0, 255, cv2.THRESH_OTSU)
        if th_para[0]<50:
            th_dif = cv2.threshold(sobel_dif,50,255,cv2.THRESH_BINARY)[1]
        else:
            th_dif = th_para[1]
        #print th_para[0]
        #背景更新
        bg_img = bg_model.img_update(sobel_fore,bg_img,sobel_dif,th_dif)

        cv2.line(process_frame, (area_x_left, area_up_y), (area_x_right, area_up_y), (0, 0, 255), 2)
        cv2.line(process_frame, (area_x_left, area_down_y), (area_x_right, area_down_y), (0, 0, 255), 2)

        #动态开窗
        detection_ori_area = th_dif[area_up_y:area_down_y, area_x_left:area_x_right]
        
        detection_ori_area = detection_ori_area.sum(axis=0)
        detection_area = detection_ori_area
        detection_area = array_blur(detection_area, 3)
        #detection_area = array_blur(detection_area, 8)
        detection_area = generate_active_window(detection_area,active_window)

        count_flag = 0
        zero_count = 0
        for index, value in enumerate(detection_area):
            if value > pixel_thresh:
                if count_flag == 0:
                    if (index-5)>=0:
                        detect_win = detect_window(x1=index-5, y1=area_up_y, x2=0, y2=area_down_y)
                        # cv2.rectangle(process_frame,(detect_win.x1,detect_win.y1),(detect_win.x2,detect_win.y2),(55,255,155),2)
                    else:
                        detect_win = detect_window(x1=0, y1=area_up_y, x2=0, y2=area_down_y)
                    
                elif count_flag >= right_lane_thresh:
                    if index == len(detection_area) - 1:
                        detect_win.x2 = index
                        # if generate_active_window(detect_win, active_window) == 0:
                        active_window.append(detect_win)
                        max_pixel.append(0)
                        image_list.append(process_frame)
                count_flag += 1
            else:
                if count_flag >= conti_thresh:
                    if (index+20)>=170:
                        detect_win.x2 = 170
                    else:
                        detect_win.x2 = index+20
                    # if generate_active_window(detect_win, active_window) == 0:
                    active_window.append(detect_win)
                    max_pixel.append(0)
                    image_list.append(process_frame)
                count_flag = 0

        #print len(active_window)

        # 监测动态窗口
        for win_index, value in enumerate(active_window):
            roi_area = th_dif[area_up_y:area_down_y, value.x1:value.x2]
            # cv2.line(process_frame, (value.x1, area_up_y), (value.x1, area_down_y), (0, 0, 255), 2)
            # cv2.line(process_frame, (value.x2, area_up_y), (value.x2, area_down_y), (0, 0, 255), 2)
            pixel_count = np.count_nonzero(roi_area)
            area_length = get_area_length(roi_area)
            if pixel_count >= 18:
                cv2.line(process_frame, (value.x1, area_up_y), (value.x1, area_down_y), (0, 255, 0), 2)
                cv2.line(process_frame, (value.x2, area_up_y), (value.x2, area_down_y), (0, 255, 0), 2)
                if value.status == 0:
                    if value.x2 == 170:
                        lane_number = which_lane(value.x1+10, area_cordinate)
                    else:
                        lane_number = which_lane((value.x1+value.x2-25)/2, area_cordinate)
                    vehicle_count[lane_number] += 1
                    value.status = 1
                else:
                    #pass
                    if max_pixel[win_index]<area_length:
                        max_pixel[win_index] = area_length
            else:
                if max_pixel[win_index]>=38:
                    vehicle_type_count["大车"]+=1
                    #cv2.imwrite("C:\\Users\\lenovo\\Desktop\\flow_calculate\\image\\pixel_count\\truck\\"+str(max_pixel[win_index])+"_"+str(frame_count)+".jpg",image_list[win_index])
                    #print "C:\\Users\\lenovo\\Desktop\\flow_calculate\\image\\pixel_count\\truck\\"+str(max_pixel[win_index])+"_"+str(frame_count)+".jpg"
                    #cv2.waitKey(0)
                    #print "小车：" + str(vehicle_type_count["小车"])+"  大车：" + str(vehicle_type_count["大车"])
                    #print
                else:
                    vehicle_type_count["小车"]+=1
                    #print "小车：" + str(vehicle_type_count["小车"]) + "  大车：" + str(vehicle_type_count["大车"])
                    #print "C:\\Users\\lenovo\\Desktop\\flow_calculate\\image\\pixel_count\\vehicle\\"+str(max_pixel[win_index])+"_"+str(frame_count)+".jpg"
                    #cv2.imwrite("C:\\Users\\lenovo\\Desktop\\flow_calculate\\image\\pixel_count\\vehicle\\"+str(max_pixel[win_index])+"_"+str(frame_count)+".jpg",image_list[win_index])
                clear_list.append(win_index)
        #print len(clear_list)

        # 删除监测窗口
        if clear_list != []:
            clear_list.sort(reverse=True)
        for clear_index in clear_list:
            # del list[clear_index]
            # print clear_list
            del active_window[clear_index]
            del max_pixel[clear_index]
            del image_list[clear_index]
        clear_list = []

        # print("car ："+str(vehicle_type_count["小车"])) 
        # print("bus ："+str(vehicle_type_count["大车"])) 

        cv2.putText(process_frame, str(vehicle_count[1]), (area_cordinate[1][0] + 20, area_down_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(process_frame, str(vehicle_count[2]), (area_cordinate[2][0] + 5, area_down_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(process_frame, str(vehicle_count[3]), (area_cordinate[3][0] + 5, area_down_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(process_frame, str(vehicle_count[4]), (area_cordinate[4][0] + 5, area_down_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(process_frame, "car:"+str(vehicle_type_count["小车"]), (area_cordinate[3][0] + 5, area_down_y - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(process_frame, "bus:"+str(vehicle_type_count["大车"]), (area_cordinate[3][0] + 5, area_down_y - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)            
        #process_frame = cv2.resize(process_frame, (200*3, 113*3),cv2.INTER_CUBIC)


        cv2.imshow("2",process_frame)

        cv2.imshow("1",th_dif)

        cv2.imshow("3",bg_img)

        cv2.imshow("4",sobel_fore)

        cv2.imshow("5",sobel_dif)


        c = cv2.waitKey(5)
        #if frame_count == 10477:
        #    cv2.waitKey(0)


        if c == 27:
            print (detection_area)
            #detection_area_plot(detection_ori_area)
            cv2.waitKey(0)
        #cv2.destroyAllWindows()
except Exception as e:
    print (e)

cv2.waitKey(0)
###############################################静态开窗###############################
'''
        for index in range(4):
            # lane = dif[area_up_y:area_down_y, area_cordinate[index + 1][0]:area_cordinate[index + 1][1]]
            lane = th_dif[area_up_y:area_down_y, area_cordinate[index + 1][0]:area_cordinate[index + 1][1]].sum(
                axis=0)
            if (float(np.count_nonzero(lane)) / lane.size) > 0.4:
                if index != 0:
                    if area_status[index + 1] == 0 and area_status[index] == 0:
                        vehicle_count[index + 1] += 1
                        area_status[index + 1] = 1
                        cv2.line(process_frame, (area_cordinate[index + 1][0], area_up_y),
                                 (area_cordinate[index + 1][1], area_up_y), (0, 255, 0), 2)
                        cv2.line(process_frame, (area_cordinate[index + 1][0], area_up_y),
                                 (area_cordinate[index + 1][0], area_down_y), (0, 255, 0), 2)
                        cv2.line(process_frame, (area_cordinate[index + 1][1], area_up_y),
                                 (area_cordinate[index + 1][1], area_down_y), (0, 255, 0), 2)
                        cv2.line(process_frame, (area_cordinate[index + 1][0], area_down_y),
                                 (area_cordinate[index + 1][1], area_down_y), (0, 255, 0), 2)

                        cv2.putText(process_frame, str(area_status[1]), (area_cordinate[1][0] + 20, area_down_y - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        pass
                else:
                    if area_status[index + 1] == 0:
                        vehicle_count[index + 1] += 1
                        area_status[index + 1] = 1
                        cv2.line(process_frame, (area_cordinate[index + 1][0], area_up_y),
                                 (area_cordinate[index + 1][1], area_up_y), (0, 255, 0), 2)
                        cv2.line(process_frame, (area_cordinate[index + 1][0], area_up_y),
                                 (area_cordinate[index + 1][0], area_down_y), (0, 255, 0), 2)
                        cv2.line(process_frame, (area_cordinate[index + 1][1], area_up_y),
                                 (area_cordinate[index + 1][1], area_down_y), (0, 255, 0), 2)
                        cv2.line(process_frame, (area_cordinate[index + 1][0], area_down_y),
                                 (area_cordinate[index + 1][1], area_down_y), (0, 255, 0), 2)
                    else:
                        pass

            elif (float(np.count_nonzero(lane)) / lane.size) < 0.2:
                area_status[index + 1] = 0



        cv2.putText(process_frame, str(area_status[1]), (area_cordinate[1][0] + 20, area_down_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(process_frame, str(area_status[2]), (area_cordinate[2][0] + 20, area_down_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(process_frame, str(area_status[3]), (area_cordinate[3][0] + 5, area_down_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(process_frame, str(area_status[4]), (area_cordinate[4][0] + 5, area_down_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
'''

