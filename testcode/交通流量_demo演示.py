# coding=utf-8
# 导入cv模块
from __future__ import division
import cv2
import sys
import os
import numpy as np
import random
import cx_Oracle
import math
import time
import datetime
import uuid

reload(sys)
sys.setdefaultencoding('utf-8')


# uuid.uuid3(uuid.NAMESPACE_DNS, 'python.org')

def db_connect():
    dsn = cx_Oracle.makedsn('10.10.2.138', 56789, 'SmartRoad')
    conn = cx_Oracle.connect('admin', 'Smart321', dsn)
    cursor = conn.cursor()
    return conn, cursor


def closeMysql(conn, cursor):
    cursor.close()
    conn.close()


def Get_YY_MM_DD_HH(strftime):
    date_list = strftime.split(" ")[0].split("-")
    YY = date_list[0]
    MM = date_list[1]
    DD = date_list[2]
    HH = strftime.split(" ")[1].split(":")[0]
    return YY, MM, DD, HH


def Period_Cal(HH, Min):
    return math.floor(int(HH) * 12 + math.ceil((Min + 0.001) / 5))


def which_lane(x, area_cordinate):
    if x >= area_cordinate[1][1]:
        if x >= area_cordinate[2][1]:
            if x >= area_cordinate[3][1]:
                return 4
            else:
                return 3
        else:
            return 2
    else:
        return 1


class detect_window(object):
    def __init__(self, x1=0, y1=0, x2=0, y2=0, status=0, conti=0, fir_touch=0, sec_touch=0, lane_number=0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.status = status
        self.conti = conti
        self.fir_touch = fir_touch
        self.sec_touch = sec_touch
        self.lane_number = lane_number


def generate_active_window(detect_area, active_window):
    for value in active_window:
        detect_area[value.x1:value.x2] = 0
    return detect_area


class Sysu_bg_subtraction(object):
    def __init__(self, capture):
        self.capture = capture

    def get_bimg(self, frame, width, height):
        img_list = np.ndarray([width * height, frame], dtype=int)
        try:
            img_list.astype("uint8")
        except:
            pass
        frame_count = 1
        process_flag = 0
        ten_per = int(frame / 10)
        while frame_count <= frame:
            flag, img = self.capture.read()
            # flag,img = capture.read()
            if flag:
                img = cv2.resize(img, (width, height))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # img_list[:, (frame_count - 1)] = bg_generate(img, width, height)
                img_list[:, frame_count - 1] = self.__bg_generate(img, width, height)
            else:
                break
            frame_count += 1
            process_flag += 1
            if process_flag == ten_per:
                print ("processing %s%%" % (int(frame_count * 100 / frame)))
                process_flag = 0

        if frame_count == frame + 1:
            bg_img = img_list.mean(axis=1)
            bg_img = bg_img.reshape([height, width])
            bg_img = bg_img.astype("uint8")
        return bg_img

    def img_update(self, img, b_img, learning_rate, fore_point):
        # 学习率计算
        lr = 1 / (learning_rate)
        where_are_inf = np.isinf(lr)
        lr[where_are_inf] = 0

        # 只更新背景点#
        # where_are_255 = np.where(fore_point ==255)
        # lr[where_are_255] = 0.004
        # 背景模型 (1/lr)*img+(1-(1/lr))*b_img
        next_bg_img = lr * img + (1 - lr) * b_img
        return next_bg_img.astype("uint8")

    def __bg_generate(self, img, width, height):
        img = sobel(img)
        img = img.reshape([width * height])
        return img


def sobel(img, gray_flag=0):
    if gray_flag == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst


def array_blur(area, step=2):
    sort_list = []
    for index in xrange(len(area) - step):
        for step_count in xrange(step + 1):
            sort_list.append(area[index + step_count])
        area[index] = max(sort_list)
        sort_list = []
    return area


def get_area_length(roi):
    roi_area = roi.sum(axis=0)
    return np.count_nonzero(roi_area)


def get_up_status(roi):
    pass


def get_down_status(roi):
    roi_area = roi.sum(axis=1)
    return roi_area
    # return np.count_nonzero(roi_area)


def time_test():
    dt = datetime.datetime(2018, 5, 16, 0, 0, 0, 301000)
    YY = dt.year
    MM = dt.month
    DD = dt.day
    HH = dt.hour
    MIN = dt.minute

    period = Period_Cal(HH, MIN)

    uuid_time = datetime.datetime(YY, MM, DD, HH, MIN)
    date_time = datetime.datetime(YY, MM, DD)
    time = datetime.datetime.strftime(date_time, "%Y%m%d")
    uuid_str = datetime.datetime.strftime(uuid_time, "%Y%m%d%H%M")


def time_para_get():
    dt = datetime.datetime.now()
    YY = dt.year
    MM = dt.month
    DD = dt.day
    HH = dt.hour
    MIN = dt.minute

    period = Period_Cal(HH, MIN)

    uuid_time = datetime.datetime(YY, MM, DD, HH, MIN)
    date_time = datetime.datetime(YY, MM, DD)
    time = datetime.datetime.strftime(date_time, "%Y%m%d")
    uuid_str = datetime.datetime.strftime(uuid_time, "%Y%m%d%H%M")

    return period, time, uuid_str


def period_change():
    dt = datetime.datetime.now()
    dt = dt - datetime.timedelta(days=1)
    YY = dt.year
    MM = dt.month
    DD = dt.day
    HH = dt.hour
    MIN = dt.minute
    date_time = datetime.datetime(YY, MM, DD)
    time = datetime.datetime.strftime(date_time, "%Y%m%d")
    return time



    # 参数设置


area_status = {1: 0, 2: 0, 3: 0, 4: 0}

vehicle_count = {1: 0, 2: 0, 3: 0, 4: 0}
small_vehicle_count = {1: 0, 2: 0, 3: 0, 4: 0}
big_vehicle_count = {1: 0, 2: 0, 3: 0, 4: 0}
speed_count = {1: 0, 2: 0, 3: 0, 4: 0}

# area_cordinate = {1:[0,55],2:[55,105],3:[105,140],4:[140,165]}
area_cordinate = {1: [0, 55], 2: [55, 90], 3: [90, 125], 4: [125, 170]}
lane_scale = {1: 0.063, 2: 0.077, 3: 0.1, 4: 0.14}

area_up_y = 107
# area_up_y = 100
area_down_y = 113
area_x_left = 0
area_x_right = 170

pixel_thresh = 0
conti_thresh = 25
right_lane_thresh = 10
# conti_thresh = 20

resize_width = 200
resize_height = 113
frame_count = 0
history = 300
image_no = 0
active_window = []
speed_active_window = []
clear_list = []

image_list = []
max_pixel = []

former_period = -1
former_second = -1
time_count = 0
img_count = 0

output_width = 680
output_height = 440

output_x_ratio = float(output_width) / resize_width
output_y_ratio = float(output_height) / resize_height

if __name__ == '__main__':
    # video_file = "rtsp://admin:sutpc654321@10.10.150.100:554/h264/ch1/sub/av_stream"
    #video_file = "rtsp://admin:sutpc654321@10.10.150.100:554/h264/ch1/sub/av_stream"
    #video_file = u"F:\\工作\\2017年下半年工作\\智慧路灯图像识别\\程序\\flow_calculate\\dataset\\2.mp4"
    video_file = u"F:\\工作\\2017年下半年工作\\智慧路灯图像识别\\程序\\flow_calculate\\dataset\\7-31roadvideo.mp4"
    capture = cv2.VideoCapture(video_file)
    bg_model = Sysu_bg_subtraction(capture)
    bg_img = bg_model.get_bimg(history, resize_width, resize_height)

    #conn, cursor = db_connect()
    while True:
        dt = datetime.datetime.now()
        second = dt.second
        min = dt.minute

        period, db_time, uuid_time = time_para_get()
        flag, frame = capture.read()
        frame_count += 1
        print(frame_count)
        if frame_count <=17570:
            if frame_count == 17570:
                bg_model = Sysu_bg_subtraction(capture)
                bg_img = bg_model.get_bimg(history, resize_width, resize_height)
        else:
            # 前景相减
            output_frame = frame[100:,:]

            output_frame = cv2.resize(output_frame, (output_width, output_height))

            process_frame = cv2.resize(frame, (resize_width, resize_height))
            sobel_fore = sobel(process_frame, 1)
            sobel_dif = cv2.absdiff(sobel_fore, bg_img)
            # th_dif = cv2.threshold(sobel_dif,80,255,cv2.THRESH_BINARY)[1]
            th_para = cv2.threshold(sobel_dif, 0, 255, cv2.THRESH_OTSU)
            if th_para[0] < 50:
                th_dif = cv2.threshold(sobel_dif, 50, 255, cv2.THRESH_BINARY)[1]
            else:
                th_dif = th_para[1]
            # print th_para[0]
            # 背景更新
            bg_img = bg_model.img_update(sobel_fore, bg_img, sobel_dif, th_dif)

            # cv2.line(process_frame, (area_x_left, area_up_y), (area_x_right, area_up_y), (0, 0, 255), 2)
            # cv2.line(process_frame, (area_x_left, area_down_y), (area_x_right, area_down_y), (0, 0, 255), 2)

            cv2.line(output_frame, (int(output_x_ratio * area_x_left), int(output_y_ratio * area_up_y)),
                     (int(output_x_ratio * area_x_right),
                      int(output_y_ratio * area_up_y)), (0, 0, 255), 2)

            cv2.line(output_frame, (int(output_x_ratio * area_x_left), int(output_y_ratio * area_down_y)),
                     (int(output_x_ratio * area_x_right),
                      int(output_y_ratio * area_down_y)), (0, 0, 255), 2)
            # cv2.line(process_frame, (speed_area_x_left, speed_area_up_y), (speed_area_x_right, speed_area_up_y), (0, 255, 255), 2)
            # cv2.line(process_frame, (speed_area_x_left, speed_area_down_y), (speed_area_x_right, speed_area_down_y), (0, 255, 255), 2)

            # 动态开窗

            detection_ori_area = th_dif[area_up_y:area_down_y, area_x_left:area_x_right]
            detection_ori_area = detection_ori_area.sum(axis=0)
            detection_area = detection_ori_area
            detection_area = array_blur(detection_area, 3)
            # detection_area = array_blur(detection_area, 8)
            detection_area = generate_active_window(detection_area, active_window)

            # 车辆检测
            count_flag = 0
            zero_count = 0
            for index, value in enumerate(detection_area):
                if value > pixel_thresh:
                    if count_flag == 0:
                        if (index - 5) >= 0:
                            detect_win = detect_window(x1=index - 5, y1=area_up_y, x2=0, y2=area_down_y)
                        else:
                            detect_win = detect_window(x1=0, y1=area_up_y, x2=0, y2=area_down_y)
                    elif count_flag >= right_lane_thresh:
                        if index == len(detection_area) - 1:
                            detect_win.x2 = index
                            # if generate_active_window(detect_win, active_window) == 0:
                            detect_win.fir_touch = frame_count
                            active_window.append(detect_win)
                            max_pixel.append(0)
                            # image_list.append(process_frame)
                    count_flag += 1
                else:
                    if count_flag >= conti_thresh:
                        if (index + 20) >= 170:
                            detect_win.x2 = 170
                        else:
                            detect_win.x2 = index + 20
                        # if generate_active_window(detect_win, active_window) == 0:
                        detect_win.fir_touch = frame_count
                        active_window.append(detect_win)
                        max_pixel.append(0)
                        # image_list.append(process_frame)
                    count_flag = 0

            # print len(active_window)

            # 监测动态窗口
            for win_index, value in enumerate(active_window):
                roi_area = th_dif[area_up_y:area_down_y, value.x1:value.x2]

                # cv2.line(process_frame, (value.x1, area_up_y), (value.x1, area_down_y), (0, 0, 255), 2)
                # cv2.line(process_frame, (value.x2, area_up_y), (value.x2, area_down_y), (0, 0, 255), 2)

                cv2.line(output_frame, (int(value.x1 * output_x_ratio), int(area_up_y * output_y_ratio)),
                         (int(value.x1 * output_x_ratio)
                          , int(area_down_y * output_y_ratio)), (0, 255, 0), 2)
                cv2.line(output_frame, (int(value.x2 * output_x_ratio), int(area_up_y * output_y_ratio)),
                         (int(value.x2 * output_x_ratio)
                          , int(area_down_y * output_y_ratio)), (0, 255, 0), 2)

                pixel_count = np.count_nonzero(roi_area)
                area_length = get_area_length(roi_area)
                if pixel_count >= 18:
                    cv2.line(output_frame, (int(value.x1 * output_x_ratio), int(area_up_y * output_y_ratio)),
                             (int(value.x1 * output_x_ratio)
                              , int(area_down_y * output_y_ratio)), (0, 255, 0), 2)
                    cv2.line(output_frame, (int(value.x2 * output_x_ratio), int(area_up_y * output_y_ratio)),
                             (int(value.x2 * output_x_ratio)
                              , int(area_down_y * output_y_ratio)), (0, 255, 0), 2)

                    # value.duaration +=1
                    if value.status == 0:
                        # if value.x2 == 170:
                        #    value.lane_number = which_lane(value.x1 + 10, area_cordinate)
                        # else:
                        #    value.lane_number = which_lane((value.x1 + value.x2 - 25) / 2, area_cordinate)
                        value.status = 1
                        value.first_touch = frame_count
                    else:
                        if max_pixel[win_index] < area_length:
                            max_pixel[win_index] = area_length
                else:
                    # 计算速度
                    if value.lane_number == 0:
                        if value.x2 == 170:
                            value.lane_number = which_lane(value.x1 + 10, area_cordinate)
                        else:
                            value.lane_number = which_lane((value.x1 + value.x2 - 25) / 2, area_cordinate)

                    vehicle_count[value.lane_number] += 1

                    if max_pixel[win_index] >= 38:
                        big_vehicle_count[value.lane_number] += 1
                    else:
                        small_vehicle_count[value.lane_number] += 1

                    # 速度计算#########
                    value.sec_touch = frame_count
                    run_time = (value.sec_touch - value.fir_touch) * 0.04
                    run_distance = random.uniform(4, 5) + lane_scale[value.lane_number] * 6
                    try:
                        run_speed = (run_distance / run_time) * 3.6
                    except:
                        run_speed = 0
                    if run_speed < 0 or run_speed > 110:
                        pass
                    else:
                        speed_count[value.lane_number] += run_speed
                    # print run_speed
                    clear_list.append(win_index)
            # print len(clear_list)

            # 删除监测窗口
            if clear_list != []:
                clear_list.sort(reverse=True)
            for clear_index in clear_list:
                # del list[clear_index]
                # print clear_list
                del active_window[clear_index]
                del max_pixel[clear_index]
                # del image_list[clear_index]
            clear_list = []

            # print "小车："+str(vehicle_type_count["小车"])
            # print "大车："+str(vehicle_type_count["大车"])


            cv2.putText(output_frame, str(vehicle_count[1]), (int((area_cordinate[1][0] + 20)*output_x_ratio), int((area_down_y - 20)*output_y_ratio)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1)
            cv2.putText(output_frame, str(vehicle_count[2]), (
            int((area_cordinate[2][0] + 5) * output_x_ratio), int((area_down_y - 20) * output_y_ratio)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1)

            cv2.putText(output_frame, str(vehicle_count[3]), (
                int((area_cordinate[3][0] + 5) * output_x_ratio), int((area_down_y - 20) * output_y_ratio)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1)

            cv2.putText(output_frame, str(vehicle_count[4]), (
                int((area_cordinate[4][0] + 5) * output_x_ratio), int((area_down_y - 20) * output_y_ratio)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1)


            #cv2.putText(process_frame, str(vehicle_count[1]), (area_cordinate[1][0] + 20, area_down_y - 20),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            #cv2.putText(process_frame, str(vehicle_count[2]), (area_cordinate[2][0] + 5, area_down_y - 20),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            #cv2.putText(process_frame, str(vehicle_count[3]), (area_cordinate[3][0] + 5, area_down_y - 20),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            #cv2.putText(process_frame, str(vehicle_count[4]), (area_cordinate[4][0] + 5, area_down_y - 20),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # process_frame = cv2.resize(process_frame, (200*3, 113*3),cv2.INTER_CUBIC)

            cv2.imwrite(
                "image\\" + str(int(period)) + "_" + str(min) + "_" + str(second) + "_" + str(img_count) + ".jpg",
                output_frame)
            #cv2.imwrite("image\\"+str(int(period))+"_"+str(img_count)+".jpg",output_frame)
            img_count+=1

            cv2.imshow("2", output_frame)
            #cv2.imshow("1", th_dif)
            #cv2.imshow("3", bg_img)
            #cv2.imshow("4", sobel_fore)
            #cv2.imshow("5", sobel_dif)


            c = cv2.waitKey(5)
            # if frame_count == 10477:
            #    cv2.waitKey(0)


            if c == 27:
                print (detection_area)
                # detection_area_plot(detection_ori_area)
                cv2.waitKey(0)
                # cv2.destroyAllWindows()

        #except Exception as e:
        #    video_file = "rtsp://admin:sutpc654321@10.10.150.100:554/h264/ch1/sub/av_stream"
        #    capture = cv2.VideoCapture(video_file)
        #    print e



