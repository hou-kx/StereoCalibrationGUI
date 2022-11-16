#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import camera_configs as cc


def open_camera():
    # 打开视频
    ID = 0
    try:
        while 1:
            cap = cv2.VideoCapture(ID)
            ret, frame = cap.read()
            if ret:
                print('camera:', str(ID))
                return cap
            else:
                ID += 1
                pass
            pass
        pass
    except OverflowError as e:
        print('Sorry, I can\'t find the camera!', e)

    pass


def main():
    # 并排打开两个窗口 left、right
    cv2.namedWindow('left')
    cv2.namedWindow('right')

    cv2.moveWindow('left', 0, 0)
    cv2.moveWindow('right', 640, 0)

    cv2.namedWindow('line')
    cv2.moveWindow('line', 0, 490)

    # 打开视频
    # cap = open_camera()
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    #  读取相机参数
    ccp = cc.CameraConfigsParametersNPZ()
    # 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
    left_map1, left_map2, right_map1, right_map2, Q = cc.getRectifyTransform(ccp)
    print(Q)
    # 进行双目分割
    while True:
        # 读取视频帧数据
        ret, frame = cap.read()
        frame1 = frame[0:480, 0:640]
        frame2 = frame[0:480, 640:1280]

        imgl_rectified, imgr_rectified = cc.rectifyImage(frame1, frame2, left_map1, left_map2, right_map1, right_map2)
        # 绘制等间距平行线，检查立体校正的效果
        draw_line = cc.draw_line(imgl_rectified, imgr_rectified)
        # cv2.imwrite('/data/检验.png', draw_line)
        # 显示窗口
        cv2.imshow('left', frame1)
        cv2.imshow('right', frame2)
        cv2.imshow('line', draw_line)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        pass

    cap.release()
    cv2.destroyAllWindows()
    pass


if __name__ == '__main__':
    main()

    # import os
    # print(os.getcwd())  # 获得当前目录

    pass

