# @
# -*- encoding:utf-8 -*-
# filename: camera_configs.py
# 我想走的路不怎么顺畅
#

import cv2
import numpy as np

# 左相机内参矩阵IntrinsicMatrix
# 我们标定出来的参数对应[fx,0,0;0,fy,0;cx,cy,1]
# 但这里我们需要的参数格式为[fx,0,cx;0,fy,cy;0,0,1]
left_camera_matrix = np.array([[928.0890, 0., 331.5813],
                               [0., 925.5305, 250.3412],
                               [0., 0., 1.]])
# 对应Matlab所得左i相机畸变参数
# RadialDistortion对应k1，k2，k3设置为0了,Tangentialistortion对应p1，p2
# OpenCV中的畸变系数的排列（k1，k2，p1，p2，k3）
left_distortion = np.array([[-0.4481, 0.3621, -2.4279e-04, -0.0041, -0.4656]])

right_camera_matrix = np.array([[924.8006, 0., 312.0632],
                                [0., 921.4080, 240.0499],
                                [0., 0., 1.]])
right_distortion = np.array([[-0.4456, 0.3723, -0.0013, -0.0029, -0.5337]])
# 如果你得到的是3*1向量可用如下代码转换为3*3距阵
# R = cv2.Rodrigues(om)[0]
# 平移关系向量3*1
# om = np.array([0.0011, -0.0102, 0.0076])  # 旋转关系向量
# R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
# 旋转关系向量
R = np.array([[1.0000, 0.0007, 0.0012],
              [-0.0006, 1.0000, -0.0035],
              [-0.0012, 0.0035, 1.0000]]).T
T = np.array([-60.0418, -0.4772, 3.2847])  # 平移关系向量

size = (640, 480)  # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
