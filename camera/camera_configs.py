#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

# 文件所在目录, 这里使用的是相对路径
npz_path_d = os.path.join(os.path.dirname(__file__), r"..\data\binocular")


def find_latest_npz(file_path):
    """
    找到file_path目录下最近保存的npz文件路径，由于其名称是根据
    :param file_path:
    :return:
    """
    import os
    fileList = os.listdir(file_path)
    _file = []

    if not len(fileList):
        print("The current file is empty!")
        # 结束程序
        import sys
        sys.exit(1)

    for file in fileList:
        path = os.path.join(file_path, file)

        if os.path.isfile(path):
            _file.append(file)
            pass
        pass
    _file.sort(reverse=True)
    return os.path.join(file_path, _file[0])


class CameraConfigsParametersNPZ(object):
    """
    相机内参、外参
    """
    def __init__(self):

        global npz_path_d
        # 获得最新配置文件路径
        npz_path_f = find_latest_npz(npz_path_d)
        camear_parameter = np.load(npz_path_f)
        """相机矩阵IntrinsicMatrix"""
        # 焦距（fx, fy）;光心（cx, cy）
        # 我们标定出来的参数对应   [fx, 0, 0;
        #                         0,fy, 0;
        #                        cx,cy, 1]
        # 但这里我们需要的参数格式为[fx, 0,cx;
        #                         0,fy,cy;
        #                         0, 0, 1]
        # 左相机
        self.left_camera_matrix = camear_parameter["left_camera_matrix"]
        """内参"""
        # RadialDistortion对应k1，k2，k3设置为0了,Tangentialistortion对应p1，p2
        # OpenCV中的畸变系数的排列（k1，k2，p1，p2，k3）
        self.left_distortion = camear_parameter["left_distortion"]

        # 右相机
        self.right_camera_matrix = camear_parameter["right_camera_matrix"]
        self.right_distortion = camear_parameter["right_distortion"]

        """外参"""
        # 如果你得到的是3*1向量可用如下代码转换为3*3距阵
        # R = cv2.Rodrigues(om)[0]
        # 平移关系向量3*1
        # om = np.array([0.0011, -0.0102, 0.0076])  # 旋转关系向量
        # R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R

        # 旋转关系向量
        self.R = camear_parameter["R"]
        # 平移关系向量
        self.T = camear_parameter["T"]

        self.size = (640, 480)  # 图像尺寸

        # 焦距
        self.focal_length = 0  # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]

        # 基线距离
        self.baseline = self.T[0]  # 单位：mm， 为平移向量的第一个参数（取绝对值）
        pass
    pass


class CameraConfigsParameters(object):
    """
    相机内参、外参
    """
    def __init__(self):
        """相机矩阵IntrinsicMatrix"""
        # 焦距（fx, fy）;光心（cx, cy）
        # 我们标定出来的参数对应   [fx, 0, 0;
        #                         0,fy, 0;
        #                        cx,cy, 1]
        # 但这里我们需要的参数格式为[fx, 0,cx;
        #                         0,fy,cy;
        #                         0, 0, 1]

        # 左相机
        self.left_camera_matrix = np.array([[928.0890, 0., 331.5813],
                                            [0., 925.5305, 250.3412],
                                            [0., 0., 1.]])
        """内参"""
        # RadialDistortion对应k1，k2，k3设置为0了,Tangentialistortion对应p1，p2
        # OpenCV中的畸变系数的排列（k1，k2，p1，p2，k3）

        # left_distortion = np.array([[-0.4481, 0.3621, -2.4279e-04, -0.0041, -0.4656]])
        self.left_distortion = np.array([[-0.4481, 0.3621, -2.4279e-04, -0.0041, 0]])

        # 右相机
        self.right_camera_matrix = np.array([[924.8006, 0., 312.0632],
                                             [0., 921.4080, 240.0499],
                                             [0., 0., 1.]])
        # right_distortion = np.array([[-0.4456, 0.3723, -0.0013, -0.0029, -0.5337]])
        self.right_distortion = np.array([[-0.4456, 0.3723, -0.0013, -0.0029, 0]])

        """外参"""
        # 如果你得到的是3*1向量可用如下代码转换为3*3距阵
        # R = cv2.Rodrigues(om)[0]
        # 平移关系向量3*1
        # om = np.array([0.0011, -0.0102, 0.0076])  # 旋转关系向量
        # R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R

        # 旋转关系向量
        self.R = np.array([[1.0000, 0.0007, 0.0012],
                           [-0.0006, 1.0000, -0.0035],
                           [-0.0012, 0.0035, 1.0000]]).T
        # 平移关系向量
        self.T = np.array([-60.0418, -0.4772, 3.2847])

        self.size = (640, 480)  # 图像尺寸

        # 焦距
        self.focal_length = 0  # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]

        # 基线距离
        self.baseline = self.T[0]  # 单位：mm， 为平移向量的第一个参数（取绝对值）
        pass
    pass


def preprocessing(image1, image2):
    """
    预处理：先将左右目图像转换为灰度图、直方图均衡
    :param image1:左目
    :param image2:右目
    :return:
    """
    # 彩色图->灰度图      ndarray.adim  表示数组的维度如二维：返回2； 三维： 返回3.
    if image1.ndim == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
    if image2.ndim == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 直方图均衡
    image1 = cv2.equalizeHist(image1)
    image2 = cv2.equalizeHist(image2)

    return image1, image2


def getRectifyTransform(ccp):
    """
    获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
    :param ccp: CameraConfigsParameters，是一个类，存储着双目标定的参数:config = stereoconfig.stereoCamera()
    :return:
    """
    # 读取内参和外参
    left_camera_matrix = ccp.left_camera_matrix
    left_distortion = ccp.left_distortion

    right_camera_matrix = ccp.right_camera_matrix
    right_distortion = ccp.right_distortion
    # size(height, width) -> (width, height)
    size = ccp.size[::-1]
    # 旋转、平移矩阵
    R = ccp.R
    T = ccp.T

    # 进行立体更正 size() -> (width, height)
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                      right_camera_matrix, right_distortion,
                                                                      size, R, T, alpha=0)
    # 计算、矫正、变换
    left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion,
                                                       R1, P1, size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion,
                                                         R2, P2, size, cv2.CV_16SC2)

    return left_map1, left_map2, right_map1, right_map2, Q


def rectifyImage(image1, image2, left_map1, left_map2, right_map1, right_map2):
    """
    畸变校正和立体校正
    :param image1:
    :param image2:
    :param left_map1:
    :param left_map2:
    :param right_map1:
    :param right_map2:
    :return:
    """
    img1_rectified = cv2.remap(image1, left_map1, left_map2, cv2.INTER_AREA)
    img2_rectified = cv2.remap(image2, right_map1, right_map2, cv2.INTER_AREA)

    return img1_rectified, img2_rectified


def stereoMatchSGBM(left_image, right_image, down_scale=False):
    """
    使用SGBM进行视差计算
    :param left_image:
    :param right_image:
    :param down_scale:
    :return:
    """
    # SGBM匹配参数设置
    if left_image.ndim == 2:    # 判断是否是灰度图
        img_channels = 1
    else:
        img_channels = 3
        pass

    blockSize = 3
    paraml = {'minDisparity': 0,
              'numDisparities': 128,
              'blockSize': blockSize,
              'P1': 8 * img_channels * blockSize ** 2,
              'P2': 32 * img_channels * blockSize ** 2,
              'disp12MaxDiff': 1,
              'preFilterCap': 63,
              'uniquenessRatio': 15,
              'speckleWindowSize': 100,
              'speckleRange': 1,
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              }
    # 构建SGBM对象
    left_matcher = cv2.StereoSGBM_create(**paraml)

    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)

    size = (left_image.shape[1], left_image.shape[0])

    # 计算视差图
    if not down_scale:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)
    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]

        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right
        pass
    # 真实视差（因为SGBM算法得到的视差是×16的）
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.
    # 主要是针对左目做出的视差，相对于右目的丢弃
    # disp, _ = stereoMatchSGBM(left_image, right_image, False):
    # cv2.imshow('视差', disp)
    return trueDisp_left, trueDisp_right


def draw_line(image1, image2):
    """
    # 立体校正检验----画线
    :param image1:
    :param image2:
    :return:输出左右拼接号带线的图像
    """
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)),
                 (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        pass
    return output

# # 读取MiddleBurry数据集的图片
# iml = cv2.imread('/data/数据/MiddleBurry/Adirondack-perfect/im0.png')  # 左图
# imr = cv2.imread('/data/数据/MiddleBurry/Adirondack-perfect/im1.png')  # 右图
# height, width = iml.shape[0:2]
#
# # 读取相机内参和外参
# config = stereoconfig.stereoCamera1()
#
# # 立体校正
# # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
# map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)
# iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
# print(Q)
#
# # 绘制等间距平行线，检查立体校正的效果
# line = draw_line(iml_rectified, imr_rectified)
# cv2.imwrite('/data/检验.png', line)
#
# # 立体匹配
# iml_, imr_ = preprocess(iml, imr)  # 预处理，一般可以削弱光照不均的影响，不做也可以
# disp, _ = stereoMatchSGBM(iml_, imr_, True)  # 这里传入的是未经立体校正的图像，因为我们使用的middleburry图片已经是校正过的了
# cv2.imwrite('/data/视差.png', disp)
#
# # 计算像素点的3D坐标（左相机坐标系下）
# points_3d = cv2.reprojectImageTo3D(disp, Q)  # 可以使用上文的stereo_config.py给出的参数
