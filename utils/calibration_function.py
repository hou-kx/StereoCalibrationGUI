# -*- coding:utf-8 -*-
import cv2
import numpy as np


def findChessboardCorners(image, pattern_size, corners, flage=cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE):
    """
    找到标定图像的角点（标定板上的内角点，与边缘不接触），
    :param image:标定图像的mat图像，8位灰度或者彩色图像
    :param pattern_size:每个棋盘图上点的行列数，一般情况下，要求行列不相同，便于后续标定程序识别标定板的方向
    :param corners:用于存储检测到的内角点图像坐标的位置
    :param flage:用于定于棋盘图上内角点查找的不同处理方式，有默认值默认int flags=CALIB_CB_ADAPTIVE_THRESH+CALIB_CB_NORMALIZE_IMAGE
    :return:
    """
    cv2.findChessboardCorners(image, pattern_size, corners, flage)
    pass


def cornerSubPix(image, corners, win_size, zero_zone, criteria):
    """
    专门用来获取棋盘图上内角点的精确位置的
    为了提高标定精度，需要在初步提取的角点信息上进一步提取亚像素信息，降低相机标定偏差，
    常用的方法是cornerSubPix，另一个方法是使用find4QuadCornerSubpix函数
    :param image:输入的Mat矩阵，最好是8位灰度图像，检测效率更高
    :param corners:初始的角点坐标向量，同时作为亚像素坐标位置的输出
    :param win_size:大小为搜索窗口的一半，（n,n） win_size = (2*n+1) * (2*n+1) 作为输入，被修改了值
    :param zero_zone:死区的一半尺寸，死区为不对搜索区的中央位置做求和运算的区域。它是用来避免自相关矩阵出现某些可能的奇异性。当值为（-1，-1）时表示没有死区
    :param criteria:定义求角点的迭代过程的终止条件，可以为迭代次数和角点精度两者的组合；
    :return: 返回None
    """
    cv2.cornerSubPix(image, corners, win_size, zero_zone, criteria)
    pass


def drawChessboardCorners(image, pattern_size, corners, pattern_was_found):
    """
    绘制被成功标定的角点
    :param image:8位灰度或者彩色图像
    :param pattern_size:每张标定棋盘上内角点的行列数
    :param corners:初始的角点坐标向量，同时作为亚像素坐标位置的输出，所以需要是浮点型数据，
    :param pattern_was_found:标志位，用来指示定义的棋盘内角点是否被完整的探测到，true表示别完整的探测到，函数会用直线依次连接所有的内角点，
                            false表示有未被探测到的内角点，这时候函数会以（红色）圆圈标记处检测到的内角点；
    :return:
    """
    cv2.drawChessboardCorners(image, pattern_size, corners, pattern_was_found)
    pass


def calibrateCamera(object_points, image_point, image_size, camera_matrix, dist_coeffs, rvecs, tvecs, flags, criteria):
    """
    获取到棋盘标定图的内角点图像坐标之后，就可以使用calibrateCamera函数进行标定，计算相机内参和外参系数
    在使用该函数进行标定运算之前，需要对棋盘上每一个内角点的空间坐标系的位置坐标进行初始化，
    标定的结果是生成相机的内参矩阵cameraMatrix、相机的5个畸变系数distCoeffs，另外每张图像都会生成属于自己的平移向量和旋转向量

    :param object_points:为世界坐标系中的三维点。在使用时，应该输入一个三维坐标点的向量的向量，
                            需要依据棋盘上单个黑白矩阵的大小，计算出（初始化）每一个内角点的世界坐标。
    :param image_point:为每一个内角点对应的图像坐标点。
    :param image_size:为图像的像素尺寸大小，在计算相机的内参和畸变矩阵时需要使用到该参数；
    :param camera_matrix:相机的内参矩阵。
    :param dist_coeffs:畸变矩阵
    :param rvecs:旋转向量；应该输入一个Mat类型的vector，即vector<Mat>rvecs
    :param tvecs:位移向量，和rvecs一样，应该为vector<Mat> tvecs
    :param flags:V_CALIB_USE_INTRINSIC_GUESS：使用该参数时，在cameraMatrix矩阵中应该有fx,fy,u0,v0的估计值。
                                                否则的话，将初始化(u0,v0）图像的中心点，使用最小二乘估算出fx，fy。
                CV_CALIB_FIX_PRINCIPAL_POINT：在进行优化时会固定光轴点。当CV_CALIB_USE_INTRINSIC_GUESS参数被设置，
                                                光轴点将保持在中心或者某个输入的值。
                CV_CALIB_FIX_ASPECT_RATIO：固定fx/fy的比值，只将fy作为可变量，进行优化计算。
                CV_CALIB_USE_INTRINSIC_GUESS没有被设置，fx和fy将会被忽略。只有fx/fy的比值在计算中会被用到。
                CV_CALIB_ZERO_TANGENT_DIST：设定切向畸变参数（p1,p2）为零。
                CV_CALIB_FIX_K1,…,CV_CALIB_FIX_K6：对应的径向畸变在优化中保持不变。
                CV_CALIB_RATIONAL_MODEL：计算k4，k5，k6三个畸变参数。如果没有设置，则只计算其它5个畸变参数。
    :param criteria:最优迭代终止条件设定
    :return: ret检测成功与否， 相机矩阵， 畸变系数， 旋转， 平移向量
    """
    cv2.calibrateCamera(object_points, image_point, image_size,
                        camera_matrix, dist_coeffs, rvecs, tvecs,
                        flags, criteria)
    pass


def project_points(object_points, rvec, tvec, camera_matrix, distcoeffs, image_points, jacobian, aspect_ratio):
    """
    对标定结果进行评价的方法是通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到空间三维点在图像上新的投影点的坐标，
    计算投影坐标和亚像素角点坐标之间的偏差，偏差越小，标定结果越好
    :param object_points:为相机坐标系中的三维点坐标；
    :param rvec:旋转向量，每一张图像都有自己的选择向量；
    :param tvec:位移向量，每一张图像都有自己的平移向量；
    :param camera_matrix:求得的相机的内参数矩阵；
    :param distcoeffs:相机的畸变矩阵；
    :param image_points:每一个内角点对应的图像上的坐标点；
    :param jacobian:是雅可比行列式；
    :param aspect_ratio:是跟相机传感器的感光单元有关的可选参数，如果设置为非0，则函数默认感光单元的dx/dy是固定的，会依此对雅可比矩阵进行调整；
    :return:
    """
    cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distcoeffs, image_points, jacobian, aspect_ratio)
    pass


def mean_error(obj_points, img_points, rvecs, tvecs, camera_matrix, distortion):
    """
           更改的矫正之后的重投影误差计算方式
           https://blog.csdn.net/qq_32998593/article/details/113063216
    :param obj_points:  真是世界的3D点
    :param img_points:  图像平面的2D点
    :param rvecs:   旋转向量
    :param tvecs:   平移向量
    :param camera_matrix:   相机内参
    :param distortion:  相机畸变矩阵
    :return:  返回整体的平均误差
    """
    meanError = 0
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, distortion)
        # error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2)
        # mean_error += error
        meanError += error * error
        pass
    total_error = np.sqrt(meanError / (len(obj_points) * len(obj_points[0])))
    return total_error


def calibrator(image_list, square_size, board_size=(9, 6)):
    """
     相机标定       # 相机矩阵，畸变参数，旋转、平移向量 matrix disp
    :param image_list: 标定图片列表
    :param square_size:标定板的方格的大小
    :param board_size: 标定板的内角点的个数  行X列
    :return:
    """

    image_number = len(image_list)
    # board_size = (9, 6)  # 也就是boardSize
    # square_size = 20

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    # objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

    # 构建标定板的点坐标，objectPoints    准备对象点, 如 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((np.prod(board_size), 3), np.float32)
    # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp[:, :2] = np.indices(board_size).T.reshape(-1, 2)
    objp *= square_size
    # 生成标定图片数目数量个矩阵
    objp = [objp] * image_number

    # 用于存储所有图像对象点与图像点的矩阵
    obj_points = []  # 在真实世界中的 3d 点
    img_points = []  # 在图像平面中的 2d 点
    count = 0  # 标记检测到的棋盘画面数量
    # img_size = (640, 480)
    for fname in image_list:
        img = cv2.imread(fname)
        # img = cv2.resize(img, (320, 240))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 因为数组的shape为行x列而图片大小为宽x高，正好是反过来的
        img_size = img_gray.shape[::-1]
        # 找到棋盘上所有的角点，这里角点指的是, 6行9列
        ret, corners = cv2.findChessboardCorners(img_gray, board_size, None)
        # 如果找到了, 便添加对象点和图像点(在细化后)
        if ret:
            # obj_points.append(objp)
            obj_points.append(objp[count])
            # 找到角点的亚像素（sub pix），即更为精确的角点的位置（精确到亚像素）另一个函数方法是cv2.find4QuadCornerSubpix 在原角点的基础上寻找亚像素角点
            # corners_sub = cf.cornerSubPix(img_gray, corners, (5, 5), (-1, -1), criteria)
            corners_sub = cv2.cornerSubPix(img_gray, corners, (3, 3), (-1, -1), criteria)
            # if corners_sub is not None:
            if corners_sub.any:
                # img_points.append(corners_sub)
                img_points.append(corners_sub / 1.0)
            else:
                img_points.append(corners)
            # img_points.append(corners)
            # 把角点绘制出来，仅仅为了显示而已
            cv2.drawChessboardCorners(img, board_size, corners, ret)
            win_name = 'img' + str(count)
            cv2.namedWindow(win_name)
            # 并排打开两个窗口 left、right
            cv2.moveWindow(win_name, 200, 200)
            cv2.imshow(win_name, img)
            # cv2.waitKey(1000)
            cv2.destroyAllWindows()
            count += 1
            pass
    pass
    # 标定，相机,ret重投影误差的内参矩阵cameraMatrix、相机的5个畸变系数distCoeffs，另外每张图像都会生成属于自己的平移向量和旋转向量
    ret, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
    # 重投影误差计算方式
    total_error = mean_error(obj_points, img_points, rvecs, tvecs, camera_matrix, distortion)
    # print("total error: ", total_error)
    # np.savetxt('data/calibrate.npz', mtx=camera_matrix, dist=distortion[0:4])
    # When everything done, release the capture
    cv2.destroyAllWindows()
    # cap.release()
    # 演示矫正后的

    return obj_points, img_points, ret, camera_matrix, distortion, rvecs, tvecs, total_error
