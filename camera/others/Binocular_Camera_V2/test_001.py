import cv2
import numpy as np
import camera_configs


if __name__ == '__main__':

    # 左键点击事件，左键点击输出深度距离
    def callbackFunc(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            cv2.rectangle(disp, (x, y), (x + 3, y + 3), (255, 255, 255), 3)
            # 反向填充
            # cv2.rectangle(img, (50, 50), (500+3, 500+3), (255, 0, 0), -1)

            cv2.imshow("image", disp)  # 显示图片，后面会讲解

            # print(threeD[y][x])
            print(threeD[y][x] / 1000)
            pass
        pass


    # 打开一个窗口命名为depth，
    cv2.namedWindow('depth')
    cv2.namedWindow('ctl')
    # 并排打开两个窗口 left、right
    cv2.moveWindow('left', 0, 0)
    cv2.moveWindow('right', 640, 0)
    # 进度条
    cv2.createTrackbar('num', 'ctl', 2, 10, lambda x: None)
    cv2.createTrackbar('blockSize', 'ctl', 8, 255, lambda x: None)

    cv2.createTrackbar('minDisparity', 'ctl', 1, 3, lambda x: None)
    cv2.createTrackbar('disp12MaxDiff', 'ctl', 1, 3, lambda x: None)
    cv2.createTrackbar('preFilterCap', 'ctl', 1, 3, lambda x: None)
    cv2.createTrackbar('uniquenessRatio', 'ctl', 10, 20, lambda x: None)
    cv2.createTrackbar('speckleWindowSize', 'ctl', 100, 200, lambda x: None)
    cv2.createTrackbar('speckleRange', 'ctl', 100, 200, lambda x: None)

    # 定义在depth窗口上的单击打事件
    cv2.setMouseCallback('depth', callbackFunc, None)
    # 打开视频
    cap = cv2.VideoCapture(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 进行双目分割
    while True:
        ret, frame = cap.read()
        frame1 = frame[0:480, 0:640]

        ret, frame = cap.read()
        frame2 = frame[0:480, 640:1280]

        img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)

        # BM匹配算法要转换为灰度图，计算匹配
        imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

        #
        num = cv2.getTrackbarPos('num', 'ctl') + 1
        blockSize = cv2.getTrackbarPos('blockSize', 'ctl')

        minDisparity = cv2.getTrackbarPos('minDisparity', 'ctl')
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'ctl')
        preFilterCap = cv2.getTrackbarPos('preFilterCap', 'ctl')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'ctl')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'ctl')
        speckleRange = cv2.getTrackbarPos('speckleRange', 'ctl')

        if blockSize % 2 == 0:
            blockSize += 1
            pass

        if blockSize < 3:
            blockSize = 3
            pass

        # 匹配， 16的倍数？ 这几个参数很重要
        # stereo = cv2.StereoBM_create(numDisparities=16 * num, blockSize=blockSize)
        # # # blockSize = 31
        img_channels = 3
        stereo = cv2.StereoSGBM_create(minDisparity=-1,
                                       numDisparities=16 * num,
                                       blockSize=blockSize,
                                       P1=8 * img_channels * blockSize * blockSize,
                                       P2=32 * img_channels * blockSize * blockSize,
                                       disp12MaxDiff=-1,
                                       preFilterCap=preFilterCap,
                                       uniquenessRatio=uniquenessRatio,
                                       speckleWindowSize=speckleWindowSize,
                                       speckleRange=speckleRange,
                                       mode=cv2.STEREO_SGBM_MODE_HH)

        # 根据Block Maching方法生成差异图（opencv里也提供了SGBM/Semi-Global Block Matching算法，有兴趣可以试试）
        # stereo = cv2.StereoBM_create(numDisparities=16 * num, blockSize=blockSize)

        # 计算视差图
        disparity = stereo.compute(imgL, imgR)
        # 显示，距离
        disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # 距离 /16
        threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., camera_configs.Q)

        # 显示窗口
        cv2.imshow('left', img1_rectified)
        cv2.imshow('right', img2_rectified)
        cv2.imshow('depth', disp)
        cv2.imshow('ctl', disp)
        # cv2.imshow('ceshi', disparity)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        # elif key == ord('s'):
        #     cv2.imwrite(path_BM_left, imgL)
        #     cv2.imwrite(path_BM_left, imgR)
        #     cv2.imwrite(path_BM_depth, disp)
        #     pass
        pass

    cap.release()
    cv2.destroyAllWindows()
    pass
