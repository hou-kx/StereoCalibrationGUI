#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Calibration"""
__author__ = "@温暖"
__copyright__ = "Copyright (C) 2021 @温暖"
__license__ = "Public Domain"
__version__ = "1.0"

import numpy as np
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import argparse
import cv2
import utils.calibration_function as cf
import time


# 图像转换为可显示在canvas中的格式
def frame2tkImage(frame, shape=(640, 480)):
    """
    图像转换，用于在画布中显示
    :param frame:
    :param shape:
    :return:
    """
    cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image)
    pil_image = pil_image.resize((shape[0], shape[1]), Image.ANTIALIAS)
    tk_image = ImageTk.PhotoImage(image=pil_image)
    return tk_image


def camera_calibrator(img_path_l, img_path_r, img_format, mode, square_size, board_size=(9, 6), img_size=(640, 480)):
    """

    :param square_size:
    :param img_path_l:
    :param img_path_r:
    :param img_format:
    :param mode:  选定是单目还是双目标定模式
    :param board_size:(x, y)每个棋盘图上点的行列数，一般情况下，要求行列不相同，便于后续标定程序识别标定板的方向
    :param img_size:
    :return:
    """
    # 读取对应目录下的图片，返回一个列表
    images_l = glob.glob(img_path_l + '/*.' + img_format)
    if len(images_l) == 0:
        return messagebox.showwarning('读取失败',
                                      'The current directory does not contain the corresponding images,'
                                      ' failed to read！！')
    # 左目标定
    obj_points_l, img_points_l, ret_l, camera_matrix_l, distortion_l, \
        rvecs_l, tvecs_l, mean_error_l = cf.calibrator(images_l, square_size, board_size)

    # 双目标定模式下再进行右目标定
    if mode == 2 and img_path_r is not None:
        images_r = glob.glob(img_path_r + '/*.' + img_format)
        if len(images_r) == 0:
            return messagebox.showwarning('读取失败',
                                          'The current directory does not contain the corresponding images_r,'
                                          ' failed to read！！')
        # 右目标定
        obj_points_r, img_points_r, ret_r, camera_matrix_r, distortion_r, \
            rvecs_r, tvecs_r, mean_error_r = cf.calibrator(images_r, square_size, board_size)

        # 立体标定
        rms, left_camera_matrix, left_distortion, right_camera_matrix, right_distortion, \
            R, T, E, F = cv2.stereoCalibrate(obj_points_r, img_points_l, img_points_r,
                                             camera_matrix_l, distortion_l,
                                             camera_matrix_r, distortion_r,
                                             img_size, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

        # tx为左右相机距离，本例中为40mm
        print('rms:', rms, '\n\nleft_camera_matrix:', left_camera_matrix, '\n\nleft_distortion:', left_distortion,
              '\n\nright_camera_matrix:', right_camera_matrix, '\n\nright_distortion:', right_distortion,
              '\n\nT:', T, '\n\nE:', E
              )

        # 保存
        np.savez('data/binocular/b' + time.strftime('%m%d%H%M%S', time.localtime(time.time())) + '.npz',
                 rms=rms,
                 left_camera_matrix=left_camera_matrix,
                 left_distortion=left_distortion,
                 right_camera_matrix=right_camera_matrix,
                 right_distortion=right_distortion,
                 R=R, T=T, E=E, F=F)
        pass
    else:
        print('ret:\n', ret_l, '\ncamera_matrix=\n', camera_matrix_l, '\n\ndistortion=\n', distortion_l,
              "\n\ntotal error: ", mean_error_l)

        # 保存obj_points_l, img_points_l, ret_l, camera_matrix_l, distortion_l, \
        #         rvecs_l, tvecs_l, mean_error_l
        np.savez('data/monocular/m' + time.strftime('%m%d%H%M%S', time.localtime(time.time())) + '.npz',
                 ret=ret_l,
                 camera_matrix=camera_matrix_l,
                 distortion=distortion_l,
                 rvecs=rvecs_l, tvecs=tvecs_l,
                 mean_error=mean_error_l)
        pass
    # When everything done, release the capture
    # cv2.destroyAllWindows()
    # cap.release()
    # 演示矫正后的

    # return ret, camera_matrix, distortion, rvecs, tvecs, total_error
    pass


""" 按键事件"""


def Btn_Browse(rw, camera):
    """
    选择视频文件获取其文件地址
    :param rw: 为RootWindows对象
    :param camera: 选择的相机left or right
    :return: None
    """
    file_path = tk.filedialog.askdirectory(title=u'选择文件夹')
    # \\转换为/
    # file_path = file_path.replace('/', '\\\\')
    if camera == 1:
        if file_path not in rw.path_left:
            rw.path_left.insert(0, file_path)
            pass
        rw.combobox_path_left['value'] = rw.path_left
        rw.combobox_path_left.current(0)
        pass
    else:
        if file_path not in rw.path_right:
            rw.path_right.insert(0, file_path)
            pass
        rw.combobox_path_right['value'] = rw.path_right
        rw.combobox_path_right.current(0)
        pass
    # print('left:', rw.path_left, '\tright:', rw.path_right, file_path)
    pass


def Btn_Run(rw):
    calibration_model = rw.combobox_model.get()
    img_format = rw.combobox_format.get()
    img_path_l = rw.combobox_path_left.get()

    square_size = float(rw.square_size_entry.get())

    if calibration_model == 'Monocular-camera':
        camera_calibrator(img_path_l, None, img_format, 1, square_size)
        pass
    else:  # 双目模式
        img_path_r = rw.combobox_path_right.get()
        camera_calibrator(img_path_l, img_path_r, img_format, 2, square_size)
        pass

    rw.combobox_model.configure(state=tk.DISABLED)
    rw.combobox_format.configure(state=tk.DISABLED)
    rw.square_size_entry.configure(state=tk.DISABLED)
    rw.btn_run.configure(state=tk.DISABLED)

    pass


def Btn_Cancel(rw):
    """
    取消当前的所有操作,参数初始化
    :param rw:RootWindow
    :return:
    """
    # 初始化
    rw.combobox_path_left.delete(0, 'end')
    rw.combobox_path_right.delete(0, 'end')

    rw.combobox_model.current(0)
    rw.combobox_model_select(rw)
    rw.combobox_format.current(0)

    rw.combobox_model.configure(state=tk.NORMAL)
    rw.combobox_format.configure(state=tk.NORMAL)
    rw.btn_run.configure(state=tk.NORMAL)

    rw.square_size_entry.configure(state=tk.NORMAL)
    rw.square_size_entry.delete(0, "end")
    rw.square_size_entry.insert(0, "25")
    pass


# 屏蔽子控件
def disableChildren(parent):
    """
    把父容器种所有的控件disable
    :param parent:
    :return:
    """
    for child in parent.winfo_children():
        wtype = child.winfo_class()
        if wtype not in ('Frame', 'Labelframe'):
            child.configure(state='disable')
        else:
            disableChildren(child)


def enableChildren(parent):
    """
    把父容器种所有的控件enable
    :param parent:
    :return:
    """
    for child in parent.winfo_children():
        wtype = child.winfo_class()
        # print(wtype)
        if wtype not in ('Frame', 'Labelframe'):
            child.configure(state='normal')
        else:
            enableChildren(child)


class RootWindow(object):
    """窗口主体"""

    def __init__(self, args, shape_window=(700, 500)):
        """

        :param args:
        :param shape_window:
        """
        """参数初始化"""
        # 参数
        self.args = args
        # 图片格式
        self.format = ''
        # 图片路径列表
        self.path_left = []
        self.path_right = []

        """窗体初始化"""
        self.root = tk.Tk()
        self.root.title('相机标定—V1—@温暖')
        # 不可拉伸
        self.root.resizable(width=False, height=False)
        # 放置位置
        sr_shape = str(shape_window[0]) + 'x' + str(shape_window[1])
        # sr_locate = '+' + str(locate[0] - shape[0]) + '+' + str(locate[1])
        self.root.geometry(sr_shape)
        # 颜色
        fram_bg = 'Aliceblue'
        select_bg = 'Turquoise4'
        button_bg = 'lightcyan'

        """下拉框frame"""
        frame_combobox = tk.Frame(self.root, bg=fram_bg)
        # 选择单双目标定，默认单目
        combobox_model_lab = tk.Label(frame_combobox, bg=fram_bg, text=' M or B')
        self.combobox_model = ttk.Combobox(frame_combobox)
        self.combobox_model['value'] = ('Monocular-camera', 'Binocular-Camera')
        self.combobox_model.current(0)
        # 选择标定图片的格式：jpg, jpeg, png
        combobox_format_lab = tk.Label(frame_combobox, bg=fram_bg, text='\tFormat')
        self.combobox_format = ttk.Combobox(frame_combobox)
        self.combobox_format['value'] = ('jpg', 'jpeg', 'png')
        self.combobox_format.current(0)
        # button to cancel
        btn_cancel = tk.Button(frame_combobox, text='Cancel', command=lambda: Btn_Cancel(self), width=8)
        # 下拉框颜色
        combostyle = ttk.Style()
        combostyle.theme_create('combostyle', parent='alt', settings={'TCombobox': {'configure': {
            'foreground': 'black',  # 前景色
            'selectbackground': select_bg,  # 选择后的背景颜色
            'fieldbackground': 'white',  # 下拉框颜色
            'background': button_bg,  # 下拉按钮颜色
        }}})
        combostyle.theme_use('combostyle')

        """加载图片所在路径的frame"""
        # tk.LabelFrame.grid_propagate(flag=0)
        frame_path = tk.LabelFrame(self.root, bg=fram_bg, labelanchor='nw', padx=5)
        # 使得设置宽高生效grid_propagate(0)
        frame_path.grid_propagate(False)
        frame_path.config(width=690, height=460, text='Load Stereo Iamges')

        """左目文件夹"""
        # 选择左目标定图像文件夹
        frame_path_left = tk.Frame(frame_path, bg=fram_bg)

        combobox_path_left_lab = tk.Label(frame_path_left, bg=fram_bg, text=' Folder for images from camera 1')
        self.combobox_path_left = ttk.Combobox(frame_path_left, width=40)
        self.combobox_path_left['value'] = []
        # self.combobox_path_left.current(0)
        combobox_path_left_btn = tk.Button(frame_path_left, text='···', command=lambda: Btn_Browse(self, 1), width=3)

        # 选择右目标定图像文件夹
        self.frame_path_right = tk.Frame(frame_path, bg=fram_bg)
        combobox_path_right_lab = tk.Label(self.frame_path_right, bg=fram_bg, text=' Folder for images from camera 2')
        # self.combobox_path_right = ttk.Combobox(self.frame_path_right, width=40, font=tkfont.Font(size=13))
        self.combobox_path_right = ttk.Combobox(self.frame_path_right, width=40)
        self.combobox_path_right['value'] = []
        # self.combobox_path_right.current(0)
        combobox_path_right_btn = tk.Button(self.frame_path_right, text='···', command=lambda: Btn_Browse(self, 2),
                                            width=3)

        # 选择标定板方块的边长单位mm
        frame_square_size = tk.Frame(frame_path, bg=fram_bg)

        # self.strv = tk.StringVar()
        # self.strv.set("25")
        square_size_left_lab = tk.Label(frame_square_size, bg=fram_bg, text='Size of checkerboard square:  ')
        # self.square_size_entry = tk.Entry(frame_square_size, textvariable=self.strv, width=10)
        self.square_size_entry = tk.Entry(frame_square_size, width=10)
        self.square_size_entry.insert(0, "25")
        square_size_right_lab = tk.Label(frame_square_size, bg=fram_bg, text=' millimeters')
        # self.combobox_path_left = ttk.Combobox(frame_square_size, width=40)

        """按钮"""
        frame_path_btn = tk.Frame(frame_path, bg=fram_bg)
        btn_show = tk.Button(frame_path_btn, text='Show', width=10)
        self.btn_run = tk.Button(frame_path_btn, text='Run', width=10, command=lambda: Btn_Run(self))
        btn_quit = tk.Button(frame_path_btn, text='Quit', width=10, command=self.root.destroy)

        """各个frame、控件放置"""
        # frame_combobox
        frame_combobox.grid(row=0, column=0, sticky=tk.NW)
        combobox_model_lab.grid(row=0, column=0, sticky=tk.W)
        self.combobox_model.grid(row=0, column=1, sticky=tk.W, ipadx=3)
        combobox_format_lab.grid(row=0, column=2, sticky=tk.W)
        self.combobox_format.grid(row=0, column=3, sticky=tk.W, ipadx=3, ipady=3)
        btn_cancel.grid(row=0, column=4, sticky=tk.E + tk.W, padx=12)

        # frame_path
        frame_path.grid(row=1, column=0, sticky=tk.NW, padx=5)

        frame_path_left.grid(row=0, column=0, columnspan=4, sticky=tk.S + tk.N, pady=30, padx=80)
        combobox_path_left_lab.grid(row=0, column=0, columnspan=2, sticky=tk.W)
        self.combobox_path_left.grid(row=1, column=1, columnspan=2, sticky=tk.W + tk.E, ipadx=12, ipady=5)
        combobox_path_left_btn.grid(row=1, column=3, sticky=tk.W + tk.E)

        self.frame_path_right.grid(row=1, column=0, columnspan=4, sticky=tk.E + tk.W, pady=50, padx=80)
        combobox_path_right_lab.grid(row=0, column=0, columnspan=2, sticky=tk.W)
        self.combobox_path_right.grid(row=1, column=1, columnspan=2, sticky=tk.E + tk.W, ipadx=12, ipady=5)
        combobox_path_right_btn.grid(row=1, column=3, sticky=tk.E + tk.W)

        frame_square_size.grid(row=2, column=3, sticky=tk.E, pady=20, padx=80)
        square_size_left_lab.grid(row=0, column=0, sticky=tk.W, padx=8)
        self.square_size_entry.grid(row=0, column=1, sticky=tk.W)
        square_size_right_lab.grid(row=0, column=2, sticky=tk.E)

        frame_path_btn.grid(row=3, column=3, sticky=tk.E, pady=30)
        btn_show.grid(row=0, column=0, sticky=tk.W)
        self.btn_run.grid(row=0, column=1, sticky=tk.W, padx=50)
        btn_quit.grid(row=0, column=2, sticky=tk.E)

        # 绑定 下拉框的触发事件
        self.combobox_model.bind("<<ComboboxSelected>>", self.combobox_model_select)
        self.combobox_path_left.bind("<<ComboboxSelected>>", self.combobox_path_left_select)
        self.combobox_path_right.bind("<<ComboboxSelected>>", self.combobox_path_right_select)
        # self.combobox_SGBM_mode.bind("<<ComboboxSelected>>", self.combobox_SGBM_mode_select)

        # 默认单目，屏蔽摄像头2获取
        self.combobox_model_select()
        pass

    """功能函数"""

    # 下拉框
    def combobox_model_select(self, *args):
        """更改下拉框修改浏览按钮状态"""
        model_data = self.combobox_model.get()
        # print(model_data)     'Monocular-camera', 'Binocular-Camera'
        # 单目模式
        if model_data == 'Monocular-camera':
            # 屏蔽掉右目路径
            disableChildren(self.frame_path_right)
            pass
        # 双目模式
        else:
            # 激活右目路径
            enableChildren(self.frame_path_right)
            pass
        # self.root_update()
        pass

    def combobox_path_left_select(self, *args):
        pass

    def combobox_path_right_select(self, *args):
        pass

    def root_update(self):
        self.root.update_idletasks()
        self.root.update()
        # self.root.deiconify()
        pass

    pass


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    # parser.add_argument('--deep_sort_weights', type=str,
    #                     default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7',
    #                     help='ckpt.t7 path')
    # # file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='0', help='source')
    # parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    # parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    # parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    # # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # # parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    # parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    # parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    # parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    # parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")

    args = parser.parse_args()
    # args.img_size = check_img_size(args.img_size)

    rw = RootWindow(args)
    # rw.root_update()
    rw.root.mainloop()
    pass


if __name__ == '__main__':
    main()
    pass
