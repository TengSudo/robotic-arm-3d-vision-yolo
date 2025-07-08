#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import threading
import numpy as np
from datetime import datetime
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QTimer
from untitled import Ui_ME5110Q
import cv2
import open3d as o3d
import pyrealsense2 as rs
from ultralytics import YOLO
from cri_lib import CRIController, CRIConnectionError, CRICommandTimeOutError
from scipy.spatial.transform import Rotation

class ME5110QGUI:
    def __init__(self, app):
        super().__init__()
        self.app = app

        # Tạo cửa sổ chính
        self.main_window = QtWidgets.QMainWindow()
        self.ui = Ui_ME5110Q()
        self.ui.setupUi(self.main_window)

        # Khởi tạo CRIController
        self.robot_controller = None

        # Khởi tạo camera RealSense
        self.pipeline = None
        self.align = None
        self.custom_colormap = None
        self.display_mode = "RGB"  # Mặc định hiển thị RGB

        # Khởi tạo mô hình YOLOv11 OBB
        try:
            self.model1 = YOLO('best_obb_1.pt')
            self.model2 = YOLO('best_obb_2.pt')
            print("Đã tải mô hình YOLOv11 OBB thành công")
            self.append_log("Đã tải mô hình YOLOv11 OBB thành công")
        except Exception as e:
            print(f"Lỗi khi tải mô hình YOLOv11: {str(e)}")
            self.append_log(f"Lỗi khi tải mô hình YOLOv11: {str(e)}")
            self.model1 = None
            self.model2 = None

        # Kết nối các sự kiện
        self.connect_signals()

        # Hiển thị cửa sổ
        self.main_window.show()
        print("ME5110Q GUI đã khởi chạy")
        self.append_log("ME5110Q GUI đã khởi chạy")

        # Thiết lập phạm vi cho slider_override
        self.ui.slider_override.setMinimum(0)
        self.ui.slider_override.setMaximum(100)
        self.ui.slider_override.setValue(0)

        # Thiết lập phạm vi cho slider_conf
        self.ui.slider_conf.setMinimum(30)  # Tương ứng 0.3
        self.ui.slider_conf.setMaximum(100)  # Tương ứng 1.0
        self.ui.slider_conf.setValue(70)  # Tương ứng 0.7

        # Timer để cập nhật trạng thái
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status_labels)
        self.status_timer.setInterval(100)

        # Timer để cập nhật luồng camera
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_stream)
        self.camera_timer.setInterval(33)  # ~30fps

        #
        self.current_depth_frame = None
        self.current_depth_intr = None
        self.model1_class0_data = []
        self.class2_centers_to_draw = []
        self.model2_data = []

        # Chương trình
        self.program_state = "STOPPED"
        self.is_looping = False
        self.current_step = None
        self.program_thread = None
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.robot_lock = threading.Lock()

    def append_log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        QtCore.QMetaObject.invokeMethod(
            self.ui.textEdit_log,
            "append",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, log_message)
        )
        QtCore.QMetaObject.invokeMethod(
            self.ui.textEdit_log.verticalScrollBar(),
            "setValue",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(int, self.ui.textEdit_log.verticalScrollBar().maximum())
        )

    def update_status_labels(self):
        if self.robot_controller is not None and self.robot_controller.connected:
            error_state = self.robot_controller.robot_state.combined_axes_error
            estop_state = str(self.robot_controller.robot_state.emergency_stop_ok)
            kin_state = str(self.robot_controller.robot_state.kinematics_state)
            pos_cart = self.robot_controller.robot_state.position_robot
            pos_cart_str = (f"X_{pos_cart.X:.2f} Y_{pos_cart.Y:.2f} Z_{pos_cart.Z:.2f} "
                           f"A_{pos_cart.A:.2f} B_{pos_cart.B:.2f} C_{pos_cart.C:.2f}")
            pos_joints = self.robot_controller.robot_state.joints_current
            pos_joints_str = (f"A1_{pos_joints.A1:.2f} A2_{pos_joints.A2:.2f} A3_{pos_joints.A3:.2f} "
                             f"A4_{pos_joints.A4:.2f} A5_{pos_joints.A5:.2f} A6_{pos_joints.A6:.2f}")
            gripper_state = "Closed" if self.robot_controller.robot_state.dout[20] else "Open"
            self.ui.label_error.setText(f"Error: {error_state}")
            self.ui.label_estop.setText(f"E-Stop: {estop_state}")
            self.ui.label_kinstate.setText(f"{kin_state}")
            self.ui.label_pos_cart.setText(f"PosCart: {pos_cart_str}")
            self.ui.label_pos_joints.setText(f"PosJoints: {pos_joints_str}")
            self.ui.label_gripper.setText(f"Gripper: {gripper_state}")
        else:
            self.ui.label_error.setText("Error: Không kết nối")
            self.ui.label_estop.setText("E-Stop: Không kết nối")
            self.ui.label_kinstate.setText("KinState: Không kết nối")
            self.ui.label_pos_cart.setText("PosCart: Không kết nối")
            self.ui.label_pos_joints.setText("PosJoints: Không kết nối")
            self.ui.label_gripper.setText("Gripper: Không kết nối")

    def connect_signals(self):
        # Camera
        self.ui.pushButton_camera_connect.clicked.connect(self.camera_connect)
        self.ui.pushButton_camera_disconnet.clicked.connect(self.camera_disconnect)
        self.ui.combo_view.currentTextChanged.connect(self.change_display_mode)
        self.ui.pushButton_obj_1.clicked.connect(
            lambda: self.model1_3d(
                self.current_depth_frame,
                self.current_depth_intr,
                self.model1_class0_data,
                self.class2_centers_to_draw
            )
        )
        self.ui.pushButton_obj_2.clicked.connect(
            lambda: self.model2_3d(
                self.current_depth_frame,
                self.current_depth_intr,
                self.model2_data
            )
        )

        self.ui.pushButton_test_1.clicked.connect(
            lambda: self.model1_test(
                self.current_depth_frame,
                self.current_depth_intr,
                self.model1_class0_data,
                self.class2_centers_to_draw
            )
        )
        self.ui.pushButton_test_2.clicked.connect(
            lambda: self.model2_test(
                self.current_depth_frame,
                self.current_depth_intr,
                self.model2_data
            )
        )

        # Robot
        self.ui.pushButton_robot_connect.clicked.connect(self.robot_connect)
        self.ui.pushButton_robot_disconnect.clicked.connect(self.robot_disconnect)
        self.ui.pushButton_reset.clicked.connect(self.robot_reset)
        self.ui.pushButton_enable.clicked.connect(self.robot_enable)
        self.ui.pushButton_ref.clicked.connect(self.robot_reference)
        self.ui.slider_override.valueChanged.connect(self.robot_override_changed)
        self.ui.pushButton_joints.clicked.connect(self.robot_move_joints)
        self.ui.pushButton_joints_relative.clicked.connect(self.robot_move_joints_relative)
        self.ui.pushButton_joints_stop.clicked.connect(self.robot_stop_motion)
        self.ui.pushButton_cart_cart.clicked.connect(self.robot_move_cartesian)
        self.ui.pushButton_cart_relative.clicked.connect(self.robot_move_relative)
        self.ui.pushButton_cart_stop.clicked.connect(self.robot_stop_motion)
        self.ui.pushButton_fold.clicked.connect(self.robot_fold)
        self.ui.pushButton_program_home.clicked.connect(self.robot_move_home)
        self.ui.pushButton_gripper_close.clicked.connect(self.robot_close_gripper)
        self.ui.pushButton_gripper_open.clicked.connect(self.robot_open_gripper)
        self.ui.pushButton_cmd_send.clicked.connect(self.robot_send_custom_command)

        #Program
        self.ui.pushButton_program_single.clicked.connect(self.run_program_single)
        self.ui.pushButton_program_loop.clicked.connect(self.run_program_loop)
        self.ui.pushButton_program_pause.clicked.connect(self.pause_program)
        self.ui.pushButton_program_stop.clicked.connect(self.stop_program)
        self.ui.pushButton_program_step1.clicked.connect(self.run_program_step1)
        self.ui.pushButton_program_step2.clicked.connect(self.run_program_step2)

    def create_purple_to_red_colormap(self):
        colormap = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            t = i / 255.0
            r = int((1 - t) * 128 + t * 255)
            g = int((1 - t) * 0 + t * 0)
            b = int((1 - t) * 128 + t * 0)
            colormap[i, 0] = [b, g, r]
        return colormap

    def draw_obb(self, frame, depth_frame, depth_intrinsics, results, model_name, color):
        if not results or not hasattr(results[0], 'obb') or results[0].obb is None:
            print(f"Không có phát hiện OBB từ {model_name}")
            return frame, None, None, [], []

        h, w = frame.shape[:2]
        image_center = (w / 2, h / 2)

        model1_temp_data = []
        model2_temp_data = []

        # Thu thập dữ liệu hộp OBB
        try:
            for r in results:
                boxes = r.obb
                for box in boxes:
                    xywhr = box.xywhr[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    rect = cv2.boxPoints(((xywhr[0], xywhr[1]), (xywhr[2], xywhr[3]), xywhr[4] * 180 / np.pi))
                    rect = np.intp(rect)
                    center = (xywhr[0], xywhr[1])
                    distance = np.sqrt((center[0] - image_center[0]) ** 2 + (center[1] - image_center[1]) ** 2)

                    if model_name == "Model 1" and cls == 0:
                        model1_temp_data.append((xywhr, rect, distance))
                    elif model_name == "Model 2":
                        model2_temp_data.append((xywhr, rect, distance))
        except Exception as e:
            print(f"Lỗi khi thu thập hộp OBB từ {model_name}: {str(e)}")
            return frame, None, None, [], []

        # Sắp xếp theo khoảng cách
        model1_temp_data.sort(key=lambda x: x[2])
        model2_temp_data.sort(key=lambda x: x[2])

        model1_class0_data = []
        model1_class2_centers = {}
        model2_data = []
        current_index = 1

        # Gán index cho Model 1 lớp 0
        for xywhr, rect, _ in model1_temp_data:
            model1_class0_data.append((xywhr, rect, current_index))
            model1_class2_centers[current_index] = []
            print(f"Hộp lớp 0 index {current_index}: {rect}, góc xoay: {xywhr[4] * 180 / np.pi:.2f} độ")
            current_index += 1

        # Gán index cho Model 2
        for xywhr, rect, _ in model2_temp_data:
            model2_data.append((xywhr, rect, current_index))
            print(f"Hộp khớp nối index {current_index}: {rect}, góc xoay: {xywhr[4] * 180 / np.pi:.2f} độ")
            current_index += 1

        model1_first_center_3d = None
        model2_first_center_3d = None

        # Tính tọa độ 3D cho tâm hộp đầu tiên của Model 1
        if model1_temp_data:
            try:
                center = (model1_temp_data[0][0][0], model1_temp_data[0][0][1])
                x, y = int(center[0]), int(center[1])
                if 0 <= x < w and 0 <= y < h:
                    depth_value = depth_frame.get_distance(x, y)
                    if depth_value > 0:
                        model1_first_center_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics,
                                                                                 [center[0], center[1]], depth_value)
                        model1_first_center_3d = (
                            model1_first_center_3d[0] * 1000,
                            model1_first_center_3d[1] * 1000,
                            model1_first_center_3d[2] * 1000
                        )
                        label_3d = f"3D: ({model1_first_center_3d[0]:.1f}, {model1_first_center_3d[1]:.1f}, {model1_first_center_3d[2]:.1f}) mm"
                        cv2.putText(frame, label_3d, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        print(f"Tọa độ thực của tâm lớp 0 (Model 1, index 1): {model1_first_center_3d} mm")
                    else:
                        print(f"Giá trị độ sâu không hợp lệ tại tâm lớp 0 (Model 1, index 1): {center}")
            except Exception as e:
                print(f"Lỗi khi tính tọa độ 3D cho Model 1: {str(e)}")

        # Tính tọa độ 3D cho tâm hộp đầu tiên của Model 2
        if model2_temp_data:
            try:
                center = (model2_temp_data[0][0][0], model2_temp_data[0][0][1])
                x, y = int(center[0]), int(center[1])
                if 0 <= x < w and 0 <= y < h:
                    depth_value = depth_frame.get_distance(x, y)
                    if depth_value > 0:
                        model2_first_center_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics,
                                                                                 [center[0], center[1]], depth_value)
                        model2_first_center_3d = (
                            model2_first_center_3d[0] * 1000,
                            model2_first_center_3d[1] * 1000,
                            model2_first_center_3d[2] * 1000
                        )
                        label_3d = f"3D: ({model2_first_center_3d[0]:.1f}, {model2_first_center_3d[1]:.1f}, {model2_first_center_3d[2]:.1f}) mm"
                        cv2.putText(frame, label_3d, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        print(f"Tọa độ thực của tâm khớp nối (Model 2, index 1): {model2_first_center_3d} mm")
                    else:
                        print(f"Giá trị độ sâu không hợp lệ tại tâm khớp nối (Model 2, index 1): {center}")
            except Exception as e:
                print(f"Lỗi khi tính tọa độ 3D cho Model 2: {str(e)}")

        class2_centers_to_draw = []

        # Xử lý lớp 2 của Model 1
        if model_name == "Model 1":
            try:
                for r in results:
                    boxes = r.obb
                    for box in boxes:
                        xywhr = box.xywhr[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        if cls == 2:
                            center = (xywhr[0], xywhr[1])
                            rect = cv2.boxPoints(((xywhr[0], xywhr[1]), (xywhr[2], xywhr[3]), xywhr[4] * 180 / np.pi))
                            rect = np.intp(rect)
                            for _, poly, idx in model1_class0_data:
                                if cv2.pointPolygonTest(poly, center, True) >= -2.0:
                                    model1_class2_centers[idx].append(center)
                                    class2_centers_to_draw.append((center, f"", idx, xywhr, rect, None))
                                    print(f"Tâm lớp 2 {center} được gán vào index {idx}")
                                    break
            except Exception as e:
                print(f"Lỗi khi xử lý lớp 2 của Model 1: {str(e)}")
                return frame, model1_first_center_3d, model2_first_center_3d, model1_class0_data, []

        # Vẽ hộp OBB
        try:
            for r in results:
                boxes = r.obb
                for box in boxes:
                    xywhr = box.xywhr[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    rect = cv2.boxPoints(((xywhr[0], xywhr[1]), (xywhr[2], xywhr[3]), xywhr[4] * 180 / np.pi))
                    rect = np.intp(rect)
                    center = (xywhr[0], xywhr[1])

                    if model_name == "Model 1":
                        if cls == 0:
                            for xywhr_ref, _, index in model1_class0_data:
                                if np.allclose(xywhr, xywhr_ref, atol=1e-5):
                                    label = f""
                                    break
                            else:
                                label = ""
                            cv2.polylines(frame, [rect], True, color, 1)
                            cv2.circle(frame, (int(center[0]), int(center[1])), 3, color, -1)
                            cv2.putText(frame, label, (int(center[0]), int(center[1] - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        elif cls == 2:
                            index = next((idx for _, poly, idx in model1_class0_data
                                          if cv2.pointPolygonTest(poly, center, True) >= -2.0), None)
                            label = f"" if index is not None else ""
                            cv2.polylines(frame, [rect], True, color, 1)
                            cv2.circle(frame, (int(center[0]), int(center[1])), 3, color, -1)
                            cv2.putText(frame, label, (int(center[0]), int(center[1] - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    elif model_name == "Model 2":
                        for xywhr_ref, _, index in model2_data:
                            if np.allclose(xywhr, xywhr_ref, atol=1e-5):
                                label = f""
                                break
                        else:
                            label = ""
                        cv2.polylines(frame, [rect], True, color, 1)
                        cv2.circle(frame, (int(center[0]), int(center[1])), 3, color, -1)
                        cv2.putText(frame, label, (int(center[0]), int(center[1] - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            print(f"Lỗi khi vẽ hộp OBB: {str(e)}")
            return frame, model1_first_center_3d, model2_first_center_3d, model1_class0_data if model_name == "Model 1" else model2_data, class2_centers_to_draw

        # Xử lý điểm trắng cho Model 1
        if model_name == "Model 1":
            updated_class2_centers = []
            try:
                for index, centers in model1_class2_centers.items():
                    if len(centers) == 2:
                        x1, y1 = centers[0]
                        x2, y2 = centers[1]
                        if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
                            print(f"Tọa độ tâm lớp 2 không hợp lệ cho index {index}")
                            continue

                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                        cv2.circle(frame, (int(mid_x), int(mid_y)), 3, (255, 0, 0), -1)

                        box_center_x, box_center_y = None, None
                        for xywhr, _, idx in model1_class0_data:
                            if idx == index:
                                box_center_x, box_center_y = xywhr[0], xywhr[1]
                                break
                        if box_center_x is None or not (0 <= box_center_x < w and 0 <= box_center_y < h):
                            print(f"Tọa độ tâm hộp lớp 0 không hợp lệ for index {index}")
                            continue

                        cv2.line(frame, (int(mid_x), int(mid_y)), (int(box_center_x), int(box_center_y)), (0, 255, 0),
                                 1)

                        ray_vec = (box_center_x - mid_x, box_center_y - mid_y)
                        cross1 = (x1 - mid_x) * ray_vec[1] - (y1 - mid_y) * ray_vec[0]
                        cross2 = (x2 - mid_x) * ray_vec[1] - (y2 - mid_y) * ray_vec[0]
                        color1 = (0, 0, 255) if cross1 > 0 else (255, 255, 0)
                        color2 = (0, 0, 255) if cross2 > 0 else (255, 255, 0)

                        for center, label, idx, xywhr, rect, white_point in class2_centers_to_draw:
                            if idx == index and center in [(x1, y1), (x2, y2)]:
                                center_color = color1 if center == (x1, y1) else color2
                                cv2.circle(frame, (int(center[0]), int(center[1])), 3, center_color, -1)
                                cv2.putText(frame, label, (int(center[0]), int(center[1] - 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, center_color, 2)
                                new_white_point = white_point
                                if center_color == (255, 255, 0):
                                    rotation = xywhr[4] * 180 / np.pi
                                    width, height = xywhr[2], xywhr[3]
                                    is_width_longer = width >= height
                                    long_side = width if is_width_longer else height
                                    short_side = height if is_width_longer else width

                                    cos_a = np.sin(np.radians(rotation)) if is_width_longer else np.cos(
                                        np.radians(rotation))
                                    sin_a = -np.cos(np.radians(rotation)) if is_width_longer else np.sin(
                                        np.radians(rotation))

                                    half_short_side = short_side / 2
                                    p1_x = xywhr[0] - half_short_side * cos_a
                                    p1_y = xywhr[1] - half_short_side * sin_a
                                    p2_x = xywhr[0] + half_short_side * cos_a
                                    p2_y = xywhr[1] + half_short_side * sin_a

                                    if not (0 <= p1_x < w and 0 <= p1_y < h and 0 <= p2_x < w and 0 <= p2_y < h):
                                        print(
                                            f"Tọa độ đường chia đôi không hợp lệ: ({p1_x}, {p1_y}) hoặc ({p2_x}, {p2_y})")
                                        updated_class2_centers.append((center, label, idx, xywhr, rect, white_point))
                                        continue

                                    cv2.line(frame, (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y)), (128, 0, 128), 1)
                                    print(
                                        f"Vẽ đường chia đôi hộp tại tâm ({center[0]}, {center[1]}): từ ({p1_x}, {p1_y}) đến ({p2_x}, {p2_y})")

                                    other_center = next(c for c in centers if c != center)
                                    box_points = rect.astype(np.float32)
                                    box_points = np.array(
                                        sorted(box_points, key=lambda p: np.arctan2(p[1] - xywhr[1], p[0] - xywhr[0])))

                                    cross_products = []
                                    line_vec = (p2_x - p1_x, p2_y - p1_y)
                                    for pt in box_points:
                                        vec_to_pt = (pt[0] - p1_x, pt[1] - p1_y)
                                        cross = vec_to_pt[0] * line_vec[1] - vec_to_pt[1] * line_vec[0]
                                        cross_products.append(cross)

                                    half1_points = []
                                    half2_points = []
                                    for i in range(4):
                                        if cross_products[i] >= 0:
                                            half1_points.append(box_points[i])
                                        else:
                                            half2_points.append(box_points[i])

                                    half1_points.extend([(p1_x, p1_y), (p2_x, p2_y)])
                                    half2_points.extend([(p1_x, p1_y), (p2_x, p2_y)])

                                    def sort_polygon_points(points):
                                        if len(points) < 3:
                                            return points
                                        centroid = np.mean(points, axis=0)
                                        return sorted(points,
                                                      key=lambda p: np.arctan2(p[1] - centroid[1], p[0] - centroid[0]))

                                    half1_points = np.array(sort_polygon_points(half1_points), dtype=np.float32)
                                    half2_points = np.array(sort_polygon_points(half2_points), dtype=np.float32)

                                    origin_in_part1 = cv2.pointPolygonTest(half1_points, other_center,
                                                                           False) >= 0 if len(
                                        half1_points) >= 3 else False
                                    origin_in_part2 = cv2.pointPolygonTest(half2_points, other_center,
                                                                           False) >= 0 if len(
                                        half2_points) >= 3 else False

                                    cross_to_other = (other_center[0] - p1_x) * line_vec[1] - (other_center[1] - p1_y) * \
                                                     line_vec[0]
                                    new_white_point = None
                                    if not origin_in_part1 and len(half1_points) >= 3 and cross_to_other < 0:
                                        part1_center = np.mean(half1_points, axis=0)
                                        vector_to_center = part1_center - np.array(center)
                                        new_white_point = np.array(center) + 1.0 * vector_to_center
                                        if 0 <= new_white_point[0] < w and 0 <= new_white_point[1] < h:
                                            cv2.circle(frame, tuple(new_white_point.astype(np.int32)), 3,
                                                       (255, 255, 255), -1)
                                            cv2.line(frame, tuple(new_white_point.astype(np.int32)),
                                                     tuple(np.array(center).astype(np.int32)), (255, 255, 255), 1)
                                            print(
                                                f"Vẽ chấm trắng tại {new_white_point} cho nửa 1 không chứa tâm xanh lá {other_center}")
                                        else:
                                            print(f"Chấm trắng tại {new_white_point} ngoài khung hình")
                                    elif not origin_in_part2 and len(half2_points) >= 3 and cross_to_other >= 0:
                                        part2_center = np.mean(half2_points, axis=0)
                                        vector_to_center = part2_center - np.array(center)
                                        new_white_point = np.array(center) + 1.05 * vector_to_center
                                        if 0 <= new_white_point[0] < w and 0 <= new_white_point[1] < h:
                                            cv2.circle(frame, tuple(new_white_point.astype(np.int32)), 3,
                                                       (255, 255, 255), -1)
                                            cv2.line(frame, tuple(new_white_point.astype(np.int32)),
                                                     tuple(np.array(center).astype(np.int32)), (255, 255, 255), 1)
                                            print(
                                                f"Vẽ chấm trắng tại {new_white_point} cho nửa 2 không chứa tâm xanh lá {other_center}")
                                        else:
                                            print(f"Chấm trắng tại {new_white_point} ngoài khung hình")

                                updated_class2_centers.append((center, label, idx, xywhr, rect, new_white_point))
                    else:
                        updated_class2_centers.extend([(center, label, idx, xywhr, rect, white_point)
                                                       for center, label, idx, xywhr, rect, white_point in
                                                       class2_centers_to_draw
                                                       if idx == index])
            except Exception as e:
                print(f"Lỗi khi xử lý điểm trắng cho Model 1: {str(e)}")
                return frame, model1_first_center_3d, model2_first_center_3d, model1_class0_data, class2_centers_to_draw

            class2_centers_to_draw = updated_class2_centers

        try:
            cv2.imwrite(f"debug_frame_{model_name}.jpg", frame)
        except Exception as e:
            print(f"Lỗi khi lưu ảnh debug cho {model_name}: {str(e)}")

        return frame, model1_first_center_3d, model2_first_center_3d, model1_class0_data if model_name == "Model 1" else model2_data, class2_centers_to_draw

    @staticmethod
    def display_open3d(point_cloud, obb, aabb, coordinate_frame, obb_coordinate_frame):
        """
        Hiển thị các đối tượng Open3D trong cửa sổ riêng, bao gồm đám mây điểm, OBB, AABB,
        hệ trục tọa độ gốc và hệ trục tọa độ của OBB.
        """
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Đám mây điểm 3D - Cụm lớn nhất với OBB, AABB và các hệ trục", width=1280,
                              height=720)

            # Thêm các đối tượng
            vis.add_geometry(point_cloud)
            vis.add_geometry(obb)
            vis.add_geometry(aabb)
            vis.add_geometry(coordinate_frame)
            vis.add_geometry(obb_coordinate_frame)

            # Tùy chỉnh view
            view_control = vis.get_view_control()
            view_control.set_front([0, 0, -1])
            view_control.set_lookat(np.mean(np.asarray(point_cloud.points), axis=0))
            view_control.set_up([0, 1, 0])
            view_control.set_zoom(0.5)

            # Tùy chỉnh render
            render_option = vis.get_render_option()
            render_option.point_size = 2.0
            render_option.line_width = 1.0

            vis.run()
            vis.destroy_window()
            print("Đóng cửa sổ Open3D thành công")
        except Exception as e:
            print(f"Lỗi khi hiển thị Open3D trong luồng riêng: {str(e)}")

    def model1_3d(self, depth_frame, depth_intrinsics, model1_class0_data, class2_centers_to_draw):
        try:
            # Debug: Log input data
            print(f"model1_class0_data: {len(model1_class0_data)} items")
            print(f"class2_centers_to_draw: {len(class2_centers_to_draw)} items")

            # Tìm hộp lớp 2 màu vàng (cyan) với index 1
            target_center = None
            target_rect = None
            target_white_point = None
            for center, label, idx, xywhr, rect, white_point in class2_centers_to_draw:
                if idx == 1:
                    box_center_x, box_center_y = None, None
                    for xywhr_class0, _, class0_idx in model1_class0_data:
                        if class0_idx == idx:
                            box_center_x, box_center_y = xywhr_class0[0], xywhr_class0[1]
                            break
                    if box_center_x is None:
                        print(f"Không tìm thấy hộp lớp 0 với index {idx}")
                        continue

                    centers = [c[0] for c in class2_centers_to_draw if c[2] == idx]
                    if len(centers) != 2:
                        print(f"Yêu cầu đúng 2 tâm lớp 2 cho index 1, tìm thấy {len(centers)}")
                        continue

                    x1, y1 = centers[0]
                    x2, y2 = centers[1]
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    ray_vec = (box_center_x - mid_x, box_center_y - mid_y)
                    cross = (center[0] - mid_x) * ray_vec[1] - (center[1] - mid_y) * ray_vec[0]
                    if cross <= 0:  # Màu vàng (cyan)
                        target_center = center
                        target_rect = rect
                        target_white_point = white_point
                        print(f"Found yellow box: center={center}, index={idx}, white_point={white_point}")
                        break

            if target_center is None or target_rect is None:
                for center, label, idx, xywhr, rect, white_point in class2_centers_to_draw:
                    if idx == 1:
                        target_center = center
                        target_rect = rect
                        target_white_point = white_point
                        print(
                            f"Không tìm thấy hộp màu vàng, chọn hộp lớp 2 mặc định: center={center}, index={idx}, white_point={white_point}")
                        break

            if target_center is None or target_rect is None:
                print("Lỗi: Không tìm thấy hộp lớp 2 với index 1")
                return False, None, None

            h, w = depth_frame.get_height(), depth_frame.get_width()
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [target_rect], 255)
            depth_data = np.asanyarray(depth_frame.get_data())

            points = []
            for y in range(h):
                for x in range(w):
                    if mask[y, x] == 255:
                        depth_value = depth_data[y, x] / 1000.0
                        if 0 < depth_value < 5.0:
                            point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth_value)
                            points.append(point_3d)

            if not points:
                print("Lỗi: Không có điểm 3D hợp lệ trong vùng hộp")
                return False, None, None

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(np.array(points))

            eps = 0.005
            min_points = 100
            labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

            if len(labels) == 0 or np.all(labels == -1):
                print("Lỗi: Không tìm thấy cụm hợp lệ sau DBSCAN")
                return False, None, None

            cluster_sizes = {label: np.sum(labels == label) for label in np.unique(labels) if label >= 0}
            if not cluster_sizes:
                print("Lỗi: Không có cụm nào được hình thành")
                return False, None, None

            largest_cluster_label = max(cluster_sizes, key=cluster_sizes.get)
            largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
            largest_cluster_points = np.array(points)[largest_cluster_indices]

            largest_point_cloud = o3d.geometry.PointCloud()
            largest_point_cloud.points = o3d.utility.Vector3dVector(largest_cluster_points)

            voxel_size = 0.001
            downsampled_point_cloud = largest_point_cloud.voxel_down_sample(voxel_size)
            colors = np.zeros((len(np.asarray(downsampled_point_cloud.points)), 3))
            colors[:, 0] = 1.0
            downsampled_point_cloud.colors = o3d.utility.Vector3dVector(colors)

            obb = downsampled_point_cloud.get_oriented_bounding_box()
            obb.color = (0, 0, 1)
            aabb = downsampled_point_cloud.get_axis_aligned_bounding_box()
            aabb.color = (0, 1, 0)
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

            # Tính white_point_3d
            white_point_3d = None
            if target_white_point is not None:
                if 0 <= target_white_point[0] < w and 0 <= target_white_point[1] < h:
                    depth_value = depth_frame.get_distance(int(target_white_point[0]), int(target_white_point[1]))
                    if depth_value > 0:
                        white_point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics,
                                                                         [target_white_point[0], target_white_point[1]],
                                                                         depth_value)
                        print(f"White point 3D tính được: {white_point_3d}")
                    else:
                        print(f"Lỗi: Độ sâu tại white_point ({target_white_point}) không hợp lệ: {depth_value}")
                else:
                    print(f"Lỗi: White point 2D ({target_white_point}) nằm ngoài khung hình ({w}x{h})")
            else:
                print("Lỗi: target_white_point là None, không thể tính white_point_3d")

            # Chọn obb_origin
            obb_origin = white_point_3d if white_point_3d is not None else obb.center
            if white_point_3d is None:
                print("Cảnh báo: Sử dụng obb.center làm obb_origin vì white_point_3d không hợp lệ")

            # Chuyển obb_origin và obb.center thành mảng numpy
            obb_origin = np.array(obb_origin)
            obb_center = np.array(obb.center)
            print(f"obb_origin (white_point_3d hoặc obb.center): {obb_origin}")
            print(f"obb.center: {obb_center}")

            # Tạo hệ trục tọa độ cho OBB tại obb_origin
            obb_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=obb_origin)

            # Lấy vector trục z của hệ OBB trong không gian thế giới
            z_axis_obb = obb.R[:, 2]  # Cột thứ 3 của ma trận xoay là trục z

            # Ma trận điều chỉnh (mặc định là ma trận đơn vị)
            adjustment_matrix = np.eye(3)

            # Bước 1: Tính toán điều chỉnh trục z
            if z_axis_obb[2] < 0:  # Trục z của OBB hướng xuống (phần âm)
                z_reflection_matrix = np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, -1]
                ])
                adjustment_matrix = np.dot(adjustment_matrix, z_reflection_matrix)
                print("Cần đảo ngược trục z vì trục z hướng xuống phần âm")
            else:
                print("Trục z đã hướng lên phần dương hoặc trung lập, không cần điều chỉnh")

            # Tính ma trận xoay tạm thời sau khi điều chỉnh trục z
            temp_rotation_matrix = np.dot(obb.R, adjustment_matrix)

            # Bước 2: Tính toán điều chỉnh trục x dựa trên vị trí tâm OBB
            obb_center_local = np.dot(np.linalg.inv(temp_rotation_matrix), obb_center - obb_origin)
            print(f"Tâm OBB trong hệ trục tại obb_origin trước điều chỉnh x: x_local = {obb_center_local[0]:.4f}")

            # Nếu tâm OBB nằm ở phần âm của trục x trong hệ OBB, áp dụng đối xứng
            if obb_center_local[0] < 0:
                x_reflection_matrix_local = np.array([
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ])
                adjustment_matrix = np.dot(adjustment_matrix, x_reflection_matrix_local)
                print("Cần đảo ngược trục x vì tâm OBB nằm ở phần âm của trục x tại obb_origin")
            else:
                print("Tâm OBB đã nằm ở phần dương hoặc trên trục x tại obb_origin, không cần điều chỉnh")

            # Áp dụng ma trận xoay cuối cùng lên obb_coordinate_frame
            final_rotation_matrix = np.dot(obb.R, adjustment_matrix)
            obb_coordinate_frame.rotate(final_rotation_matrix, center=obb_origin)

            # Kiểm tra hướng trục z sau khi điều chỉnh
            adjusted_z_axis = (final_rotation_matrix @ np.array([0, 0, 1]))[2]
            print(f"Hướng trục z sau điều chỉnh: z = {adjusted_z_axis:.4f}")
            if adjusted_z_axis > 0:
                print("Xác nhận: Trục z của OBB hướng lên phần dương, đáp ứng yêu cầu")
            else:
                print("Cảnh báo: Trục z của OBB vẫn không hướng lên phần dương, cần kiểm tra thêm")

            # Kiểm tra tâm OBB trong hệ trục OBB sau điều chỉnh
            obb_center_local_final = np.dot(np.linalg.inv(final_rotation_matrix), obb_center - obb_origin)
            print(f"Tâm OBB trong hệ trục tại obb_origin sau điều chỉnh: x_local = {obb_center_local_final[0]:.4f}")
            if obb_center_local_final[0] >= 0:
                print("Xác nhận: Tâm OBB nằm ở phần dương hoặc trên trục x trong hệ trục tại obb_origin")
            else:
                print("Cảnh báo: Tâm OBB vẫn nằm ở phần âm của trục x trong hệ trục tại obb_origin, cần kiểm tra thêm")

            print("Hệ trục tọa độ OBB được tạo và điều chỉnh tại vị trí white_point_3d hoặc obb.center")

            # Tính góc quay Euler từ ma trận xoay cuối cùng
            det = np.linalg.det(final_rotation_matrix)
            if det < 0:
                final_rotation_matrix[:, 2] *= -1
                print("Đã sửa ma trận xoay để chuyển từ hệ trái tay sang hệ phải tay")
            rotation = Rotation.from_matrix(final_rotation_matrix)
            euler_angles_1 = rotation.as_euler('xyz', degrees=True)
            white_point_mm = np.array(obb_origin) * 1000

            print(
                f"White point coordinates (mm): ({white_point_mm[0]:.2f}, {white_point_mm[1]:.2f}, {white_point_mm[2]:.2f})")
            print(
                f"Euler angles (degrees): Roll={euler_angles_1[0]:.2f}, Pitch={euler_angles_1[1]:.2f}, Yaw={euler_angles_1[2]:.2f}")
            self.append_log(
                f"Model 1: X={white_point_mm[0]:.2f} mm, Y={white_point_mm[1]:.2f} mm, Z={white_point_mm[2]:.2f} mm")
            self.append_log(
                f"Model 1: Roll={euler_angles_1[0]:.2f}°, Pitch={euler_angles_1[1]:.2f}°, Yaw={euler_angles_1[2]:.2f}°")

            # def display_open3d():
            #     vis = o3d.visualization.Visualizer()
            #     vis.create_window(window_name="Đám mây điểm 3D", width=1280, height=720)
            #     vis.add_geometry(downsampled_point_cloud)
            #     vis.add_geometry(obb)
            #     vis.add_geometry(aabb)
            #     vis.add_geometry(coordinate_frame)
            #     vis.add_geometry(obb_coordinate_frame)
            #     view_control = vis.get_view_control()
            #     view_control.set_front([0, 0, -1])
            #     view_control.set_lookat(np.mean(np.asarray(downsampled_point_cloud.points), axis=0))
            #     view_control.set_up([0, 1, 0])
            #     view_control.set_zoom(0.5)
            #     render_option = vis.get_render_option()
            #     render_option.point_size = 2.0
            #     render_option.line_width = 1.0
            #     vis.run()
            #     vis.destroy_window()
            #
            # thread = threading.Thread(target=display_open3d)
            # thread.daemon = True
            # thread.start()

            return True, white_point_mm, euler_angles_1

        except Exception as e:
            print(f"Lỗi khi hiển thị đám mây điểm 3D: {str(e)}")
            self.append_log(f"Lỗi khi hiển thị đám mây điểm 3D: {str(e)}")
            return False, None, None

    def model2_3d(self, depth_frame, depth_intrinsics, model2_data):
        """
        Hiển thị đám mây điểm 3D của cụm lớn nhất từ hộp OBB 2D của Model 2 index 1,
        """
        try:
            # Debug: Log input data
            print(f"model2_data: {len(model2_data)} items")

            # Tìm hộp OBB 2D của Model 2 với index 1
            target_center = None
            target_rect = None
            for xywhr, rect, idx in model2_data:
                if idx == 1:
                    target_center = (xywhr[0], xywhr[1])
                    target_rect = rect
                    print(f"Found Model 2 box: center={target_center}, index={idx}")
                    break

            if target_center is None or target_rect is None:
                print("Lỗi: Không tìm thấy hộp Model 2 với index 1")
                return False, None, None

            # Lấy kích thước khung hình
            h, w = depth_frame.get_height(), depth_frame.get_width()

            # Tạo mask cho vùng hộp OBB 2D
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [target_rect], 255)

            # Lấy dữ liệu độ sâu
            depth_data = np.asanyarray(depth_frame.get_data())

            # Tạo đám mây điểm
            points = []
            for y in range(h):
                for x in range(w):
                    if mask[y, x] == 255:
                        depth_value = depth_data[y, x] / 1000.0  # Chuyển từ mm sang m
                        if 0 < depth_value < 5.0:  # Lọc độ sâu hợp lệ
                            point_3d = rs.rs2_deproject_pixel_to_point(
                                depth_intrinsics, [x, y], depth_value
                            )
                            points.append(point_3d)

            if not points:
                print("Lỗi: Không có điểm 3D hợp lệ trong vùng hộp")
                return False, None, None

            # Log số lượng điểm ban đầu
            print(f"Số lượng điểm 3D ban đầu: {len(points)}")

            # Tạo đối tượng PointCloud trong Open3D
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(np.array(points))

            # Phân cụm bằng DBSCAN
            eps = 0.005  # 5mm
            min_points = 100
            labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

            if len(labels) == 0 or np.all(labels == -1):
                print("Lỗi: Không tìm thấy cụm hợp lệ sau DBSCAN")
                return False, None, None

            # Tìm cụm lớn nhất
            cluster_sizes = {label: np.sum(labels == label) for label in np.unique(labels) if label >= 0}
            if not cluster_sizes:
                print("Lỗi: Không có cụm nào được hình thành")
                return False, None, None

            largest_cluster_label = max(cluster_sizes, key=cluster_sizes.get)
            largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
            largest_cluster_points = np.array(points)[largest_cluster_indices]

            # Log số lượng điểm trong cụm lớn nhất
            print(f"Số lượng điểm trong cụm lớn nhất: {len(largest_cluster_points)}")

            # Tạo PointCloud mới cho cụm lớn nhất
            largest_point_cloud = o3d.geometry.PointCloud()
            largest_point_cloud.points = o3d.utility.Vector3dVector(largest_cluster_points)

            # Giảm mẫu đám mây điểm
            voxel_size = 0.001  # 1mm
            downsampled_point_cloud = largest_point_cloud.voxel_down_sample(voxel_size)
            colors = np.zeros((len(np.asarray(downsampled_point_cloud.points)), 3))
            colors[:, 0] = 1.0  # Màu đỏ
            downsampled_point_cloud.colors = o3d.utility.Vector3dVector(colors)

            # Log số lượng điểm sau giảm mẫu
            print(f"Số lượng điểm sau giảm mẫu: {len(np.asarray(downsampled_point_cloud.points))}")

            # Tạo hộp OBB 3D
            obb = downsampled_point_cloud.get_oriented_bounding_box()
            obb.color = (0, 0, 1)  # Xanh lam

            # Tạo hộp AABB
            aabb = downsampled_point_cloud.get_axis_aligned_bounding_box()
            aabb.color = (0, 1, 0)  # Xanh lá

            # Tạo hệ trục tọa độ gốc
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

            # Tính tọa độ 3D của tâm hộp OBB 2D
            center_2d = target_center
            if 0 <= center_2d[0] < w and 0 <= center_2d[1] < h:
                depth_value = depth_frame.get_distance(int(center_2d[0]), int(center_2d[1]))
                if depth_value > 0:
                    center_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [center_2d[0], center_2d[1]],
                                                                depth_value)
                    print(f"Center 3D (m): {center_3d}")
                else:
                    print(f"Giá trị độ sâu không hợp lệ tại center: {center_2d}")
                    center_3d = obb.center
            else:
                print(f"Tọa độ tâm ngoài khung hình: {center_2d}")
                center_3d = obb.center

            # Tạo hệ trục tọa độ cho OBB tại center_3d
            obb_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=center_3d)
            obb_coordinate_frame.rotate(obb.R, center=center_3d)

            # Lấy vector trục z của hệ OBB trong không gian thế giới
            z_axis_obb = obb.R[:, 2]  # Cột thứ 3 của ma trận xoay là trục z

            # Kiểm tra hướng trục z của OBB so với hệ tọa độ thế giới
            if z_axis_obb[2] < 0:  # Trục z của OBB hướng lên (phần dương)
                # Tạo ma trận đối xứng qua mặt phẳng xy để đảo ngược trục z
                reflection_matrix = np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, -1]
                ])
                obb_coordinate_frame.rotate(reflection_matrix, center=center_3d)
                print("Đã đảo ngược trục z của OBB bằng phép đối xứng vì trục z hướng lên phần dương")
            else:
                print("Giữ nguyên hệ trục OBB vì trục z đã hướng xuống phần âm hoặc trung lập")

            # Kiểm tra lại hướng trục z sau khi điều chỉnh
            adjusted_z_axis = (obb_coordinate_frame.get_rotation_matrix_from_xyz((0, 0, 0)) @ np.array([0, 0, 1]))[2]
            print(f"Hướng trục z sau điều chỉnh: z = {adjusted_z_axis:.4f}")
            if adjusted_z_axis < 0:
                print("Xác nhận: Trục z của OBB hướng xuống phần âm, đáp ứng yêu cầu")
            else:
                print("Cảnh báo: Trục z của OBB vẫn không hướng xuống phần âm, cần kiểm tra thêm")

            print("Hệ trục tọa độ OBB được tạo và điều chỉnh theo yêu cầu")

            # Tính góc quay Euler từ ma trận xoay của OBB
            rotation = Rotation.from_matrix(obb.R)
            euler_angles_2 = rotation.as_euler('xyz', degrees=True)

            # Chuyển center_3d sang mm
            center_mm = np.array(center_3d) * 1000

            # In tọa độ và góc Euler vào log
            print(f"Center coordinates (mm): ({center_mm[0]:.2f}, {center_mm[1]:.2f}, {center_mm[2]:.2f})")
            print(
                f"Euler angles (degrees): Roll={euler_angles_2[0]:.2f}, Pitch={euler_angles_2[1]:.2f}, Yaw={euler_angles_2[2]:.2f}")
            self.append_log(f"Model 2: X={center_mm[0]:.2f}mm, Y={center_mm[1]:.2f}mm, Z={center_mm[2]:.2f}mm")
            self.append_log(
                f"Model 2: Roll={euler_angles_2[0]:.2f}°, Pitch={euler_angles_2[1]:.2f}°, Yaw={euler_angles_2[2]:.2f}°")

            # Hiển thị Open3D trong luồng riêng
            # def display_open3d():
            #     try:
            #         print("Khởi tạo cửa sổ Open3D trong luồng riêng...")
            #         vis = o3d.visualization.Visualizer()
            #         vis.create_window(window_name="Đám mây điểm 3D - Model 2 Index 1", width=1280, height=720)
            #         vis.add_geometry(downsampled_point_cloud)
            #         vis.add_geometry(obb)
            #         vis.add_geometry(aabb)
            #         vis.add_geometry(coordinate_frame)
            #         vis.add_geometry(obb_coordinate_frame)
            #         view_control = vis.get_view_control()
            #         view_control.set_front([0, 0, -1])
            #         view_control.set_lookat(np.mean(np.asarray(downsampled_point_cloud.points), axis=0))
            #         view_control.set_up([0, 1, 0])
            #         view_control.set_zoom(0.5)
            #         render_option = vis.get_render_option()
            #         render_option.point_size = 2.0
            #         render_option.line_width = 1.0
            #         print("Hiển thị cửa sổ Open3D...")
            #         vis.run()
            #         print("Đóng cửa sổ Open3D...")
            #         vis.destroy_window()
            #     except Exception as e:
            #         print(f"Lỗi khi hiển thị Open3D trong luồng riêng: {str(e)}")
            #
            # thread = threading.Thread(target=display_open3d)
            # thread.daemon = True
            # thread.start()
            # print("Đã khởi động luồng hiển thị Open3D")

            return True, center_mm, euler_angles_2

        except Exception as e:
            print(f"Lỗi khi hiển thị đám mây điểm 3D cho Model 2: {str(e)}")
            return False, None, None

    def model1_test(self, depth_frame, depth_intrinsics, model1_class0_data, class2_centers_to_draw):
        try:
            # Debug: Log input data
            print(f"model1_class0_data: {len(model1_class0_data)} items")
            print(f"class2_centers_to_draw: {len(class2_centers_to_draw)} items")

            # Tìm hộp lớp 2 màu vàng (cyan) với index 1
            target_center = None
            target_rect = None
            target_white_point = None
            for center, label, idx, xywhr, rect, white_point in class2_centers_to_draw:
                if idx == 1:
                    box_center_x, box_center_y = None, None
                    for xywhr_class0, _, class0_idx in model1_class0_data:
                        if class0_idx == idx:
                            box_center_x, box_center_y = xywhr_class0[0], xywhr_class0[1]
                            break
                    if box_center_x is None:
                        print(f"Không tìm thấy hộp lớp 0 với index {idx}")
                        continue

                    centers = [c[0] for c in class2_centers_to_draw if c[2] == idx]
                    if len(centers) != 2:
                        print(f"Yêu cầu đúng 2 tâm lớp 2 cho index 1, tìm thấy {len(centers)}")
                        continue

                    x1, y1 = centers[0]
                    x2, y2 = centers[1]
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    ray_vec = (box_center_x - mid_x, box_center_y - mid_y)
                    cross = (center[0] - mid_x) * ray_vec[1] - (center[1] - mid_y) * ray_vec[0]
                    if cross <= 0:  # Màu vàng (cyan)
                        target_center = center
                        target_rect = rect
                        target_white_point = white_point
                        print(f"Found yellow box: center={center}, index={idx}, white_point={white_point}")
                        break

            if target_center is None or target_rect is None:
                for center, label, idx, xywhr, rect, white_point in class2_centers_to_draw:
                    if idx == 1:
                        target_center = center
                        target_rect = rect
                        target_white_point = white_point
                        print(
                            f"Không tìm thấy hộp màu vàng, chọn hộp lớp 2 mặc định: center={center}, index={idx}, white_point={white_point}")
                        break

            if target_center is None or target_rect is None:
                print("Lỗi: Không tìm thấy hộp lớp 2 với index 1")
                return False, None, None

            h, w = depth_frame.get_height(), depth_frame.get_width()
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [target_rect], 255)
            depth_data = np.asanyarray(depth_frame.get_data())

            points = []
            for y in range(h):
                for x in range(w):
                    if mask[y, x] == 255:
                        depth_value = depth_data[y, x] / 1000.0
                        if 0 < depth_value < 5.0:
                            point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth_value)
                            points.append(point_3d)

            if not points:
                print("Lỗi: Không có điểm 3D hợp lệ trong vùng hộp")
                return False, None, None

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(np.array(points))

            eps = 0.005
            min_points = 100
            labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

            if len(labels) == 0 or np.all(labels == -1):
                print("Lỗi: Không tìm thấy cụm hợp lệ sau DBSCAN")
                return False, None, None

            cluster_sizes = {label: np.sum(labels == label) for label in np.unique(labels) if label >= 0}
            if not cluster_sizes:
                print("Lỗi: Không có cụm nào được hình thành")
                return False, None, None

            largest_cluster_label = max(cluster_sizes, key=cluster_sizes.get)
            largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
            largest_cluster_points = np.array(points)[largest_cluster_indices]

            largest_point_cloud = o3d.geometry.PointCloud()
            largest_point_cloud.points = o3d.utility.Vector3dVector(largest_cluster_points)

            voxel_size = 0.001
            downsampled_point_cloud = largest_point_cloud.voxel_down_sample(voxel_size)
            colors = np.zeros((len(np.asarray(downsampled_point_cloud.points)), 3))
            colors[:, 0] = 1.0
            downsampled_point_cloud.colors = o3d.utility.Vector3dVector(colors)

            obb = downsampled_point_cloud.get_oriented_bounding_box()
            obb.color = (0, 0, 1)
            aabb = downsampled_point_cloud.get_axis_aligned_bounding_box()
            aabb.color = (0, 1, 0)
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

            # Tính white_point_3d
            white_point_3d = None
            if target_white_point is not None:
                if 0 <= target_white_point[0] < w and 0 <= target_white_point[1] < h:
                    depth_value = depth_frame.get_distance(int(target_white_point[0]), int(target_white_point[1]))
                    if depth_value > 0:
                        white_point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics,
                                                                         [target_white_point[0], target_white_point[1]],
                                                                         depth_value)
                        print(f"White point 3D tính được: {white_point_3d}")
                    else:
                        print(f"Lỗi: Độ sâu tại white_point ({target_white_point}) không hợp lệ: {depth_value}")
                else:
                    print(f"Lỗi: White point 2D ({target_white_point}) nằm ngoài khung hình ({w}x{h})")
            else:
                print("Lỗi: target_white_point là None, không thể tính white_point_3d")

            # Chọn obb_origin
            obb_origin = white_point_3d if white_point_3d is not None else obb.center
            if white_point_3d is None:
                print("Cảnh báo: Sử dụng obb.center làm obb_origin vì white_point_3d không hợp lệ")

            # Chuyển obb_origin và obb.center thành mảng numpy
            obb_origin = np.array(obb_origin)
            obb_center = np.array(obb.center)
            print(f"obb_origin (white_point_3d hoặc obb.center): {obb_origin}")
            print(f"obb.center: {obb_center}")

            # Tạo hệ trục tọa độ cho OBB tại obb_origin
            obb_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=obb_origin)

            # Lấy vector trục z của hệ OBB trong không gian thế giới
            z_axis_obb = obb.R[:, 2]  # Cột thứ 3 của ma trận xoay là trục z

            # Ma trận điều chỉnh (mặc định là ma trận đơn vị)
            adjustment_matrix = np.eye(3)

            # Bước 1: Tính toán điều chỉnh trục z
            if z_axis_obb[2] < 0:  # Trục z của OBB hướng xuống (phần âm)
                z_reflection_matrix = np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, -1]
                ])
                adjustment_matrix = np.dot(adjustment_matrix, z_reflection_matrix)
                print("Cần đảo ngược trục z vì trục z hướng xuống phần âm")
            else:
                print("Trục z đã hướng lên phần dương hoặc trung lập, không cần điều chỉnh")

            # Tính ma trận xoay tạm thời sau khi điều chỉnh trục z
            temp_rotation_matrix = np.dot(obb.R, adjustment_matrix)

            # Bước 2: Tính toán điều chỉnh trục x dựa trên vị trí tâm OBB
            obb_center_local = np.dot(np.linalg.inv(temp_rotation_matrix), obb_center - obb_origin)
            print(f"Tâm OBB trong hệ trục tại obb_origin trước điều chỉnh x: x_local = {obb_center_local[0]:.4f}")

            # Nếu tâm OBB nằm ở phần âm của trục x trong hệ OBB, áp dụng đối xứng
            if obb_center_local[0] < 0:
                x_reflection_matrix_local = np.array([
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ])
                adjustment_matrix = np.dot(adjustment_matrix, x_reflection_matrix_local)
                print("Cần đảo ngược trục x vì tâm OBB nằm ở phần âm của trục x tại obb_origin")
            else:
                print("Tâm OBB đã nằm ở phần dương hoặc trên trục x tại obb_origin, không cần điều chỉnh")

            # Áp dụng ma trận xoay cuối cùng lên obb_coordinate_frame
            final_rotation_matrix = np.dot(obb.R, adjustment_matrix)
            obb_coordinate_frame.rotate(final_rotation_matrix, center=obb_origin)

            # Kiểm tra hướng trục z sau khi điều chỉnh
            adjusted_z_axis = (final_rotation_matrix @ np.array([0, 0, 1]))[2]
            print(f"Hướng trục z sau điều chỉnh: z = {adjusted_z_axis:.4f}")
            if adjusted_z_axis > 0:
                print("Xác nhận: Trục z của OBB hướng lên phần dương, đáp ứng yêu cầu")
            else:
                print("Cảnh báo: Trục z của OBB vẫn không hướng lên phần dương, cần kiểm tra thêm")

            # Kiểm tra tâm OBB trong hệ trục OBB sau điều chỉnh
            obb_center_local_final = np.dot(np.linalg.inv(final_rotation_matrix), obb_center - obb_origin)
            print(f"Tâm OBB trong hệ trục tại obb_origin sau điều chỉnh: x_local = {obb_center_local_final[0]:.4f}")
            if obb_center_local_final[0] >= 0:
                print("Xác nhận: Tâm OBB nằm ở phần dương hoặc trên trục x trong hệ trục tại obb_origin")
            else:
                print("Cảnh báo: Tâm OBB vẫn nằm ở phần âm của trục x trong hệ trục tại obb_origin, cần kiểm tra thêm")

            print("Hệ trục tọa độ OBB được tạo và điều chỉnh tại vị trí white_point_3d hoặc obb.center")

            # Tính góc quay Euler từ ma trận xoay cuối cùng
            det = np.linalg.det(final_rotation_matrix)
            if det < 0:
                final_rotation_matrix[:, 2] *= -1
                print("Đã sửa ma trận xoay để chuyển từ hệ trái tay sang hệ phải tay")
            rotation = Rotation.from_matrix(final_rotation_matrix)
            euler_angles_1 = rotation.as_euler('xyz', degrees=True)
            white_point_mm = np.array(obb_origin) * 1000

            print(
                f"White point coordinates (mm): ({white_point_mm[0]:.2f}, {white_point_mm[1]:.2f}, {white_point_mm[2]:.2f})")
            print(
                f"Euler angles (degrees): Roll={euler_angles_1[0]:.2f}, Pitch={euler_angles_1[1]:.2f}, Yaw={euler_angles_1[2]:.2f}")
            self.append_log(
                f"Model 1: X={white_point_mm[0]:.2f} mm, Y={white_point_mm[1]:.2f} mm, Z={white_point_mm[2]:.2f} mm")
            self.append_log(
                f"Model 1: Roll={euler_angles_1[0]:.2f}°, Pitch={euler_angles_1[1]:.2f}°, Yaw={euler_angles_1[2]:.2f}°")

            def display_open3d():
                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name="Đám mây điểm 3D", width=1280, height=720)
                vis.add_geometry(downsampled_point_cloud)
                vis.add_geometry(obb)
                vis.add_geometry(aabb)
                vis.add_geometry(coordinate_frame)
                vis.add_geometry(obb_coordinate_frame)
                view_control = vis.get_view_control()
                view_control.set_front([0, 0, -1])
                view_control.set_lookat(np.mean(np.asarray(downsampled_point_cloud.points), axis=0))
                view_control.set_up([0, 1, 0])
                view_control.set_zoom(0.5)
                render_option = vis.get_render_option()
                render_option.point_size = 2.0
                render_option.line_width = 1.0
                vis.run()
                vis.destroy_window()

            thread = threading.Thread(target=display_open3d)
            thread.daemon = True
            thread.start()

            return True, white_point_mm, euler_angles_1

        except Exception as e:
            print(f"Lỗi khi hiển thị đám mây điểm 3D: {str(e)}")
            self.append_log(f"Lỗi khi hiển thị đám mây điểm 3D: {str(e)}")
            return False, None, None

    def model2_test(self, depth_frame, depth_intrinsics, model2_data):
        """
        Hiển thị đám mây điểm 3D của cụm lớn nhất từ hộp OBB 2D của Model 2 index 1,
        """
        try:
            # Debug: Log input data
            print(f"model2_data: {len(model2_data)} items")

            # Tìm hộp OBB 2D của Model 2 với index 1
            target_center = None
            target_rect = None
            for xywhr, rect, idx in model2_data:
                if idx == 1:
                    target_center = (xywhr[0], xywhr[1])
                    target_rect = rect
                    print(f"Found Model 2 box: center={target_center}, index={idx}")
                    break

            if target_center is None or target_rect is None:
                print("Lỗi: Không tìm thấy hộp Model 2 với index 1")
                return False, None, None

            # Lấy kích thước khung hình
            h, w = depth_frame.get_height(), depth_frame.get_width()

            # Tạo mask cho vùng hộp OBB 2D
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [target_rect], 255)

            # Lấy dữ liệu độ sâu
            depth_data = np.asanyarray(depth_frame.get_data())

            # Tạo đám mây điểm
            points = []
            for y in range(h):
                for x in range(w):
                    if mask[y, x] == 255:
                        depth_value = depth_data[y, x] / 1000.0  # Chuyển từ mm sang m
                        if 0 < depth_value < 5.0:  # Lọc độ sâu hợp lệ
                            point_3d = rs.rs2_deproject_pixel_to_point(
                                depth_intrinsics, [x, y], depth_value
                            )
                            points.append(point_3d)

            if not points:
                print("Lỗi: Không có điểm 3D hợp lệ trong vùng hộp")
                return False, None, None

            # Log số lượng điểm ban đầu
            print(f"Số lượng điểm 3D ban đầu: {len(points)}")

            # Tạo đối tượng PointCloud trong Open3D
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(np.array(points))

            # Phân cụm bằng DBSCAN
            eps = 0.005  # 5mm
            min_points = 100
            labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

            if len(labels) == 0 or np.all(labels == -1):
                print("Lỗi: Không tìm thấy cụm hợp lệ sau DBSCAN")
                return False, None, None

            # Tìm cụm lớn nhất
            cluster_sizes = {label: np.sum(labels == label) for label in np.unique(labels) if label >= 0}
            if not cluster_sizes:
                print("Lỗi: Không có cụm nào được hình thành")
                return False, None, None

            largest_cluster_label = max(cluster_sizes, key=cluster_sizes.get)
            largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
            largest_cluster_points = np.array(points)[largest_cluster_indices]

            # Log số lượng điểm trong cụm lớn nhất
            print(f"Số lượng điểm trong cụm lớn nhất: {len(largest_cluster_points)}")

            # Tạo PointCloud mới cho cụm lớn nhất
            largest_point_cloud = o3d.geometry.PointCloud()
            largest_point_cloud.points = o3d.utility.Vector3dVector(largest_cluster_points)

            # Giảm mẫu đám mây điểm
            voxel_size = 0.001  # 1mm
            downsampled_point_cloud = largest_point_cloud.voxel_down_sample(voxel_size)
            colors = np.zeros((len(np.asarray(downsampled_point_cloud.points)), 3))
            colors[:, 0] = 1.0  # Màu đỏ
            downsampled_point_cloud.colors = o3d.utility.Vector3dVector(colors)

            # Log số lượng điểm sau giảm mẫu
            print(f"Số lượng điểm sau giảm mẫu: {len(np.asarray(downsampled_point_cloud.points))}")

            # Tạo hộp OBB 3D
            obb = downsampled_point_cloud.get_oriented_bounding_box()
            obb.color = (0, 0, 1)  # Xanh lam

            # Tạo hộp AABB
            aabb = downsampled_point_cloud.get_axis_aligned_bounding_box()
            aabb.color = (0, 1, 0)  # Xanh lá

            # Tạo hệ trục tọa độ gốc
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

            # Tính tọa độ 3D của tâm hộp OBB 2D
            center_2d = target_center
            if 0 <= center_2d[0] < w and 0 <= center_2d[1] < h:
                depth_value = depth_frame.get_distance(int(center_2d[0]), int(center_2d[1]))
                if depth_value > 0:
                    center_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [center_2d[0], center_2d[1]],
                                                                depth_value)
                    print(f"Center 3D (m): {center_3d}")
                else:
                    print(f"Giá trị độ sâu không hợp lệ tại center: {center_2d}")
                    center_3d = obb.center
            else:
                print(f"Tọa độ tâm ngoài khung hình: {center_2d}")
                center_3d = obb.center

            # Tạo hệ trục tọa độ cho OBB tại center_3d
            obb_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=center_3d)
            obb_coordinate_frame.rotate(obb.R, center=center_3d)

            # Lấy vector trục z của hệ OBB trong không gian thế giới
            z_axis_obb = obb.R[:, 2]  # Cột thứ 3 của ma trận xoay là trục z

            # Kiểm tra hướng trục z của OBB so với hệ tọa độ thế giới
            if z_axis_obb[2] < 0:  # Trục z của OBB hướng lên (phần dương)
                # Tạo ma trận đối xứng qua mặt phẳng xy để đảo ngược trục z
                reflection_matrix = np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, -1]
                ])
                obb_coordinate_frame.rotate(reflection_matrix, center=center_3d)
                print("Đã đảo ngược trục z của OBB bằng phép đối xứng vì trục z hướng lên phần dương")
            else:
                print("Giữ nguyên hệ trục OBB vì trục z đã hướng xuống phần âm hoặc trung lập")

            # Kiểm tra lại hướng trục z sau khi điều chỉnh
            adjusted_z_axis = (obb_coordinate_frame.get_rotation_matrix_from_xyz((0, 0, 0)) @ np.array([0, 0, 1]))[2]
            print(f"Hướng trục z sau điều chỉnh: z = {adjusted_z_axis:.4f}")
            if adjusted_z_axis < 0:
                print("Xác nhận: Trục z của OBB hướng xuống phần âm, đáp ứng yêu cầu")
            else:
                print("Cảnh báo: Trục z của OBB vẫn không hướng xuống phần âm, cần kiểm tra thêm")

            print("Hệ trục tọa độ OBB được tạo và điều chỉnh theo yêu cầu")

            # Tính góc quay Euler từ ma trận xoay của OBB
            rotation = Rotation.from_matrix(obb.R)
            euler_angles_2 = rotation.as_euler('xyz', degrees=True)

            # Chuyển center_3d sang mm
            center_mm = np.array(center_3d) * 1000

            # In tọa độ và góc Euler vào log
            print(f"Center coordinates (mm): ({center_mm[0]:.2f}, {center_mm[1]:.2f}, {center_mm[2]:.2f})")
            print(
                f"Euler angles (degrees): Roll={euler_angles_2[0]:.2f}, Pitch={euler_angles_2[1]:.2f}, Yaw={euler_angles_2[2]:.2f}")
            self.append_log(f"Model 2: X={center_mm[0]:.2f}mm, Y={center_mm[1]:.2f}mm, Z={center_mm[2]:.2f}mm")
            self.append_log(
                f"Model 2: Roll={euler_angles_2[0]:.2f}°, Pitch={euler_angles_2[1]:.2f}°, Yaw={euler_angles_2[2]:.2f}°")

            def display_open3d():
                try:
                    print("Khởi tạo cửa sổ Open3D trong luồng riêng...")
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(window_name="Đám mây điểm 3D - Model 2 Index 1", width=1280, height=720)
                    vis.add_geometry(downsampled_point_cloud)
                    vis.add_geometry(obb)
                    vis.add_geometry(aabb)
                    vis.add_geometry(coordinate_frame)
                    vis.add_geometry(obb_coordinate_frame)
                    view_control = vis.get_view_control()
                    view_control.set_front([0, 0, -1])
                    view_control.set_lookat(np.mean(np.asarray(downsampled_point_cloud.points), axis=0))
                    view_control.set_up([0, 1, 0])
                    view_control.set_zoom(0.5)
                    render_option = vis.get_render_option()
                    render_option.point_size = 2.0
                    render_option.line_width = 1.0
                    print("Hiển thị cửa sổ Open3D...")
                    vis.run()
                    print("Đóng cửa sổ Open3D...")
                    vis.destroy_window()
                except Exception as e:
                    print(f"Lỗi khi hiển thị Open3D trong luồng riêng: {str(e)}")

            thread = threading.Thread(target=display_open3d)
            thread.daemon = True
            thread.start()
            print("Đã khởi động luồng hiển thị Open3D")

            return True, center_mm, euler_angles_2

        except Exception as e:
            print(f"Lỗi khi hiển thị đám mây điểm 3D cho Model 2: {str(e)}")
            return False, None, None

    def setup_camera_stream(self):
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            self.custom_colormap = self.create_purple_to_red_colormap()
            self.camera_timer.start()

            # Lấy thông tin camera
            device = profile.get_device()
            device_name = device.get_info(rs.camera_info.name)
            serial_number = device.get_info(rs.camera_info.serial_number)

            # Lấy thông tin USB
            usb_speed = device.get_info(rs.camera_info.usb_type_descriptor)
            if usb_speed == "2.0":
                usb_info = "USB 2.0"
            elif usb_speed in ["3.0", "3.1", "3.2"]:
                usb_info = f"{usb_speed}"
            else:
                usb_info = "Unknown"

            # Cập nhật giao diện
            self.ui.label_usb.setText(f"USB: {usb_info}")
            self.ui.label_status_camera.setText(f"Camera: {device_name}")
            print(f"Camera RealSense đã khởi động: {device_name}, Serial: {serial_number}, USB: {usb_info}")
            self.append_log(f"Camera RealSense đã khởi động: {device_name}, Serial: {serial_number}, USB: {usb_info}")
        except Exception as e:
            print(f"Lỗi khởi động camera: {str(e)}")
            self.append_log(f"Lỗi khởi động camera: {str(e)}")
            self.ui.label_usb.setText("USB: Không kết nối")
            self.ui.label_status_camera.setText("Camera: Lỗi kết nối")

    def update_camera_stream(self):
        if not self.pipeline:
            self.append_log("Lỗi: Pipeline chưa được khởi tạo")
            return
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                self.append_log("Lỗi: Không lấy được frame màu hoặc độ sâu")
                return

            color_image = np.asanyarray(color_frame.get_data())
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            if self.display_mode == "RGB":
                display_image = color_image_rgb
            elif self.display_mode == "Depth":
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_image = np.clip(depth_image, 0, 500)
                depth_scaled = cv2.convertScaleAbs(depth_image, alpha=255.0 / 500.0)
                display_image = cv2.LUT(cv2.merge([depth_scaled, depth_scaled, depth_scaled]), self.custom_colormap)
                display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            elif self.display_mode == "Detection":
                if self.model1 is not None and self.model2 is not None:
                    conf_value = self.ui.slider_conf.value() / 100.0
                    self.ui.label_conf.setText(f"Confidence: {conf_value:.2f}")
                    results1 = self.model1.predict(color_image, conf=conf_value)
                    results2 = self.model2.predict(color_image, conf=conf_value)
                    display_image = color_image_rgb.copy()

                    depth_profile = self.pipeline.get_active_profile().get_stream(rs.stream.depth)
                    depth_intr = depth_profile.as_video_stream_profile().get_intrinsics()
                    self.current_depth_frame = depth_frame
                    self.current_depth_intr = depth_intr

                    # Xử lý Model 1
                    display_image, model1_center_3d, _, model1_class0_data, class2_centers_to_draw = self.draw_obb(
                        display_image, depth_frame, depth_intr, results1, "Model 1", (0, 255, 0)
                    )
                    self.model1_class0_data = model1_class0_data
                    self.class2_centers_to_draw = class2_centers_to_draw
                    print(
                        f"Saved {len(self.model1_class0_data)} class 0 boxes, {len(self.class2_centers_to_draw)} class 2 boxes")

                    # Xử lý Model 2
                    display_image, _, model2_center_3d, model2_data, _ = self.draw_obb(
                        display_image, depth_frame, depth_intr, results2, "Model 2", (0, 0, 255)
                    )
                    self.model2_data = model2_data
                    print(f"Saved {len(self.model2_data)} Model 2 boxes")
                else:
                    display_image = color_image_rgb
                    self.append_log("Lỗi: Mô hình YOLOv11 không được tải")
            else:
                display_image = color_image_rgb

            # Hiển thị hình ảnh
            height, width, channel = display_image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(display_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.ui.stream.setPixmap(QtGui.QPixmap.fromImage(q_image).scaled(
                self.ui.stream.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        except Exception as e:
            self.append_log(f"Lỗi cập nhật luồng camera: {str(e)}")

    def change_display_mode(self, mode):
        self.display_mode = mode
        print(f"Chuyển chế độ hiển thị: {mode}")
        self.append_log(f"Chuyển chế độ hiển thị: {mode}")

    def camera_connect(self):
        print("Kết nối camera...")
        self.setup_camera_stream()

    def camera_disconnect(self):
        print("Ngắt kết nối camera...")
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
            self.camera_timer.stop()
        self.ui.stream.clear()
        self.ui.label_usb.setText("USB: Không kết nối")
        self.ui.label_status_camera.setText("Camera: Không kết nối")
        self.append_log("Ngắt kết nối camera")

    def robot_connect(self):
        try:
            ip = self.ui.lineEdit_ip.text()
            port = int(self.ui.lineEdit_port.text())
        except:
            print("Cổng không hợp lệ, vui lòng nhập số nguyên")
            self.ui.label_robot_status.setText("Status: Lỗi - Cổng không hợp lệ")
            self.append_log("Lỗi: Cổng không hợp lệ, vui lòng nhập số nguyên")
            return
        if self.robot_controller is None:
            self.robot_controller = CRIController()
        try:
            if self.robot_controller.connect(ip, port):
                print(f"Đã kết nối với robot tại {ip}:{port}")
                self.ui.label_robot_status.setText("Status: Kết nối thành công")
                self.append_log(f"Đã kết nối với robot tại {ip}:{port}")
                self.ui.slider_override.setValue(30)
                self.ui.pushButton_robot_connect.setEnabled(False)
                self.status_timer.start()
            else:
                print(f"Kết nối thất bại tại {ip}:{port}")
                self.ui.label_robot_status.setText("Status: Kết nối thất bại")
                self.append_log(f"Kết nối thất bại tại {ip}:{port}")
        except Exception as e:
            print(f"Lỗi kết nối: {str(e)}")
            self.ui.label_robot_status.setText("Status: Lỗi kết nối")
            self.append_log(f"Lỗi kết nối: {str(e)}")

    def robot_disconnect(self):
        if self.robot_controller is None or not self.robot_controller.connected:
            print("Không có kết nối robot")
            self.ui.label_robot_status.setText("Status: Không kết nối")
            self.append_log("Không có kết nối robot")
            self.ui.pushButton_robot_connect.setEnabled(True)
            self.status_timer.stop()
            self.ui.label_error.setText("Error: Không kết nối")
            self.ui.label_estop.setText("E-Stop: Không kết nối")
            self.ui.label_kinstate.setText("KinState: Không kết nối")
            self.ui.label_pos_cart.setText("PosCart: Không kết nối")
            self.ui.label_pos_joints.setText("PosJoints: Không kết nối")
            self.ui.label_gripper.setText("Gripper: Không")
            return
        try:
            self.robot_controller.close()
            print("Ngắt kết nối robot thành công")
            self.ui.label_robot_status.setText("Status: Không kết nối")
            self.append_log("Ngắt kết nối robot thành công")
            self.ui.pushButton_robot_connect.setEnabled(True)
            self.status_timer.stop()
            self.ui.label_error.setText("Error: Không kết nối")
            self.ui.label_estop.setText("E-Stop: Không kết nối")
            self.ui.label_kinstate.setText("KinState: Không kết nối")
            self.ui.label_pos_cart.setText("PosCart: Không kết nối")
            self.ui.label_pos_joints.setText("PosJoints: Không kết nối")
            self.ui.label_gripper.setText("Gripper: Không")
        except Exception as e:
            print(f"Lỗi ngắt kết nối: {str(e)}")
            self.ui.label_robot_status.setText("Status: Lỗi ngắt kết nối")
            self.append_log(f"Lỗi ngắt kết nối: {str(e)}")
        finally:
            self.robot_controller = None

    def robot_reset(self):
        try:
            if self.robot_controller is None or not self.robot_controller.connected:
                print("Error: Robot chưa kết nối")
                self.append_log("Error: Robot chưa kết nối")
                return
            if self.robot_controller.reset():
                print("Successfully reset robot")
                self.append_log("Successfully reset robot")
            else:
                print("Failed to reset robot")
                self.append_log("Failed to reset robot")
        except Exception as e:
            print(f"Error resetting robot: {str(e)}")
            self.append_log(f"Error resetting robot: {str(e)}")

    def robot_enable(self):
        try:
            if self.robot_controller is None or not self.robot_controller.connected:
                print("Error: Robot chưa kết nối")
                self.append_log("Error: Robot chưa kết nối")
                return
            if self.robot_controller.enable():
                print("Successfully enabled robot")
                self.append_log("Successfully enabled robot")
            else:
                print("Failed to enable robot")
                self.append_log("Failed to enable robot")
        except Exception as e:
            print(f"Error enabling robot: {str(e)}")
            self.append_log(f"Error enabling robot: {str(e)}")

    def robot_reference(self):
        try:
            if self.robot_controller is None or not self.robot_controller.connected:
                print("Error: Robot chưa kết nối")
                self.append_log("Error: Robot chưa kết nối")
                return
            selected_option = self.ui.combo_ref.currentText()
            if selected_option == "All joints":
                if self.robot_controller.reference_all_joints():
                    print("Successfully referenced all joints")
                    self.append_log("Successfully referenced all joints")
                else:
                    print("Failed to reference all joints")
                    self.append_log("Failed to reference all joints")
            else:
                joint_map = {
                    "Joint 1": "A1",
                    "Joint 2": "A2",
                    "Joint 3": "A3",
                    "Joint 4": "A4",
                    "Joint 5": "A5"
                }
                joint = joint_map.get(selected_option)
                if joint and self.robot_controller.reference_single_joint(joint):
                    print(f"Successfully referenced joint {selected_option}")
                    self.append_log(f"Successfully referenced joint {selected_option}")
                else:
                    print(f"Failed to reference joint {selected_option}")
                    self.append_log(f"Failed to reference joint {selected_option}")
        except Exception as e:
            print(f"Error referencing robot: {str(e)}")
            self.append_log(f"Error referencing robot: {str(e)}")

    def robot_override_changed(self):
        try:
            if self.robot_controller is None or not self.robot_controller.connected:
                return
            override_value = self.ui.slider_override.value()
            if self.robot_controller.set_override(override_value):
                print(f"Override set to {override_value}%")
                self.ui.label_override.setText(f"Override: {override_value}")
            else:
                print(f"Override set to {override_value}% failed")
                self.ui.label_override.setText("Override failed")
        except Exception as e:
            print(f"Error setting override: {str(e)}")

    def robot_get_joint_inputs(self):
        try:
            a1 = float(self.ui.lineEdit_a1.text())
            a2 = float(self.ui.lineEdit_a2.text())
            a3 = float(self.ui.lineEdit_a3.text())
            a4 = float(self.ui.lineEdit_a4.text())
            a5 = float(self.ui.lineEdit_a5.text())
            return a1, a2, a3, a4, a5
        except ValueError:
            print("Giá trị khớp không hợp lệ, vui lòng nhập số thực.")
            self.append_log("Lỗi: Giá trị khớp không hợp lệ, vui lòng nhập số thực.")
            return None

    def robot_move_joints(self):
        if self.robot_controller is None or not self.robot_controller.connected:
            print("Không thể di chuyển: Robot chưa được kết nối")
            self.append_log("Lỗi: Không thể di chuyển, robot chưa được kết nối")
            return
        inputs = self.robot_get_joint_inputs()
        if inputs is None:
            return
        a1, a2, a3, a4, a5 = inputs
        a6, e1, e2, e3 = 0.0, 0.0, 0.0, 0.0
        velocity = 100.0
        def move_task():
            try:
                if self.robot_controller.move_joints(
                    a1, a2, a3, a4, a5, a6, e1, e2, e3, velocity, wait_move_finished=True
                ):
                    print(
                        f"Đã hoàn thành di chuyển khớp đến A1={a1:.2f}, A2={a2:.2f}, "
                        f"A3={a3:.2f}, A4={a4:.2f}, A5={a5:.2f}"
                    )
                    self.append_log(
                        f"Đã hoàn thành di chuyển khớp đến A1={a1:.2f}, A2={a2:.2f}, "
                        f"A3={a3:.2f}, A4={a4:.2f}, A5={a5:.2f}"
                    )
                else:
                    print("Di chuyển khớp thất bại")
                    self.append_log("Di chuyển khớp thất bại")
            except CRICommandTimeOutError:
                print("Hết thời gian chờ di chuyển khớp")
                self.append_log("Lỗi: Hết thời gian chờ di chuyển khớp")
            except Exception as e:
                print(f"Lỗi khi di chuyển khớp: {str(e)}")
                self.append_log(f"Lỗi khi di chuyển khớp: {str(e)}")
        threading.Thread(target=move_task, daemon=True).start()

    def robot_move_joints_relative(self):
        if self.robot_controller is None or not self.robot_controller.connected:
            print("Không thể di chuyển: Robot chưa được kết nối")
            self.append_log("Lỗi: Không thể di chuyển, robot chưa được kết nối")
            return
        inputs = self.robot_get_joint_inputs()
        if inputs is None:
            return
        a1, a2, a3, a4, a5 = inputs
        a6, e1, e2, e3 = 0.0, 0.0, 0.0, 0.0
        def move_task():
            try:
                if self.robot_controller.move_joints_relative(
                    a1, a2, a3, a4, a5, a6, e1, e2, e3, velocity=100.0, wait_move_finished=True
                ):
                    print(
                        f"Đã hoàn thành di chuyển khớp tương đối với A1={a1:.2f}, A2={a2:.2f}, "
                        f"A3={a3:.2f}, A4={a4:.2f}, A5={a5:.2f}"
                    )
                    self.append_log(
                        f"Đã hoàn thành di chuyển khớp tương đối với: A1={a1:.2f}, A2={a2:.2f}, "
                        f"A3={a3:.2f}, A4={a4:.2f}, A5={a5:.2f}"
                    )
                else:
                    print("Di chuyển khớp tương đối thất bại")
                    self.append_log("Di chuyển khớp tương đối thất bại")
            except CRICommandTimeOutError:
                print("Hết thời gian chờ di chuyển khớp tương đối")
                self.append_log("Lỗi: Hết thời gian chờ di chuyển khớp tương đối")
            except Exception as e:
                print(f"Lỗi khi di chuyển khớp tương đối: {str(e)}")
                self.append_log(f"Lỗi khi di chuyển khớp tương đối: {str(e)}")
        threading.Thread(target=move_task, daemon=True).start()

    def robot_get_cartesian_inputs(self):
        try:
            x = float(self.ui.lineEdit_x.text())
            y = float(self.ui.lineEdit_y.text())
            z = float(self.ui.lineEdit_z.text())
            a = float(self.ui.lineEdit_a.text())
            b = float(self.ui.lineEdit_b.text())
            c = float(self.ui.lineEdit_c.text())
            return x, y, z, a, b, c
        except ValueError:
            print("Giá trị tọa độ không hợp lệ, vui lòng nhập số thực.")
            self.append_log("Lỗi: Giá trị tọa độ không hợp lệ, vui lòng nhập số thực.")
            return None

    def robot_move_cartesian(self):
        if self.robot_controller is None or not self.robot_controller.connected:
            print("Không thể di chuyển: Robot chưa được kết nối")
            self.append_log("Lỗi: Không thể di chuyển, robot chưa được kết nối")
            return
        inputs = self.robot_get_cartesian_inputs()
        if inputs is None:
            return
        x, y, z, a, b, c = inputs
        def move_task():
            try:
                if self.robot_controller.move_cartesian(
                    x, y, z, a, b, c, 0.0, 0.0, 0.0, velocity=100.0, wait_move_finished=True
                ):
                    print(
                        f"Đã hoàn thành di chuyển tuyệt đối đến X={x:.2f}, Y={y:.2f}, Z={z:.2f}, "
                        f"A={a:.2f}, B={b:.2f}, C={c:.2f}"
                    )
                    self.append_log(
                        f"Đã hoàn thành di chuyển tuyệt đối đến X={x:.2f}, Y={y:.2f}, Z={z:.2f}, "
                        f"A={a:.2f}, B={b:.2f}, C={c:.2f}"
                    )
                else:
                    print("Di chuyển tuyệt đối thất bại")
                    self.append_log("Di chuyển tuyệt đối thất bại")
            except CRICommandTimeOutError:
                print("Hết thời gian chờ di chuyển tuyệt đối")
                self.append_log("Lỗi: Hết thời gian chờ di chuyển tuyệt đối")
            except Exception as e:
                print(f"Lỗi khi di chuyển tuyệt đối: {str(e)}")
                self.append_log(f"Lỗi khi di chuyển tuyệt đối: {str(e)}")
        threading.Thread(target=move_task, daemon=True).start()

    def robot_move_relative(self):
        if self.robot_controller is None or not self.robot_controller.connected:
            print("Không thể di chuyển: Robot chưa được kết nối")
            self.append_log("Lỗi: Không thể di chuyển, robot chưa được kết nối")
            return
        inputs = self.robot_get_cartesian_inputs()
        if inputs is None:
            return
        x, y, z, a, b, c = inputs
        def move_task():
            try:
                if self.robot_controller.move_base_relative(
                    x, y, z, a, b, c, 0.0, 0.0, 0.0, velocity=100.0, wait_move_finished=True
                ):
                    print(
                        f"Đã hoàn thành di chuyển tương đối X={x:.2f}, Y={y:.2f}, Z={z:.2f}, "
                        f"A={a:.2f}, B={b:.2f}, C={c:.2f}"
                    )
                    self.append_log(
                        f"Đã hoàn thành di chuyển tương đối X={x:.2f}, Y={y:.2f}, Z={z:.2f}, "
                        f"A={a:.2f}, B={b:.2f}, C={c:.2f}"
                    )
                else:
                    print("Di chuyển tương đối thất bại")
                    self.append_log("Di chuyển tương đối thất bại")
            except CRICommandTimeOutError:
                print("Hết thời gian chờ di chuyển tương đối")
                self.append_log("Lỗi: Hết thời gian chờ di chuyển tương đối")
            except Exception as e:
                print(f"Lỗi khi di chuyển tương đối: {str(e)}")
                self.append_log(f"Lỗi khi di chuyển tương đối: {str(e)}")
        threading.Thread(target=move_task, daemon=True).start()

    def robot_stop_motion(self):
        try:
            if self.robot_controller is None or not self.robot_controller.connected:
                print("Không thể dừng: Robot chưa được kết nối")
                self.append_log("Lỗi: Không thể dừng, robot chưa được kết nối")
                return
            if self.robot_controller.stop_move():
                print("Đã dừng chuyển động của robot")
                self.append_log("Đã dừng chuyển động")
            else:
                print("Dừng chuyển động thất bại")
                self.append_log("Lỗi: Dừng chuyển động thất bại")
        except Exception as e:
            print(f"Lỗi khi dừng chuyển động: {str(e)}")
            self.append_log(f"Lỗi: {str(e)}")

    def robot_fold(self):
        if self.robot_controller is None or not self.robot_controller.connected:
            print("Không thể gập robot: Robot chưa được kết nối")
            self.append_log("Lỗi: Không thể gập robot, robot chưa được kết nối")
            return
        def move_task_fold():
            try:
                ok1 = self.robot_controller.move_joints(
                    0.0, -44.89, 54.89, 40.0, 0.0, 0.0, 0.0, 255, 0.0, velocity=50.0, wait_move_finished=True
                )
                if ok1:
                    print("Đã di chuyển đến vị trí gập robot 1")
                else:
                    print("Gập robot thất bại ở vị trí")
                    return
                ok2 = self.robot_controller.move_joints(
                    0.0, -44.89, 54.89, 65.0, 0.0, 0.0, 0.0, 255, 0.0, velocity=50.0, wait_move_finished=True
                )
                if ok2:
                    print("Đã di chuyển đến vị trí gập robot 2")
                    self.append_log("Đã di chuyển đến vị trí gập robot")
                else:
                    print("Gập robot thất bại ở vị trí 2")
                    self.append_log("Gập robot thất bại")
            except Exception as e:
                print(f"Lỗi khi gập robot: {str(e)}")
                self.append_log(f"Lỗi khi gập robot: {str(e)}")
        threading.Thread(target=move_task_fold, daemon=True).start()

    def robot_move_home(self):
        if self.robot_controller is None or not self.robot_controller.connected:
            print("Không thể di chuyển: Robot chưa được kết nối")
            self.append_log("Lỗi: Không thể di chuyển, robot chưa được kết nối")
            return
        def move_task_home():
            try:
                if self.robot_controller.move_joints(
                    0.0, 2.26, 1.88, 85.86, 0.0, 0.0, 0.0, 0.0, 0.0, velocity=100.0, wait_move_finished=True
                ):
                    print("Đã di chuyển robot về vị trí home")
                    self.append_log("Đã di chuyển robot về vị trí home")
                else:
                    print("Di chuyển về vị trí home thất bại")
                    self.append_log("Di chuyển về vị trí home thất bại")
            except Exception as e:
                print(f"Lỗi khi di chuyển về vị trí home: {str(e)}")
                self.append_log(f"Lỗi khi di chuyển về vị trí home: {str(e)}")
        threading.Thread(target=move_task_home, daemon=True).start()

    def robot_close_gripper(self):
        try:
            if self.robot_controller is None or not self.robot_controller.connected:
                print("Error: Robot chưa kết nối")
                self.append_log("Error: Không thể đóng gripper")
                return
            if self.robot_controller.set_dout(22, True):
                print("Successfully closed gripper")
                self.append_log("Successfully closed gripper")
            else:
                print("Failed to close gripper")
                self.append_log("Failed to close gripper")
        except Exception as e:
            print(f"Error closing gripper: {str(e)}")
            self.append_log(f"Error closing gripper: {str(e)}")

    def robot_open_gripper(self):
        try:
            if self.robot_controller is None or not self.robot_controller.connected:
                print("Error: Robot chưa kết nối")
                self.append_log("Error: Không thể mở gripper")
                return
            if self.robot_controller.set_dout(22, False):
                print("Successfully opened gripper")
                self.append_log("Successfully opened gripper")
            else:
                print("Failed to open gripper")
                self.append_log("Failed to open gripper")
        except Exception as e:
            print(f"Error opening gripper: {str(e)}")
            self.append_log(f"Error opening gripper: {str(e)}")

    def robot_send_custom_command(self):
        if self.robot_controller is None or not self.robot_controller.connected:
            print("Không thể gửi lệnh: Robot chưa được kết nối")
            self.append_log("Lỗi: Không thể gửi lệnh, robot chưa được kết nối")
            return
        input_text = self.ui.lineEdit_cmd.text().strip()
        cmd = "CMD " + self.ui.lineEdit_cmd.text().strip()
        if not cmd:
            self.append_log("Lỗi: Chưa nhập lệnh để gửi")
            return
        def send_task():
            try:
                self.robot_controller._send_command(cmd)
                print(f"Đã gửi lệnh: {input_text}")
                self.append_log(f"Đã gửi lệnh: {input_text}")
            except Exception as e:
                print(f"Lỗi khi gửi lệnh: {str(e)}")
                self.append_log(f"Lỗi khi gửi lệnh: {str(e)}")
        threading.Thread(target=send_task, daemon=True).start()

    def check_robot_state(self):
        if self.robot_controller is None or not self.robot_controller.connected:
            return False, "Robot chưa kết nối"
        if not self.robot_controller.robot_state.emergency_stop_ok:
            return False, "Robot đang ở trạng thái dừng khẩn cấp"
        # Kiểm tra lỗi trục, chấp nhận "NoError" hoặc 0 là trạng thái hợp lệ
        error_state = self.robot_controller.robot_state.combined_axes_error
        if str(error_state) != "NoError" and error_state != 0:
            return False, f"Robot có lỗi: {error_state}"
        return True, "Robot sẵn sàng"

    def update_button_states(self):
        running = self.program_state == "RUNNING"
        paused = self.program_state == "PAUSED"
        if not hasattr(self.ui, 'pushButton_program_pause'):
            self.append_log("Lỗi: Không tìm thấy pushButton_program_pause trong giao diện")
            return
        self.ui.pushButton_program_pause.setEnabled(running or paused)
        pause_label = "Continue" if paused else "Pause"
        self.ui.pushButton_program_pause.setText(pause_label)

    def run_program(self, loop=False):
        if self.program_state in ["RUNNING", "PAUSED"]:
            self.append_log("Chương trình đang chạy hoặc tạm dừng, không thể bắt đầu mới")
            return
        if not self.check_robot_state()[0]:
            self.append_log("Robot không sẵn sàng để chạy chương trình")
            return
        self.program_state = "RUNNING"
        self.is_looping = loop
        self.stop_event.clear()
        self.pause_event.set()
        self.current_step = None
        # Đảm bảo chỉ một luồng program_task
        if self.program_thread is not None and self.program_thread.is_alive():
            self.append_log("Luồng chương trình cũ vẫn chạy, đang dừng...")
            self.stop_program()
            time.sleep(0.1)  # Chờ luồng cũ dừng
        self.program_thread = threading.Thread(target=self.program_task, daemon=True)
        self.program_thread.start()
        self.update_button_states()
        self.append_log(f"Đã bắt đầu chương trình (loop={loop})")

    def run_program_single(self):
        """
        Chạy chương trình một lần duy nhất (giai đoạn 1 và giai đoạn 2).
        """
        self.run_program(loop=False)

    def run_program_loop(self):
        """
        Chạy chương trình lặp lại liên tục cho đến khi dừng.
        """
        self.run_program(loop=True)

    def program_task(self):
        try:
            while not self.stop_event.is_set():
                if self.program_state == "PAUSED":
                    time.sleep(0.1)  # Giảm tải CPU khi tạm dừng
                    continue
                if not self.pause_event.is_set():
                    time.sleep(0.1)  # Đảm bảo không chạy khi pause_event bị xóa
                    continue
                if self.is_looping or self.current_step is None:
                    self.current_step = "step1"
                    if self.program_state == "RUNNING" and self.pause_event.is_set():
                        self.run_step1_internal()
                    if self.stop_event.is_set():
                        break
                    self.current_step = "step2"
                    if self.program_state == "RUNNING" and self.pause_event.is_set():
                        self.run_step2_internal()
                    if self.stop_event.is_set():
                        break
                if not self.is_looping:
                    break
        except Exception as e:
            self.append_log(f"Lỗi trong chương trình: {str(e)}")
        finally:
            self.program_state = "STOPPED"
            self.current_step = None
            self.update_button_states()
            self.append_log("Chương trình đã dừng")

    def update_button_states(self):
        running = self.program_state == "RUNNING"
        paused = self.program_state == "PAUSED"
        if not hasattr(self.ui, 'pushButton_program_pause'):
            self.append_log("Lỗi: Không tìm thấy pushButton_program_pause trong giao diện")
            return
        self.ui.pushButton_program_pause.setEnabled(running or paused)
        pause_label = "Continue" if paused else "Pause"
        self.ui.pushButton_program_pause.setText(pause_label)

    def run_program_step1(self):
        """
        Chạy độc lập giai đoạn 1 (khớp).
        """
        if self.program_state in ["RUNNING", "PAUSED"]:
            self.append_log("Chương trình đang chạy hoặc tạm dừng, không thể chạy bước 1")
            return
        if not self.check_robot_state()[0]:
            self.append_log("Robot không sẵn sàng để chạy bước 1")
            return
        self.program_state = "RUNNING"
        self.stop_event.clear()
        self.pause_event.set()
        self.current_step = "step1"
        self.run_step1_internal()
        self.program_state = "STOPPED"
        self.current_step = None
        self.update_button_states()

    def run_program_step2(self):
        """
        Chạy độc lập giai đoạn 2 (Descartes).
        """
        if self.program_state in ["RUNNING", "PAUSED"]:
            self.append_log("Chương trình đang chạy hoặc tạm dừng, không thể chạy bước 2")
            return
        if not self.check_robot_state()[0]:
            self.append_log("Robot không sẵn sàng để chạy bước 2")
            return
        self.program_state = "RUNNING"
        self.stop_event.clear()
        self.pause_event.set()
        self.current_step = "step2"
        self.run_step2_internal()
        self.program_state = "STOPPED"
        self.current_step = None
        self.update_button_states()

    def run_step1_internal(self):
        is_ready, message = self.check_robot_state()
        if not is_ready:
            self.append_log(f"Lỗi giai đoạn 1: {message}")
            self.stop_event.set()
            return
        print(f"run_step1_internal: state={self.program_state}, pause_event={self.pause_event.is_set()}")

        def move_task_step1():
            with self.robot_lock:
                if self.stop_event.is_set() or not self.pause_event.is_set():
                    print("Bỏ qua di chuyển bước 1 do dừng hoặc tạm dừng")
                    return
                try:
                    # Kiểm tra và đặt chế độ hiển thị thành Detection
                    current_view = self.ui.combo_view.currentText()
                    if current_view != "Detection":
                        QtCore.QMetaObject.invokeMethod(
                            self.ui.combo_view,
                            "setCurrentText",
                            QtCore.Qt.QueuedConnection,
                            QtCore.Q_ARG(str, "Detection")
                        )
                        self.append_log("Đã đặt chế độ hiển thị thành Detection")
                        time.sleep(3.0)  # Đợi 3 giây để đảm bảo Detection kích hoạt
                    else:
                        self.append_log("Chế độ hiển thị đã là Detection, không cần thay đổi")

                    # Di chuyển robot về vị trí ban đầu bằng move_joints
                    self.append_log("Di chuyển robot về vị trí ban đầu")
                    if not self.robot_controller.move_joints(
                            0.00, 2.25, 1.88, 85.87, 0.00, 0.00, 0.00, 0.00, 0.00, velocity=100.0,
                            wait_move_finished=True
                    ):
                        self.append_log("Di chuyển về vị trí ban đầu thất bại")
                        self.stop_event.set()
                        return
                    self.append_log("Đã di chuyển robot về vị trí ban đầu")
                    self.robot_controller.set_dout(22, False)

                    # Vòng lặp để lấy tọa độ và di chuyển
                    while True:
                        # Kích hoạt model1_3d
                        if self.current_depth_frame and self.current_depth_intr:
                            result = self.model1_3d(
                                self.current_depth_frame,
                                self.current_depth_intr,
                                self.model1_class0_data,
                                self.class2_centers_to_draw
                            )
                            if isinstance(result, tuple) and len(result) == 3:
                                success, white_point_mm, euler_angles_1 = result
                                if success:
                                    self.append_log("Đã lấy tọa độ và góc quay từ model1_3d thành công")
                                    x1, y1, z1 = white_point_mm
                                    self.append_log(f"Di chuyển tương đối: X={-y1:.2f}, Y={-x1:.2f}")

                                    # Kiểm tra giá trị tuyệt đối của x1 và y1
                                    if abs(-y1) <= 5 and abs(-x1) <= 5:
                                        self.append_log("Tọa độ X và Y đều nhỏ hơn hoặc bằng 5, dừng di chuyển")
                                        break

                                    # Di chuyển đến tọa độ
                                    self.append_log(f"Di chuyển đến tọa độ X={-y1:.2f}, Y={-x1:.2f}, Z=-50.0")
                                    if not self.robot_controller.move_base_relative(
                                            -y1, -x1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, velocity=100.0,
                                            wait_move_finished=True
                                    ):
                                        self.append_log("Di chuyển đến tọa độ thất bại")
                                        self.stop_event.set()
                                        return
                                    self.append_log("Đã di chuyển đến tọa độ")
                                else:
                                    self.append_log("Lỗi: Không thể lấy tọa độ và góc quay từ model1_3d")
                                    self.stop_event.set()
                                    return
                            else:
                                self.append_log(f"Lỗi: model1_3d trả về định dạng không đúng: {result}")
                                self.stop_event.set()
                                return
                        else:
                            self.append_log("Lỗi: Dữ liệu độ sâu hoặc thông số nội tại không khả dụng")
                            self.stop_event.set()
                            return

                    # Lấy lại tọa độ 3D và góc quay Euler lần nữa
                    if self.current_depth_frame and self.current_depth_intr:
                        result_final = self.model1_3d(
                            self.current_depth_frame,
                            self.current_depth_intr,
                            self.model1_class0_data,
                            self.class2_centers_to_draw
                        )
                        if isinstance(result_final, tuple) and len(result_final) == 3:
                            success_final, white_point_mm_final, euler_angles_final = result_final
                            if success_final:
                                self.append_log("Đã lấy tọa độ và góc quay từ model1_3d lần cuối thành công")
                                x_final, y_final, z_final = white_point_mm_final
                                roll_final, pitch_final, yaw_final = euler_angles_final

                                # Chuyển góc Euler sang radian
                                roll_rad = np.radians(roll_final)
                                pitch_rad = np.radians(pitch_final)
                                yaw_rad = np.radians(yaw_final)

                                # Tạo ma trận quay ZYX
                                Rz = np.array([
                                    [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                                    [0, 0, 1]
                                ])
                                Ry = np.array([
                                    [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                                    [0, 1, 0],
                                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
                                ])
                                Rx = np.array([
                                    [1, 0, 0],
                                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                                    [0, np.sin(roll_rad), np.cos(roll_rad)]
                                ])
                                R = np.dot(Rz, np.dot(Ry, Rx))

                                # Tạo ma trận đồng nhất 4x4
                                T_camera = np.eye(4)
                                T_camera[:3, :3] = R
                                T_camera[:3, 3] = [x_final, y_final, z_final]

                                # Ma trận chuyển đổi đã cho
                                transformation_matrix = np.array([
                                    [0.04285152, -0.99896471, 0.01882049, 45.0045843],
                                    [0.99901115, 0.04320237, -0.00912998, 28.5242124],
                                    [0.00990466, 0.01840713, 0.99978040, -82.1672456],
                                    [0.0, 0.0, 0.0, 1.0]
                                ])
                                T_end_effector = np.dot(transformation_matrix, T_camera)
                                x, y, z = T_end_effector[:3, 3]
                                R = T_end_effector[:3, :3]
                                rotation = Rotation.from_matrix(R)
                                euler_angles_end = rotation.as_euler('xyz', degrees=True)
                                roll, pitch, yaw = euler_angles_end
                                self.append_log(f"Model 1: X={x:.2f} mm, Y={y:.2f} mm, Z={z:.2f} mm")
                                self.append_log(f"Model 1: Roll={roll:.2f}°, Pitch={pitch:.2f}°, Yaw={yaw:.2f}°")
                                if not self.robot_controller.move_base_relative(
                                        x, y, -z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, velocity=100.0,
                                        wait_move_finished=True
                                ):
                                    self.stop_event.set()
                                    return
                                if not self.robot_controller.move_joints_relative(
                                        0.00, 0.00, 0.00, 0.00, yaw, 0.00, 0.00, 0.00, 0.00, velocity=100.0,
                                        wait_move_finished=True
                                ):
                                    self.stop_event.set()
                                    return
                                if not self.robot_controller.move_base_relative(
                                        0.0, 0.0, -23.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, velocity=100.0,
                                        wait_move_finished=True
                                ):
                                    self.stop_event.set()
                                    return
                                self.robot_controller.set_dout(22, True)
                                time.sleep(0.2)
                                if not self.robot_controller.move_joints(
                                        -19.12, 35.90, -29.62, 83.72, 70.87, 0.0, 0.0, 0.0, 0.0, velocity=100.0,
                                        wait_move_finished=True
                                ):
                                    self.stop_event.set()
                                    return
                                if not self.robot_controller.move_base_relative(
                                        0, 0, -45, 0, 0.0, 0, 0.0, 0.0, 0.0, velocity=100.0,
                                        wait_move_finished=True
                                ):
                                    self.stop_event.set()
                                    return
                                self.robot_controller.set_dout(22, False)
                                time.sleep(0.2)
                                if not self.robot_controller.move_cartesian(
                                        366.0, -145.0, 200, 135.0, 0.0, 180.0, 0.0, 0.0, 0.0, velocity=100.0,
                                        wait_move_finished=True
                                ):
                                    self.stop_event.set()
                                    return
                                if not self.robot_controller.move_cartesian(
                                        366.0, -145.0, 182.5, 135.0, 0.0, 180.0, 0.0, 0.0, 0.0, velocity=100.0,
                                        wait_move_finished=True
                                ):
                                    self.stop_event.set()
                                    return
                                if not self.robot_controller.move_cartesian(
                                        370.0, -150.0, 190, 135.0, 0.0, 180.0, 0.0, 0.0, 0.0, velocity=100.0,
                                        wait_move_finished=True
                                ):
                                    self.stop_event.set()
                                    return
                                self.robot_controller.set_dout(22, True)
                                if not self.robot_controller.move_cartesian(
                                        370.0, -150.0, 182, 135.0, 0.0, 180.0, 0.0, 0.0, 0.0, velocity=100.0,
                                        wait_move_finished=True
                                ):
                                    self.stop_event.set()
                                    return
                                if not self.robot_controller.move_joints(
                                        0.00, 2.25, 1.88, 85.87, 0.00, 0.00, 0.00, 0.00, 0.00, velocity=100.0,
                                        wait_move_finished=True
                                ):
                                    self.stop_event.set()
                                    return
                                self.robot_controller.set_dout(22, False)

                            else:
                                self.append_log("Lỗi: Không thể lấy tọa độ và góc quay từ model1_3d lần cuối")
                                self.stop_event.set()
                                return
                        else:
                            self.append_log(f"Lỗi: model1_3d lần cuối trả về định dạng không đúng: {result_final}")
                            self.stop_event.set()
                            return
                    else:
                        self.append_log("Lỗi: Dữ liệu độ sâu hoặc thông số nội tại không khả dụng")
                        self.stop_event.set()
                        return
                except Exception as e:
                    self.append_log(f"Lỗi giai đoạn 1: {str(e)}")
                    self.stop_event.set()

            if not self.stop_event.is_set():
                self.append_log("Hoàn thành giai đoạn 1")

        threading.Thread(target=move_task_step1, daemon=True).start()

    def run_step2_internal(self):
        """
        Thực hiện giai đoạn 2: Di chuyển robot về vị trí khớp ban đầu, căn chỉnh tọa độ X, Y bằng
        move_base_relative, lấy tọa độ 3D cuối cùng, và chuyển đổi sang hệ end-effector.
        """
        is_ready, message = self.check_robot_state()
        if not is_ready:
            self.append_log(f"Lỗi giai đoạn 2: {message}")
            self.stop_event.set()
            return
        print(f"run_step2_internal: state={self.program_state}, pause_event={self.pause_event.is_set()}")

        def move_task_step2():
            with self.robot_lock:
                if self.stop_event.is_set() or not self.pause_event.is_set():
                    print("Bỏ qua di chuyển bước 2 do dừng hoặc tạm dừng")
                    return
                try:
                    # Bước 1: Đặt chế độ hiển thị thành Detection
                    current_view = self.ui.combo_view.currentText()
                    if current_view != "Detection":
                        QtCore.QMetaObject.invokeMethod(
                            self.ui.combo_view,
                            "setCurrentText",
                            QtCore.Qt.QueuedConnection,
                            QtCore.Q_ARG(str, "Detection")
                        )
                        self.append_log("Đã đặt chế độ hiển thị thành Detection")
                        time.sleep(3.0)  # Đợi 3 giây để đảm bảo Detection kích hoạt
                    else:
                        self.append_log("Chế độ hiển thị đã là Detection, không cần thay đổi")

                    # Bước 2: Di chuyển robot về vị trí khớp ban đầu
                    self.append_log("Di chuyển robot về vị trí khớp ban đầu")
                    if not self.robot_controller.move_joints(
                            0.00, 2.25, 1.88, 85.87, 0.00, 0.00, 0.00, 0.00, 0.00, velocity=100.0,
                            wait_move_finished=True
                    ):
                        self.append_log("Di chuyển về vị trí khớp ban đầu thất bại")
                        self.stop_event.set()
                        return
                    self.append_log("Đã di chuyển robot về vị trí khớp ban đầu")
                    self.robot_controller.set_dout(22, False)

                    # Bước 3: Lặp để căn chỉnh tọa độ X, Y
                    while not self.stop_event.is_set():
                        # Lấy tọa độ 3D từ model2_3d
                        if self.current_depth_frame and self.current_depth_intr:
                            success, center_mm, euler_angles_2 = self.model2_3d(
                                self.current_depth_frame,
                                self.current_depth_intr,
                                self.model2_data
                            )
                            if not success:
                                self.append_log("Lỗi: Không thể lấy tọa độ 3D từ model2_3d")
                                self.stop_event.set()
                                return
                            self.append_log("Đã lấy tọa độ 3D từ model2_3d thành công")
                        else:
                            self.append_log("Lỗi: Dữ liệu độ sâu hoặc thông số nội tại không khả dụng")
                            self.stop_event.set()
                            return

                        # Kiểm tra sai số X, Y
                        if abs(center_mm[0]) <= 5 and abs(center_mm[1]) <= 5:
                            self.append_log("Sai số X, Y nhỏ hơn hoặc bằng 5mm, thoát vòng lặp căn chỉnh")
                            break

                        # Di chuyển tương đối để căn chỉnh X, Y
                        self.append_log(f"Di chuyển tương đối: X={-center_mm[1]:.2f}mm, Y={-center_mm[0]:.2f}mm")
                        if not self.robot_controller.move_base_relative(
                                -center_mm[1], -center_mm[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                velocity=100.0, wait_move_finished=True
                        ):
                            self.append_log("Di chuyển tương đối thất bại")
                            self.stop_event.set()
                            return
                        time.sleep(1.0)  # Đợi để camera cập nhật khung mới

                    # Bước 4: Lấy tọa độ 3D lần nữa và chuyển đổi sang hệ end-effector
                    if not self.stop_event.is_set():
                        # Lấy lại tọa độ 3D và góc quay Euler lần nữa
                        if self.current_depth_frame and self.current_depth_intr:
                            result_final = self.model2_3d(
                                self.current_depth_frame,
                                self.current_depth_intr,
                                self.model2_data
                            )
                            if isinstance(result_final, tuple) and len(result_final) == 3:
                                success_final, white_point_mm_final, euler_angles_final = result_final
                                if success_final:
                                    self.append_log("Đã lấy tọa độ và góc quay từ model2_3d lần cuối thành công")
                                    x_final, y_final, z_final = white_point_mm_final
                                    roll_final, pitch_final, yaw_final = euler_angles_final

                                    # Chuyển góc Euler sang radian
                                    roll_rad = np.radians(roll_final)
                                    pitch_rad = np.radians(pitch_final)
                                    yaw_rad = np.radians(yaw_final)

                                    # Tạo ma trận quay ZYX
                                    Rz = np.array([
                                        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                                        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                                        [0, 0, 1]
                                    ])
                                    Ry = np.array([
                                        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                                        [0, 1, 0],
                                        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
                                    ])
                                    Rx = np.array([
                                        [1, 0, 0],
                                        [0, np.cos(roll_rad), -np.sin(roll_rad)],
                                        [0, np.sin(roll_rad), np.cos(roll_rad)]
                                    ])
                                    R = np.dot(Rz, np.dot(Ry, Rx))

                                    # Tạo ma trận đồng nhất 4x4
                                    T_camera = np.eye(4)
                                    T_camera[:3, :3] = R
                                    T_camera[:3, 3] = [x_final, y_final, z_final]

                                    # Ma trận chuyển đổi đã cho
                                    transformation_matrix = np.array([
                                        [0.04285152, -0.99896471, 0.01882049, 45.0045843],
                                        [0.99901115, 0.04320237, -0.00912998, 28.5242124],
                                        [0.00990466, 0.01840713, 0.99978040, -82.1672456],
                                        [0.0, 0.0, 0.0, 1.0]
                                    ])
                                    T_end_effector = np.dot(transformation_matrix, T_camera)
                                    x, y, z = T_end_effector[:3, 3]
                                    R = T_end_effector[:3, :3]
                                    rotation = Rotation.from_matrix(R)
                                    euler_angles_end = rotation.as_euler('xyz', degrees=True)
                                    roll, pitch, yaw = euler_angles_end
                                    self.append_log(f"Model 2: X={x:.2f} mm, Y={y:.2f} mm, Z={z:.2f} mm")
                                    self.append_log(f"Model 2: Roll={roll:.2f}°, Pitch={pitch:.2f}°, Yaw={yaw:.2f}°")

                                    # Điều chỉnh yaw nếu abs(yaw) > 90
                                    if abs(yaw) > 90:
                                        yaw = 90 - yaw
                                        self.append_log(f"Đã điều chỉnh yaw: {yaw:.2f}°")

                                    if not self.robot_controller.move_base_relative(
                                            x, y, -z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, velocity=100.0,
                                            wait_move_finished=True
                                    ):
                                        self.stop_event.set()
                                        return
                                    if not self.robot_controller.move_joints_relative(
                                            0.00, 0.00, 0.00, 0.00, yaw, 0.00, 0.00, 0.00, 0.00, velocity=100.0,
                                            wait_move_finished=True
                                    ):
                                        self.stop_event.set()
                                        return
                                    if not self.robot_controller.move_base_relative(
                                            0.0, 0.0, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, velocity=100.0,
                                            wait_move_finished=True
                                    ):
                                        self.stop_event.set()
                                        return
                                    self.robot_controller.set_dout(22, True)
                                    time.sleep(0.2)
                                    if not self.robot_controller.move_joints(
                                            -31.72, 20.91, 14.37, 54.72, -31.72, 0.0, 0.0, 0.0, 0.0, velocity=100.0,
                                            wait_move_finished=True
                                    ):
                                        self.stop_event.set()
                                        return
                                    if not self.robot_controller.move_cartesian(
                                            315, -153.70, 161.0, 180.0, 0.0, 180.0, 0.0, 0.0, 0.0, velocity=50.0,
                                            wait_move_finished=True
                                    ):
                                        self.stop_event.set()
                                        return
                                    self.robot_controller.set_dout(22, False)
                                    time.sleep(0.2)
                                    if not self.robot_controller.move_cartesian(
                                            315, -153.70, 180.0, 180.0, 0.0, 180.0, 0.0, 0.0, 0.0, velocity=100.0,
                                            wait_move_finished=True
                                    ):
                                        self.stop_event.set()
                                        return
                                    if not self.robot_controller.move_joints(
                                            0.00, 2.25, 1.88, 85.87, 0.00, 0.00, 0.00, 0.00, 0.00, velocity=100.0,
                                            wait_move_finished=True
                                    ):
                                        self.stop_event.set()
                                        return
                                else:
                                    self.append_log("Lỗi: Lấy tọa độ cuối cùng từ model2_3d thất bại")
                                    self.stop_event.set()
                                    return
                            else:
                                self.append_log("Lỗi: Kết quả từ model2_3d không hợp lệ")
                                self.stop_event.set()
                                return
                        else:
                            self.append_log("Lỗi: Dữ liệu độ sâu hoặc thông số nội tại không khả dụng")
                            self.stop_event.set()
                            return

                    if not self.stop_event.is_set():
                        self.append_log("Hoàn thành giai đoạn 2")
                except Exception as e:
                    self.append_log(f"Lỗi giai đoạn 2: {str(e)}")
                    self.stop_event.set()

        threading.Thread(target=move_task_step2, daemon=True).start()

    def pause_program(self):
        if self.program_state == "RUNNING":
            self.program_state = "PAUSED"
            self.pause_event.clear()

            # Kiểm tra trạng thái robot trước khi dừng
            is_ready, message = self.check_robot_state()
            if not is_ready:
                self.append_log(f"Không thể dừng robot: {message}")
                self.append_log("Chương trình đã tạm dừng (không dừng được robot)")
                self.update_button_states()
                return

            # Chạy stop_move trong luồng riêng để không chặn GUI
            def stop_move_task():
                try:
                    with self.robot_lock:
                        if self.robot_controller is not None and self.robot_controller.connected:
                            # Cố gắng tăng timeout nếu API hỗ trợ
                            try:
                                self.robot_controller.set_timeout(5.0)  # Tăng timeout lên 5 giây
                            except AttributeError:
                                pass  # Bỏ qua nếu không hỗ trợ set_timeout
                            self.robot_controller.stop_move()
                            self.append_log("Đã dừng chuyển động robot khi tạm dừng")
                except Exception as e:
                    self.append_log(f"Lỗi khi dừng robot: {str(e)}")

            # Khởi động luồng dừng
            stop_thread = threading.Thread(target=stop_move_task, daemon=True)
            stop_thread.start()
            # Chờ luồng hoàn tất tối đa 0.1 giây để không chặn GUI
            stop_thread.join(timeout=0.1)

            self.append_log("Chương trình đã tạm dừng")
        elif self.program_state == "PAUSED":
            self.program_state = "RUNNING"
            self.pause_event.set()
            self.append_log("Chương trình đã tiếp tục")
        self.update_button_states()

    def stop_program(self):
        if self.program_state in ["RUNNING", "PAUSED"]:
            self.stop_event.set()
            self.pause_event.set()  # Đánh thức luồng nếu đang tạm dừng
            with self.robot_lock:
                self.robot_controller.stop_move()
            self.program_state = "STOPPED"
            self.append_log("Chương trình đã dừng")
            self.update_button_states()

def main():
    app = QtWidgets.QApplication(sys.argv)
    try:
        gui = ME5110QGUI(app)
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()