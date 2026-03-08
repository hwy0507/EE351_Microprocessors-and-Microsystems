import cv2
import cv2.aruco as aruco
import numpy as np
import threading
from datalink_serial import datalink
import time
from picamera2 import Picamera2

# 加载原始标定数据
with np.load('/home/pi/Aruco/calibration_data.npz') as data:
    original_camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

# ArUco字典和检测参数
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

def run_following_mode(picam2):
    """
    ArUco 跟随模式主函数
    参数:
        picam2: 已经初始化并启动的 Picamera2 对象
    """
    print("\n" + "="*40)
    print("Starting ArUco Following Mode (Module)")
    print("="*40 + "\n")

    # 1. 获取当前相机分辨率
    # 注意：picam2.camera_configuration 是一个字典结构
    # 通常在 main 配置中可以找到 size
    try:
        stream_config = picam2.camera_configuration['main']
        width = stream_config['size'][0]
        height = stream_config['size'][1]
        current_size = (width, height)
        print(f"Detected Camera Resolution: {width}x{height}")
    except Exception as e:
        print(f"Warning: Could not detect resolution from config ({e}), defaulting to 640x480")
        current_size = (640, 480)
        width, height = 640, 480

    # 2. 自动适配相机内参
    cal_cx = original_camera_matrix[0, 2]
    
    # 如果标定中心点明显偏离当前分辨率中心（判定阈值设为 0.6 倍宽度）
    # 例如标定是 1280 (cx~640)，当前是 640，则 640 > 640*0.6，需要缩放
    # 如果标定是 640 (cx~320)，当前是 640，则 320 < 384，不需要缩放
    if cal_cx > width * 0.6:
        est_orig_width = cal_cx * 2
        scale_factor = width / est_orig_width
        print(f"Calibration mismatch: cal_cx={cal_cx}, current_w={width}. Scaling by {scale_factor:.3f}")
        
        camera_matrix = original_camera_matrix * scale_factor
        camera_matrix[2, 2] = 1.0 # 保持齐次坐标系数为1
    else:
        print("Calibration matches current resolution.")
        camera_matrix = original_camera_matrix.copy()

    f_x = camera_matrix[0, 0]
    f_y = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    print(f"Using Camera Matrix:\n{camera_matrix}")

    # 3. 初始化控制参数
    aruco_size = 0.1  # ArUco码的实际边长，单位：米
    kp_x = 0.6 
    kp_y = 0.6
    kp_alt = 0.6
    kp_yaw = 0.3

    # 4. 初始化 Datalink
    dl = datalink()
    data_thread = threading.Thread(target=dl.drone)
    heartbeat_thread = threading.Thread(target=dl.heartbeat)
    data_thread.daemon = True
    heartbeat_thread.daemon = True
    data_thread.start()
    heartbeat_thread.start()
    
    print("Datalink initialized. Waiting for stream...")
    time.sleep(1)

    try:
        while True:
            frame = picam2.capture_array()
            # 保持与原代码一致的处理逻辑（不翻转，或者根据需要翻转）
            # 注意：vision_final 中通常有 cv2.flip(frame, 1)，这里如果需要保持一致性请确认
            # 原 aruco_dectction_pi5.py 没有翻转，这里保持原样
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            if ids is not None:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, aruco_size, camera_matrix, dist_coeffs)
                
                for i in range(len(ids)):
                    tvec = tvecs[i][0]

                    aruco.drawDetectedMarkers(frame, corners)
                    
                    dz_m = tvec[2]  # 前后
                    dx_m = tvec[0]  # 左右
                    dy_m = -tvec[1] # 高度

                    # 计算中心点
                    corner_points = corners[i][0]
                    x1, y1 = corner_points[0]
                    x2, y2 = corner_points[2]
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    dx_pixel = center_x - cx
                    d_yaw = np.arctan(dx_pixel / f_x)

                    # 目标距离设定 (单位: 米)
                    # 之前由于分辨率内参未缩放，导致测距偏大(约2倍)，设定1.5m实际停在0.75m
                    # 现在内参已修复，测距准确。为了保持较近的跟随距离，将目标设为 1m
                    TARGET_DIST = 1 
                    dx_1 = dz_m - TARGET_DIST
                    
                    dy_1 = dx_m
                    d_alt_1 = dy_m

                    dl.set_pose(kp_x * dx_1, kp_y * dy_1, kp_alt * d_alt_1, kp_yaw * d_yaw)
                    
                    label = f'dx:{dx_1:.2f}m dy:{dy_1:.2f}m dz:{d_alt_1:.2f}m yaw:{d_yaw:.2f}'
                    cv2.putText(frame, label, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    print(f"Target: {label}")

            else:
                # print("No ArUco marker detected.")
                pass

            cv2.imshow('ArUco Following Mode', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting Following Mode...")
                break

    except KeyboardInterrupt:
        print("Stopping Following Mode")
    finally:
        dl.set_pose(0, 0, 0, 0)
        dl.socket_tcp.close()
        cv2.destroyWindow('ArUco Following Mode')
        # 注意：不要在这里 picam2.stop()，因为外部还要用

# 兼容旧的独立运行方式
def qr_code_detection():
    print("Running in standalone mode...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (1280, 720)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    
    try:
        run_following_mode(picam2)
    finally:
        picam2.stop()

if __name__ == "__main__":
    qr_code_detection() 
