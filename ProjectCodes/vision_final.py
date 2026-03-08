import time
import sys
import cv2
import numpy as np
from picamera2 import Picamera2
import easyocr
import torch
import argparse
import mediapipe as mp
from datalink_serial import datalink
import threading
import os
from datetime import datetime
import aruco_dectction_pi5  
# ==========================================
#              1. 全局配置参数
# ==========================================

CAM_SIZE = (640, 480)
SHOW_DISPLAY = True

# --- 相机标定参数 ---
with np.load('/home/pi/Aruco/calibration_data.npz') as data:
    original_camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

# 自动适配分辨率
cal_cx = original_camera_matrix[0, 2]
if cal_cx > CAM_SIZE[0] * 0.6:
    scale_factor = CAM_SIZE[0] / (cal_cx * 2)
    camera_matrix = original_camera_matrix * scale_factor
    camera_matrix[2, 2] = 1.0
else:
    camera_matrix = original_camera_matrix

# --- OCR / ArUco 配置 (同步 test2) ---
OCR_EVERY_N_FRAMES = 6
OCR_MAX_SIDE = 320     # 【优化点】限制 OCR 输入图像最大边长
CONF_THRESH = 0.70     # 【优化点】置信度阈值
ARUCO_MIN_COUNT = 4
PAD_PIX = 20
OCR_TRIALS = 3
OCR_MIN_HITS = 2

# --- 飞行控制参数 (同步 test2) ---
ARUCO_SIZE = 0.1
RIGHT_FLIGHT_VELOCITY = 0.3  # 搜索速度
LOCK_DURATION = 3.0          # 盲飞搜索持续时间

# --- 手势识别配置 ---
HOVER_CONFIRM_FRAMES = 10
TIP_IDS = [4, 8, 12, 16, 20]

torch.set_num_threads(1)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ==========================================
#              2. 辅助工具函数
# ==========================================

def count_fingers(hand_landmarks):
    """计算伸出的手指数量 (几何特征法)"""
    lm_list = hand_landmarks.landmark
    fingers = []

    # 拇指判定 (欧氏距离法)
    thumb_tip = lm_list[4]
    thumb_ip = lm_list[2]
    palm_center = lm_list[9]
    
    thumb_tip_dist = ((thumb_tip.x - palm_center.x) ** 2 + (thumb_tip.y - palm_center.y) ** 2) ** 0.5
    thumb_ip_dist = ((thumb_ip.x - palm_center.x) ** 2 + (thumb_ip.y - palm_center.y) ** 2) ** 0.5
    
    # 阈值 1.05
    fingers.append(1 if thumb_tip_dist > thumb_ip_dist * 1.05 else 0)

    # 其他四指判定 (Y轴高度法)
    for finger_idx in range(1, 5):
        tip_y = lm_list[TIP_IDS[finger_idx]].y
        pip_y = lm_list[TIP_IDS[finger_idx] - 2].y
        # 阈值 0.03
        fingers.append(1 if tip_y < pip_y - 0.03 else 0)

    return fingers.count(1)

def init_ocr():
    print("正在加载 EasyOCR 模型...")
    return easyocr.Reader(['en'], gpu=False)

def get_gate_roi(frame, corners):
    """提取门框中心区域 ROI"""
    pts = np.concatenate(corners, axis=0).reshape(-1, 2)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    x1 = max(int(x1) - PAD_PIX, 0)
    y1 = max(int(y1) - PAD_PIX, 0)
    x2 = min(int(x2) + PAD_PIX, frame.shape[1] - 1)
    y2 = min(int(y2) + PAD_PIX, frame.shape[0] - 1)
    return x1, y1, x2, y2

def run_ocr(reader, roi_bgr):
    """
    【核心优化】执行 OCR 识别
    包含：自适应缩放 + 灰度化 + 白名单过滤
    """
    h, w, _ = roi_bgr.shape
    scale = 1.0
    max_side = max(h, w)
    
    # 1. 自适应缩放 (源自 test2)
    if max_side > OCR_MAX_SIDE:
        scale = OCR_MAX_SIDE / float(max_side)
        roi_bgr = cv2.resize(roi_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    # 2. 推理
    results = reader.readtext(roi_bgr, allowlist='1234', detail=1, paragraph=False)
    
    best = None
    for bbox, text, conf in results:
        if not text.isdigit():
            continue
        if conf < CONF_THRESH:
            continue
        if best is None or conf > best[2]:
            box = (np.array(bbox) / scale).tolist() if scale != 1.0 else bbox
            best = (text, box, conf)
    return best

# ==========================================
#              3. 核心流程函数
# ==========================================

def phase_1_gesture_selection(picam2):
    """
    Phase 1: 交互与模式选择 (保持原 vision_final 的双重确认逻辑)
    """
    print("\n" + "="*50)
    print("Phase 1: 交互与模式选择")
    print("操作说明: 握拳(ArUco) / 五指(数字)")
    print("="*50 + "\n")

    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    STATE_WAITING_START = "WAITING_START"
    STATE_CONFIRM_FIST = "CONFIRM_FIST"
    STATE_WAITING_NUMBER = "WAITING_NUMBER"
    STATE_CONFIRM_NUMBER = "CONFIRM_NUMBER"

    current_state = STATE_WAITING_START
    
    fist_window = []
    FIST_WINDOW_SIZE = 20
    FIST_TRIGGER_COUNT = 15

    hover_window = []
    HOVER_WINDOW_SIZE = 20
    HOVER_TRIGGER_COUNT = 15

    number_window = []
    NUMBER_WINDOW_SIZE = 30
    NUMBER_TRIGGER_COUNT = 25

    candidate_number = None
    frame_count = 0
    PROCESS_EVERY_N_FRAMES = 3

    try:
        while True:
            frame = picam2.capture_array()
            frame = cv2.flip(frame, 1) # 镜像翻转方便交互
            
            frame_count += 1
            if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                cv2.putText(frame, f"State: {current_state}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if SHOW_DISPLAY:
                    cv2.imshow("Vision System", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): return None
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            fingers_count = -1
            is_fist = 0
            is_hover = 0
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers_count = count_fingers(hand_landmarks)
                
                if fingers_count == 0: is_fist = 1
                if fingers_count == 5: is_hover = 1
            
            # 更新滑动窗口
            fist_window.append(is_fist)
            if len(fist_window) > FIST_WINDOW_SIZE: fist_window.pop(0)
            
            hover_window.append(is_hover)
            if len(hover_window) > HOVER_WINDOW_SIZE: hover_window.pop(0)

            # --- 状态机逻辑 ---
            
            # 全局重置 (拳头)
            if current_state in [STATE_WAITING_NUMBER, STATE_CONFIRM_NUMBER]:
                if sum(fist_window) >= FIST_TRIGGER_COUNT:
                    print(">>> 重置: 返回初始状态 <<<")
                    current_state = STATE_WAITING_START
                    fist_window = []
                    hover_window = []
                    number_window = []
                    candidate_number = None
                    continue

            if current_state == STATE_WAITING_START:
                cv2.putText(frame, "WAITING: Fist->Aruco, Hover->Num", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if sum(fist_window) >= FIST_TRIGGER_COUNT:
                    current_state = STATE_CONFIRM_FIST
                    fist_window = []
                elif sum(hover_window) >= HOVER_TRIGGER_COUNT:
                    current_state = STATE_WAITING_NUMBER
                    hover_window = []
                    number_window = []

            elif current_state == STATE_CONFIRM_FIST:
                cv2.putText(frame, "CONFIRM: Fist->GO, Hover->Cancel", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                if sum(fist_window) >= FIST_TRIGGER_COUNT:
                    print(">>> 确认进入 ArUco 模式 <<<")
                    return "ARUCO_MODE"
                if sum(hover_window) >= HOVER_TRIGGER_COUNT:
                    current_state = STATE_WAITING_START
                    hover_window = []

            elif current_state == STATE_WAITING_NUMBER:
                cv2.putText(frame, "SHOW NUMBER (1-4)", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                if 1 <= fingers_count <= 4:
                    number_window.append(fingers_count)
                else:
                    number_window.append(0)
                
                if len(number_window) > NUMBER_WINDOW_SIZE: number_window.pop(0)
                
                valid_nums = [n for n in number_window if n != 0]
                if len(valid_nums) > NUMBER_TRIGGER_COUNT:
                    from collections import Counter
                    most_common, count = Counter(valid_nums).most_common(1)[0]
                    if count >= NUMBER_TRIGGER_COUNT:
                        candidate_number = most_common
                        current_state = STATE_CONFIRM_NUMBER
                        hover_window = []

            elif current_state == STATE_CONFIRM_NUMBER:
                cv2.putText(frame, f"CONFIRM {candidate_number}? Hover->Yes", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if sum(hover_window) >= HOVER_TRIGGER_COUNT:
                    print(f">>> 确认数字目标: {candidate_number} <<<")
                    return candidate_number

            # 显示状态
            cv2.putText(frame, f"State: {current_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if SHOW_DISPLAY:
                cv2.imshow("Vision System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        hands.close()
        cv2.destroyAllWindows()
    return None

def phase_2_tracking(picam2, target_number):
    """
    【移植自 test2】Phase 2: 数字追踪与闭环搜索
    逻辑：LOCK搜寻 -> HOVER检测 -> OCR识别 -> 匹配? -> (拍照/重搜)
    """
    print("\n" + "="*40)
    print(f"Phase 2: 启动数字追踪 (Target: {target_number})")
    print("逻辑: 搜索(Lock) -> 悬停(Hover) -> 识别(OCR) -> 决策")
    print("="*40 + "\n")

    # 初始化通信
    dl = datalink()
    threading.Thread(target=dl.drone, daemon=True).start()
    threading.Thread(target=dl.heartbeat, daemon=True).start()
    time.sleep(1)

    reader = init_ocr()
    
    # 状态机初始化
    mode = "SEARCH_FLYING"  # 初始状态：盲飞搜索
    ocr_trial_results = []
    lock_start_time = None
    frame_count = 0
    
    save_dir = "/home/pi/captured_images"
    os.makedirs(save_dir, exist_ok=True)

    try:
        while True:
            frame = picam2.capture_array()
            frame_count += 1
            disp = frame.copy()

            # 1. 检测 ArUco
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
            
            have_4_aruco = (ids is not None and len(ids) >= ARUCO_MIN_COUNT)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(disp, corners, ids)

            # ============ 状态机逻辑 (Test2 Core) ============

            if mode == "SEARCH_FLYING":
                # [状态: 盲飞搜索] 向右平飞寻找门框
                if lock_start_time is None:
                    print(f">>> 启动搜索: 向右飞行 {LOCK_DURATION}秒 <<<")
                    lock_start_time = time.time()
                
                elapsed = time.time() - lock_start_time
                if elapsed < LOCK_DURATION:
                    # 发送飞行指令 (Vx=0, Vy=右, Vz=0, Yaw=0)
                    dl.set_pose(0, RIGHT_FLIGHT_VELOCITY, 0, 0)
                    cv2.putText(disp, f"SEARCHING >> {LOCK_DURATION - elapsed:.1f}s", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    print(">>> 搜索结束，进入悬停检测 <<<")
                    mode = "HOVER"
                    lock_start_time = None

            elif mode == "HOVER":
                # [状态: 悬停检测]
                dl.set_pose(0, 0, 0, 0) # 悬停指令
                
                if have_4_aruco:
                    # 发现门框 -> 提取 ROI 并识别
                    x1, y1, x2, y2 = get_gate_roi(frame, corners)
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    
                    if frame_count % OCR_EVERY_N_FRAMES == 0:
                        roi = frame[y1:y2, x1:x2]
                        best = run_ocr(reader, roi)
                        
                        hit_target = False
                        if best:
                            text, box, conf = best
                            # 坐标还原用于绘制
                            box = np.array(box, dtype=int)
                            box[:, 0] += x1
                            box[:, 1] += y1
                            
                            hit_target = (text == str(target_number))
                            color = (0, 255, 0) if hit_target else (0, 0, 255)
                            
                            cv2.polylines(disp, [box], True, color, 2)
                            cv2.putText(disp, f"OCR: {text}", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            
                            if hit_target:
                                print(f">>> 命中目标! 识别到: {text} <<<")
                            else:
                                print(f">>> 错靶! 识别到: {text}, 目标: {target_number} <<<")
                        
                        # 累计结果
                        ocr_trial_results.append((bool(hit_target), best))
                        
                        # 决策逻辑
                        if len(ocr_trial_results) >= OCR_TRIALS:
                            hits = sum([r[0] for r in ocr_trial_results])
                            if hits >= OCR_MIN_HITS:
                                # 成功 -> 拍照
                                mode = "PHOTO"
                            else:
                                # 失败 -> 触发重搜闭环
                                print(">>> 目标不匹配，重新搜索... <<<")
                                mode = "SEARCH_FLYING"
                                lock_start_time = None
                            ocr_trial_results = []
                else:
                    # 未发现门框，保持悬停等待
                    cv2.putText(disp, "HOVERING: Waiting for Gate...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            elif mode == "PHOTO":
                # [状态: 任务完成]
                dl.set_pose(0, 0, 0, 0)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(save_dir, f"target_{target_number}_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                
                print(f">>> 任务完成! 照片已保存: {filename} <<<")
                mode = "MISSION_COMPLETE"

            elif mode == "MISSION_COMPLETE":
                dl.set_pose(0, 0, 0, 0)
                cv2.putText(disp, "MISSION COMPLETE", (10, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # UI 显示
            cv2.putText(disp, f"Mode: {mode}", (10, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if SHOW_DISPLAY:
                cv2.imshow("Vision System", disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    print("EMERGENCY STOP!")
                    dl.set_pose(0, 0, 0, 0)
                    time.sleep(1)

    except KeyboardInterrupt:
        print("User Stopped")
    finally:
        dl.set_pose(0, 0, 0, 0)
        print("Tracking Ended.")

# ==========================================
#              4. 主程序入口
# ==========================================

def main():
    print("初始化摄像头...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": CAM_SIZE})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    try:
        # Phase 1: 交互
        target = phase_1_gesture_selection(picam2)
        
        if target == "ARUCO_MODE":
            print(">>> 启动 ArUco 跟随模块 <<<")
            picam2.stop()
            cv2.destroyAllWindows()
            aruco_dectction_pi5.run_following_mode(None) # 注意: 需确认该模块是否需要传递camera对象
        elif target is not None:
            # Phase 2: 数字追踪 (Test2 逻辑)
            phase_2_tracking(picam2, target)
        else:
            print("未选择目标，程序结束。")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
