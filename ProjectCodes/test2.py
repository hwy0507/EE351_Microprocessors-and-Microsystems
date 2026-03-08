import time
import sys
import cv2
import numpy as np
from picamera2 import Picamera2
import easyocr
import torch
import mediapipe as mp
from datalink_serial import datalink
import threading
import os
from datetime import datetime

CAM_SIZE = (640, 480)
SHOW_DISPLAY = True

# Load camera calibration data
with np.load('/home/pi/Aruco/calibration_data.npz') as data:
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
f_x = camera_matrix[0, 0]
f_y = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

# OCR / ArUco configuration
OCR_EVERY_N_FRAMES = 6
OCR_MAX_SIDE = 320
CONF_THRESH = 0.70
ARUCO_MIN_COUNT = 4
PAD_PIX = 20
OCR_TRIALS = 3
OCR_MIN_HITS = 2

# Gesture recognition configuration
HOVER_CONFIRM_FRAMES = 10
TIP_IDS = [4, 8, 12, 16, 20]

# Flight control configuration - 降低速度
ARUCO_SIZE = 0.1
RIGHT_FLIGHT_VELOCITY = 0.3  # 从0.5降低到0.2 m/s
LOCK_DURATION = 3.0  # LOCK模式持续3秒

torch.set_num_threads(1)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def count_fingers(hand_landmarks):
    lm_list = hand_landmarks.landmark
    fingers = []

    thumb_tip = lm_list[4]
    thumb_ip = lm_list[2]
    palm_center = lm_list[9]
    
    thumb_tip_dist = ((thumb_tip.x - palm_center.x) ** 2 + (thumb_tip.y - palm_center.y) ** 2) ** 0.5
    thumb_ip_dist = ((thumb_ip.x - palm_center.x) ** 2 + (thumb_ip.y - palm_center.y) ** 2) ** 0.5
    
    fingers.append(1 if thumb_tip_dist > thumb_ip_dist * 1.05 else 0)

    for finger_idx in range(1, 5):
        tip_y = lm_list[TIP_IDS[finger_idx]].y
        pip_y = lm_list[TIP_IDS[finger_idx] - 2].y
        fingers.append(1 if tip_y < pip_y - 0.03 else 0)

    return fingers.count(1)

def init_ocr():
    print("Loading EasyOCR model...")
    return easyocr.Reader(['en'], gpu=False)

def get_gate_roi(frame, corners):
    """获取门框的ROI区域"""
    pts = np.concatenate(corners, axis=0).reshape(-1, 2)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    x1 = max(int(x1) - PAD_PIX, 0)
    y1 = max(int(y1) - PAD_PIX, 0)
    x2 = min(int(x2) + PAD_PIX, frame.shape[1] - 1)
    y2 = min(int(y2) + PAD_PIX, frame.shape[0] - 1)
    return x1, y1, x2, y2

def run_ocr(reader, roi_bgr):
    h, w, _ = roi_bgr.shape
    scale = 1.0
    max_side = max(h, w)
    if max_side > OCR_MAX_SIDE:
        scale = OCR_MAX_SIDE / float(max_side)
        roi_bgr = cv2.resize(roi_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
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

def phase_1_gesture_selection(picam2):
    print("\n" + "="*40)
    print("Phase 1: Gesture Target Selection")
    print("Please face the camera:")
    print("1. Open all five fingers (Hover) and hold still to activate")
    print("2. Show 1-4 fingers to select the target number")
    print("="*40 + "\n")

    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    state = "waiting_hover"
    hover_counter = 0
    pending_number = None
    number_counter = 0
    number_window = []
    confirm_hover_counter = 0
    selected_target = None
    confirm_hover_start_time = None
    candidate = None

    try:
        while True:
            frame = picam2.capture_array()
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            fingers_count = 0

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers_count = count_fingers(hand_landmarks)

                if state == "waiting_hover":
                    if fingers_count == 5:
                        hover_counter += 1
                        if hover_counter >= HOVER_CONFIRM_FRAMES:
                            print(">>> Hover gesture detected, please show the number (1-4) <<<")
                            cv2.putText(frame, "Now show the number to track", (40, 240),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                            cv2.imshow("Vision System", frame)
                            cv2.waitKey(2000)
                            state = "waiting_number"
                    else:
                        hover_counter = 0

                elif state == "waiting_number":
                    if 1 <= fingers_count <= 4:
                        number_window.append(fingers_count)
                        if len(number_window) > 100:
                            number_window.pop(0)
                        from collections import Counter
                        counter = Counter(number_window)
                        most_common = counter.most_common(1)
                        if most_common:
                            most_num, most_count = most_common[0]
                            if most_count >= 80:
                                candidate = most_num
                                print(f"\n>>> Number {candidate} detected (>=80/100 frames), please show 5 fingers again to confirm <<<")
                                state = "confirm_hover"
                                confirm_hover_counter = 0
                                confirm_hover_start_time = time.time()
                                cv2.putText(frame, f"Confirm {candidate}? Show 5 fingers", (40, 240),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                                cv2.imshow("Vision System", frame)
                                cv2.waitKey(2000)
                                number_window = []
                    else:
                        number_window = []

                elif state == "confirm_hover":
                    if confirm_hover_start_time is not None and (time.time() - confirm_hover_start_time) > 5.0:
                        print("No Hover detected for 5 seconds, returning to number confirmation.")
                        state = "waiting_number"
                        number_counter = 0
                        pending_number = None
                        cv2.putText(frame, "No Hover detected. Please show the number again.", (40, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.imshow("Vision System", frame)
                        cv2.waitKey(2000)
                        continue
                    if fingers_count == 5:
                        confirm_hover_counter += 1
                        if confirm_hover_counter >= HOVER_CONFIRM_FRAMES:
                            selected_target = candidate
                            print(f"\n>>> Final confirmation: Number {selected_target} <<<")
                            cv2.putText(frame, f"FINAL TARGET: {selected_target}", (50, 240),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                            cv2.imshow("Vision System", frame)
                            cv2.waitKey(2000)
                            break
                    else:
                        confirm_hover_counter = 0

            status_text = f"State: {state} | Fingers: {fingers_count}"
            color = (0, 255, 0) if state == "waiting_hover" else (0, 255, 255)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if SHOW_DISPLAY:
                cv2.imshow("Vision System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User cancelled")
                    return None

    finally:
        hands.close()

    return selected_target


def phase_2_tracking(picam2, target_number):
    print("\n" + "="*40)
    print(f"Phase 2: Tracking target {target_number}")
    print("="*40 + "\n")

    dl = datalink()
    data_thread = threading.Thread(target=dl.drone, daemon=True)
    heartbeat_thread = threading.Thread(target=dl.heartbeat, daemon=True)
    data_thread.start()
    heartbeat_thread.start()
    time.sleep(1)

    reader = init_ocr()

    # 修改状态机：直接从HOVER开始
    mode = "HOVER"
    gate_lost_frames = 0
    ocr_trial_results = []
    lock_start_time = None
    unlock_start_time = None
    
    frame_count = 0

    save_dir = "/home/pi/captured_images"
    os.makedirs(save_dir, exist_ok=True)

    try:
        while True:
            frame = picam2.capture_array()
            frame_count += 1
            disp = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect ArUco markers
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

            raw_count = 0 if ids is None else len(ids)
            have_4_aruco = (raw_count >= ARUCO_MIN_COUNT)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(disp, corners, ids)

            frame_center_x = frame.shape[1] / 2.0
            frame_center_y = frame.shape[0] / 2.0
            cv2.drawMarker(disp, (int(frame_center_x), int(frame_center_y)), (0, 0, 255), 
                          cv2.MARKER_CROSS, 20, 2)

# ============ 新的状态机逻辑 ============
            


            if mode == "SEARCH_FLYING":
                # 进入LOCK模式飞行
                if lock_start_time is None:
                    print(f"\n>>> Entering LOCK mode, flying right for {LOCK_DURATION}s <<<")
                    lock_start_time = time.time()
                elapsed_lock = time.time() - lock_start_time
                if elapsed_lock < LOCK_DURATION:
                    dl.set_pose(0, RIGHT_FLIGHT_VELOCITY, 0, 0)
                    cv2.putText(disp, f"LOCK MODE - FLYING RIGHT ({LOCK_DURATION - elapsed_lock:.1f}s)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    print(f"\n>>> LOCK complete, entering HOVER mode <<<")
                    mode = "HOVER"
                    lock_start_time = None
                cv2.putText(disp, f"Velocity: {RIGHT_FLIGHT_VELOCITY} m/s", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


            # 取消UNLOCK_FLYING，lock flying后直接进入HOVER


            elif mode == "HOVER":
                # 悬停，检测Aruco并OCR。未检测到4个Aruco时持续等待，不切换状态。
                dl.set_pose(0, 0, 0, 0)
                if have_4_aruco:
                    gate_lost_frames = 0
                    x1, y1, x2, y2 = get_gate_roi(frame, corners)
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    # 直接进入OCR模式
                    if frame_count % OCR_EVERY_N_FRAMES == 0:
                        roi = frame[y1:y2, x1:x2]
                        best = run_ocr(reader, roi)
                        hit_target = False
                        if best:
                            text, box, conf = best
                            box = np.array(box, dtype=int)
                            box[:, 0] += x1
                            box[:, 1] += y1
                            hit_target = (text == str(target_number))
                            color = (0, 255, 0) if hit_target else (0, 255, 255)
                            cv2.polylines(disp, [box], True, color, 2)
                            cv2.putText(disp, f"{text} ({conf:.2f})", (x1, max(0, y1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            if hit_target:
                                print(f"\n>>> OCR success: {text} - MATCH! <<<")
                            else:
                                print(f"\n>>> OCR detected: {text}, not target {target_number} <<<")
                        ocr_trial_results.append((bool(hit_target), best))
                        # 完成OCR_TRIALS次尝试后判断
                        if len(ocr_trial_results) >= OCR_TRIALS:
                            hits = sum([r[0] for r in ocr_trial_results[:OCR_TRIALS]])
                            if hits >= OCR_MIN_HITS:
                                print(f"\n>>> Target number {target_number} confirmed! Taking photo... <<<")
                                mode = "PHOTO"
                            else:
                                print(f"\n>>> Number mismatch! Start searching right <<<")
                                mode = "SEARCH_FLYING"
                                lock_start_time = None
                            ocr_trial_results = []
                else:
                    # 未检测到4个Aruco时持续悬停等待
                    gate_lost_frames += 1
                    # 不切换状态，持续等待
                # 不在屏幕上打印hovering相关信息

            elif mode == "OCR":
                # 4. OCR识别
                dl.set_pose(0, 0, 0, 0)
                
                if have_4_aruco:
                    gate_lost_frames = 0
                    x1, y1, x2, y2 = get_gate_roi(frame, corners)
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    
                    # 每N帧进行一次OCR识别
                    if frame_count % OCR_EVERY_N_FRAMES == 0:
                        roi = frame[y1:y2, x1:x2]
                        best = run_ocr(reader, roi)
                        hit_target = False
                        
                        if best:
                            text, box, conf = best
                            box = np.array(box, dtype=int)
                            box[:, 0] += x1
                            box[:, 1] += y1
                            
                            hit_target = (text == str(target_number))
                                
                            color = (0, 255, 0) if hit_target else (0, 255, 255)
                            cv2.polylines(disp, [box], True, color, 2)
                            cv2.putText(disp, f"{text} ({conf:.2f})", (x1, max(0, y1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            if hit_target:
                                print(f"\n>>> OCR success: {text} - MATCH! <<<")
                            else:
                                print(f"\n>>> OCR detected: {text}, not target {target_number} <<<")

                        ocr_trial_results.append((bool(hit_target), best))
                        
                        # 完成OCR_TRIALS次尝试后判断
                        if len(ocr_trial_results) >= OCR_TRIALS:
                            hits = sum([r[0] for r in ocr_trial_results[:OCR_TRIALS]])
                            
                            if hits >= OCR_MIN_HITS:
                                # 是目标数字 - 拍照
                                print(f"\n>>> Target number {target_number} confirmed! Taking photo... <<<")
                                mode = "PHOTO"
                            else:
                                # 不是目标数字 - 重新开始搜索流程
                                print(f"\n>>> Number mismatch! Restarting search flow <<<")
                                mode = "SEARCH_FLYING"
                                lock_start_time = None
                            
                            ocr_trial_results = []
                else:
                    gate_lost_frames += 1
                    if gate_lost_frames >= 30:
                        print(f"\n>>> Lost gate during OCR, restarting search <<<")
                        mode = "SEARCH_FLYING"
                        lock_start_time = None
                
                cv2.putText(disp, f"OCR MODE (Trials: {len(ocr_trial_results)}/{OCR_TRIALS})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            elif mode == "PHOTO":
                dl.set_pose(0, 0, 0, 0)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(save_dir, f"target_{target_number}_{timestamp}.jpg")
                
                cv2.imwrite(filename, frame)
                
                print("\n" + "="*60)
                print(f">>> PHOTO CAPTURED AND SAVED! <<<")
                print(f">>> Filename: {filename} <<<")
                print(f">>> MISSION COMPLETE! <<<")
                print(f">>> Drone hovering in place... <<<")
                print("="*60 + "\n")
                
                cv2.putText(disp, f"MISSION COMPLETE! Target: {target_number}", (10, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(disp, f"Photo saved: {filename}", (10, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if SHOW_DISPLAY:
                    cv2.imshow("Vision System", disp)
                    cv2.waitKey(3000)
                
                mode = "MISSION_COMPLETE"

            elif mode == "MISSION_COMPLETE":
                dl.set_pose(0, 0, 0, 0)
                
                cv2.putText(disp, f"MISSION COMPLETE! Hovering...", (10, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(disp, f"Target: {target_number} | Photo Saved", (10, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(disp, "Press 'q' to exit", (10, 320),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # 显示当前状态信息
            cv2.putText(disp, f"MODE: {mode} | ArUco: {raw_count}", (10, disp.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if SHOW_DISPLAY:
                cv2.imshow("Vision System", disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    print("EMERGENCY STOP! User pressed 's'")
                    dl.set_pose(0, 0, 0, 0)
                    lock_start_time = None
                    unlock_start_time = None
                    time.sleep(1)

    except KeyboardInterrupt:
        print("\n>>> Keyboard Interrupt detected <<<")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Shutting down safely...")
        dl.set_pose(0, 0, 0, 0)
        time.sleep(0.5)
        print(">>> Drone hovering, connection maintained <<<")


def main():
    print("="*50)
    print("Modified ArUco Gate Detection System")
    print("="*50)
    print("Mission Flow:")
    print("  1. LOCK mode → Fly right 3s")
    print("  2. UNLOCK mode → Continue flying until 4 ArUco detected")
    print("  3. Detect GATE → Immediate hover")
    print("  4. OCR recognition → Check if target number")
    print("  5. If YES → Take photo → Mission complete")
    print("  6. If NO → Repeat from step 1")
    print("\nControl Parameters:")
    print(f"  - Right flight velocity: {RIGHT_FLIGHT_VELOCITY} m/s (REDUCED)")
    print(f"  - Lock duration: {LOCK_DURATION}s")
    print(f"  - OCR trials: {OCR_TRIALS}")
    print(f"  - OCR min hits: {OCR_MIN_HITS}")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' for emergency stop")
    print("="*50 + "\n")
    
    print("Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": CAM_SIZE})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    try:
        target = phase_1_gesture_selection(picam2)
        
        if target is not None:
            phase_2_tracking(picam2, target)
        else:
            print("No target selected, program ended.")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing camera and windows...")
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()