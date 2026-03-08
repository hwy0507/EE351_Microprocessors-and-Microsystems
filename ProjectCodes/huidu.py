import cv2
import numpy as np
import os
import sys
import time

# --- 尝试导入 Picamera2 (树莓派专用) ---
try:
    from picamera2 import Picamera2
    USING_PICAM2 = True
except ImportError:
    USING_PICAM2 = False
    print("未找到 Picamera2，将使用 OpenCV 默认驱动")

# ================= 配置区域 =================
# 模版文件夹路径 (根据你的描述)
TEMPLATE_DIR = os.path.join("digit_templates")

# 待识别的数字列表
TARGET_NUMS = ['1', '2', '3', '4']

# 统一缩放尺寸 (宽, 高) - 非常关键，必须和模版一致
# 建议在这个尺寸下进行匹配，既快又准
MATCH_SIZE = (50, 80)

# 匹配阈值 (0.0 - 1.0)，越高越严格
CONF_THRESH = 0.40

# 直接使用中心 ROI，不再依赖找纸张轮廓（背景对比度不够时更稳）
USE_DOC_DETECT = False
FALLBACK_ROI_SIZE = 240  # 中心 ROI 边长
# ===========================================

def order_points(pts):
    """排序四个角点：左上，右上，右下，左下"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """透视变换：把歪斜的纸张拉正"""
    rect = order_points(pts)
    # 为了方便处理，我们把拉正后的图固定为 200x200
    maxWidth = 200
    maxHeight = 200
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def load_templates():
    """加载 Aruco/digit_templates 下的 1,2,3,4 图片"""
    templates = {}
    print(f"正在加载模版，路径: {TEMPLATE_DIR}")
    
    if not os.path.exists(TEMPLATE_DIR):
        print(f"【错误】找不到文件夹: {TEMPLATE_DIR}")
        sys.exit(1)

    for num in TARGET_NUMS:
        # 尝试常见后缀
        found = False
        for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
            path = os.path.join(TEMPLATE_DIR, num + ext)
            if os.path.exists(path):
                # 读取并转为灰度
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                # 二值化处理（确保模版也是黑底白字风格，防止模版源图片风格不统一）
                # 这里假设模版是白底黑字的截图，我们需要反转成黑底白字来匹配
                _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
                
                # 强制缩放到标准大小
                img_resized = cv2.resize(img_bin, MATCH_SIZE)
                templates[num] = img_resized
                print(f"  - 已加载: {num} (来自 {path})")
                found = True
                break
        if not found:
            print(f"  [警告] 未找到数字 {num} 的模版文件！")

    if not templates:
        print("【错误】一个模版都没加载到，请检查文件名是否为 1.jpg, 2.png 等。")
        sys.exit(1)
        
    return templates

def process_digit_extract(warped_img):
    """
    从拉正的图片中，分割出数字
    """
    # 1. 转灰度
    if len(warped_img.shape) == 3:
        gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = warped_img

    # 2. 自适应阈值 + 反转，适应光照不均
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

    # 2.1 形态学开运算，去小噪点
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    
    # 3. 去除边框噪点 (透视变换可能会在边缘产生黑线，干扰识别)
    h, w = th.shape
    border = 5
    cv2.rectangle(th, (0,0), (w, h), (0,0,0), border*2)

    # 4. 找数字轮廓
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None

    # 找最大的轮廓（假设是数字）
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # 过滤太小的噪点
    if w * h < 300: 
        return None, None

    # 裁剪出数字 ROI
    digit_roi = th[y:y+h, x:x+w]
    
    # 关键：缩放到与模版完全一致的大小
    digit_ready = cv2.resize(digit_roi, MATCH_SIZE)
    
    return digit_ready, digit_roi

def match_target(input_img, templates):
    """模版匹配逻辑"""
    best_score = -1
    best_label = None
    
    for label, templ_img in templates.items():
        # 执行匹配
        res = cv2.matchTemplate(input_img, templ_img, cv2.TM_CCOEFF_NORMED)
        score = np.max(res)
        
        if score > best_score:
            best_score = score
            best_label = label
            
    return best_label, best_score

def main():
    # 1. 加载模版
    templates = load_templates()
    
    # 2. 启动相机
    if USING_PICAM2:
        picam2 = Picamera2()
        # 320x240 分辨率足够图像处理，且 FPS 高
        config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)})
        picam2.configure(config)
        picam2.start()
        time.sleep(1)
        print("相机启动完成...")
    else:
        cap = cv2.VideoCapture(0)
        
    print("开始运行。请将写有数字的白纸置于镜头下。")
    print(f"只识别: {TARGET_NUMS}")

    try:
        while True:
            # 获取图像
            if USING_PICAM2:
                frame = picam2.capture_array()
            else:
                ret, frame = cap.read()
                if not ret: break

            # --- 简化：直接用中心 ROI，不找纸张 ---
            digit_ready = None
            h, w, _ = frame.shape
            sz = FALLBACK_ROI_SIZE
            cx, cy = w // 2, h // 2
            x0 = max(cx - sz // 2, 0)
            y0 = max(cy - sz // 2, 0)
            x1 = min(cx + sz // 2, w)
            y1 = min(cy + sz // 2, h)
            roi = frame[y0:y1, x0:x1]
            digit_ready, _ = process_digit_extract(roi)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

            if digit_ready is not None:
                # 3. 模版匹配
                label, score = match_target(digit_ready, templates)
                text = f"Score:{score:.2f}"
                cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                if score > CONF_THRESH:
                    text2 = f"Num: {label} ({int(score*100)}%)"
                    cv2.putText(frame, text2, (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                # 在右下角显示“机器眼中的数字”，方便调试
                h2, w2 = digit_ready.shape
                frame[-h2:, -w2:] = cv2.cvtColor(digit_ready, cv2.COLOR_GRAY2BGR)

            # 显示画面
            cv2.imshow("Drone Digit Match", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        if USING_PICAM2:
            picam2.stop()
        else:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()