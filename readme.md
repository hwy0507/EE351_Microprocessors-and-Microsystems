# 基于视觉导航的智能消防救援无人机系统
本项目是一个集成了 **手势交互**、**ArUco 视觉定位**、**OCR 数字识别** 与 **自主飞行控制** 的综合系统。基于 Raspberry Pi 5 和 Picamera2 开发，旨在模拟火场环境下的无人化侦察任务。

---

## 🌟 核心功能 (Key Features)

1.  **高鲁棒性交互 (Robust Interaction)**
    *   基于 MediaPipe 的手势识别。
    *   引入**时序滑动窗口滤波**与**双重确认状态机**，防止误触。
    *   支持“握拳全局重置”功能。

2.  **双模式作业 (Dual-Mode Operation)**
    *   **模式 A (ArUco Mode)**：纯视觉跟随，锁定 ArUco 码进行姿态同步。
    *   **模式 B (Digit Mode)**：智能数字侦察，寻找特定数字并拍照。

3.  **自主闭环决策 (Autonomous Closed-Loop)**
    *   **空间注意力机制**：通过 ArUco 门框定位 ROI 区域，屏蔽背景干扰。
    *   **自愈式搜索**：若 OCR 识别结果与目标不符，无人机会自动切换至盲飞模式寻找下一个目标，直至任务完成。

---

## 🎬 功能演示 (Demo Videos)

为便于快速查看核心能力，这里提供两个功能演示视频：

### 1) OCR 数字识别 (Digit Recognition)

<video src="assets/videos/recognition.mp4" controls width="720">
  Your browser does not support the video tag.
</video>

演示链接（备用）：[recognition.mp4](assets/videos/recognition.mp4)

### 2) ArUco 视觉跟随 (Visual Following)

<video src="assets/videos/following.mp4" controls width="720">
  Your browser does not support the video tag.
</video>

演示链接（备用）：[following.mp4](assets/videos/following.mp4)

> 若当前页面未直接预览视频，请点击链接在新标签页打开。

---

## 🛠️ 环境依赖 (Requirements)

请确保树莓派已连接 Picamera2 及飞控（通过 UART）。

```bash
# 1. 激活虚拟环境
source ~/rpi_env/bin/activate

# 2. 安装依赖库
pip install -r requirements.txt
```

**关键库版本：**
*   `opencv-python` (视觉处理)
*   `mediapipe` (手势识别)
*   `easyocr` (数字识别)
*   `picamera2` (相机驱动)
*   `torch` (深度学习后端)

---

## 🚀 快速开始 (Quick Start)

### 1. 启动系统
```bash
python vision_final.py
```

### 2. 交互流程 (Phase 1)
系统启动后，站在摄像头前进行手势控制：

| 目标模式           | 触发手势               | 确认操作       | 功能描述                                            |
| :----------------- | :--------------------- | :------------- | :-------------------------------------------------- |
| **ArUco 跟随模式** | **✊ 握拳 (Fist)**      | 保持握拳 1秒   | 调用 `aruco_dectction_pi5`，无人机跟随 ArUco 移动。 |
| **数字侦察模式**   | **🖐️ 张开五指 (Hover)** | 再次张开五指   | 启动智能搜索闭环，寻找指定的数字门牌。              |
| **全局重置**       | **✊ 握拳 (Fist)**      | (任意选择阶段) | 若选错数字或想取消，握拳可立即返回初始状态。        |

> **数字选择细节**：进入数字模式后，伸出 **1, 2, 3 或 4** 根手指来选择目标数字，确认无误后再次张开五指 (Hover) 即可起飞。

### 3. 任务执行流程 (Phase 2 - 仅数字模式)
一旦确认数字目标，无人机将自动执行以下闭环：

1.  **LOCK 搜索**：向右平飞 3 秒（寻找门框）。
2.  **HOVER 检测**：悬停，等待画面出现完整的门框（4个 ArUco）。
3.  **OCR 识别**：
    *   ✅ **匹配成功**：打印日志，拍照保存至 `/home/pi/captured_images`，任务结束。
    *   ❌ **匹配失败**：打印 "Mismatch"，自动切回 **LOCK 搜索** 状态，飞往下一个位置。

---

## 📂 文件结构说明 (File Structure)

```text
.
├── vision_final.py          # [主程序] 集成了手势交互 + 智能搜索闭环 (推荐运行)
├── test2.py                 # [测试脚本] 仅用于测试 OCR 算法逻辑 (无复杂手势)
├── aruco_dectction_pi5.py   # [外部模块] 被主程序调用的 ArUco 纯跟随逻辑
├── datalink_serial.py       # [驱动] 飞控通信接口 (MAVLink封装)
├── biaoding.py              # 单目测距功能实现及摄像头畸变系数的标定
├── assets/
│   └── videos/
│       ├── recognition.mp4  # OCR 数字识别演示
│       └── following.mp4    # ArUco 视觉跟随演示
├── requirements.txt         # 依赖列表
└── readme.md                # 说明文档
```

---

## ⚠️ 注意事项 (Safety & Notes)

1.  **紧急停止 (Emergency Stop)**
    *   运行过程中按键盘 **`s`** 键，程序会发送停止指令。
    *   按 **`q`** 键退出程序。

2.  **相机冲突**
    *   `vision_final.py` 在进入 ArUco 模式时会自动释放摄像头资源，以便外部脚本调用。请勿强行终止，否则可能导致相机资源被占用（需重启解决）。

3.  **光照与环境**
    *   OCR 识别对光照敏感，虽然加入了灰度化和自适应缩放，但强逆光仍可能影响识别率。
    *   请确保 ArUco 门框尺寸约为实际环境中的 15cm-20cm 左右以获得最佳 ROI 提取效果。

---

## 📝 开发者信息

*   **项目名称**：微机原理与微系统期末项目 - 智能消防救援无人机
*   **开发者**：胡伟毅12313505 && 姚瑞诣12312816
*   **日期**：2026-1-3
