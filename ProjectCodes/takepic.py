import os
import cv2
import libcamera
from picamera2 import Picamera2

def Camera_Init():
    global picamera
    picamera = Picamera2()
    config = picamera.create_preview_configuration(main={"format": "RGB888", "size": (1280, 720)})
    config["transform"] = libcamera.Transform(hflip=0, vflip=0)
    picamera.configure(config)
    picamera.start()

def Video_Demo():
    global index
    while True:
        video_demo = picamera.capture_array()
        # 显示本地画面（可选）
        cv2.imshow('CSI-Camera window', video_demo)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            image_path = os.path.join(output_dir,f"images_{index}.jpg")
            index += 1
            cv2.imwrite(image_path,video_demo)
            print(f"saved in {image_path}")
        elif key == ord('q'):
            print("quit now")
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    index = 0
    output_dir = "./images"
    Camera_Init()
    Video_Demo()