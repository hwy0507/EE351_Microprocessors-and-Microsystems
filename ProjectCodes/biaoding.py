import cv2
import numpy as np
import glob

# 棋盘格规格（内角点的数量） {#棋盘格规格内角点的数量  data-source-line="52"}
chessboard_size = (9, 6)

# 准备对象点，例如(0,0,0), (1,0,0), (2,0,0) ..., (8,5,0) {#准备对象点例如000-100-200--850  data-source-line="55"}
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 存储所有的对象点和图像点 {#存储所有的对象点和图像点  data-source-line="59"}
objpoints = []  # 在真实世界坐标系中的3D点
imgpoints = []  # 在图像平面的2D点

# 获取棋盘格图片 {#获取棋盘格图片  data-source-line="63"}
images = glob.glob('/home/pi/images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 找到棋盘格的角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # 如果找到足够的角点，添加对象点和图像点
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 显示角点
        img = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 标定摄像头 {#标定摄像头  data-source-line="85"}
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("相机内参矩阵：\n", camera_matrix)
print("畸变系数：\n", dist_coeffs)

# 保存标定结果 {#保存标定结果  data-source-line="91"}
np.savez('calibration_data.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
