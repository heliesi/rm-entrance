import numpy as np
import cv2
import glob

# ================= 1. 参数设置 =================
# 棋盘格内部角点的行列数（注意：是内部交点个数，不是方块个数）
# 例如 10x7 的方块，内部角点就是 9x6
CHECKERBOARD = (9, 6) 
SQUARE_SIZE = 19.0  # 物理方块的边长，单位可以是毫米(mm)

# 亚像素精确化时的迭代终止条件 (最大迭代30次 或 精度达到0.001)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ================= 2. 准备 3D 世界坐标点 =================
# 创建一个形如 (9*6, 3) 的矩阵，存放 (0,0,0), (25,0,0), (50,0,0) ...
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
# np.mgrid 生成网格坐标，然后乘以物理尺寸
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

# 用于存储所有图像的 3D 点(世界坐标)和 2D 点(像素坐标)
objpoints = [] 
imgpoints = [] 

# ================= 3. 提取每张图像的角点 =================
# 读取文件夹下所有的 jpg 图像
images = glob.glob('calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 如果找到了，就添加到数组中
    if ret == True:
        objpoints.append(objp)

        # 进一步提取亚像素精度的角点
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # 可选：在图像上画出角点并显示，方便你检查算法是否找对了
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# ================= 4. 执行相机标定 =================
print("正在计算相机内参，请稍候...")
# 返回值：误差, 内参矩阵, 畸变系数, 旋转向量, 平移向量
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n=== 标定结果 ===")
print("内参矩阵 (Camera Matrix):\n", mtx)
print("\n畸变系数 (Distortion Coefficients):\n", dist)

# ================= 5. 计算重投影误差 =================
# 这是一个重要指标：把 3D 点用算出来的内参重新投影成 2D 点，算一下和实际像素点的距离差
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("\n总平均重投影误差 (越接近0越好): {:.4f} 像素".format(mean_error / len(objpoints)))