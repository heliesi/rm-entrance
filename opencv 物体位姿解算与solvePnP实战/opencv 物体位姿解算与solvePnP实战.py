import cv2
import numpy as np

# ================== 1. 参数配置 ==================
# 你画的/打印的矩形的真实物理尺寸（单位：毫米）
RECT_WIDTH = 100.0  
RECT_HEIGHT = 50.0  

# 相机内参矩阵 (1920x1080 分辨率)
camera_matrix = np.array([
    [1366.42,       0.0,  952.87],
    [      0.0, 1391.09,  538.14],
    [      0.0,       0.0,      1.0]
], dtype=np.float32)

# 假设无畸变
dist_coeffs = np.zeros((4, 1), dtype=np.float32)

# ================== 2. 构建 objectPoints (3D) ==================
# 按照文档提示，以矩形面中心为原点 (0,0,0)，Z轴垂直纸面
# 顺序必须严格为：左上, 右上, 右下, 左下
object_points = np.array([
    [-RECT_WIDTH / 2, -RECT_HEIGHT / 2, 0.0],
    [ RECT_WIDTH / 2, -RECT_HEIGHT / 2, 0.0],
    [ RECT_WIDTH / 2,  RECT_HEIGHT / 2, 0.0],
    [-RECT_WIDTH / 2,  RECT_HEIGHT / 2, 0.0]
], dtype=np.float32)

# 定义要画出的 3D 坐标轴上的点（分别代表 原点, X轴端点, Y轴端点, Z轴端点）
# Z轴使用负值，是为了让蓝色的Z轴“戳出”纸面（OpenCV遵循右手坐标系，Z正方向默认指向屏幕内）
axis_length = 50.0 # 坐标轴画多长 (mm)
axis_points = np.array([
    [0.0, 0.0, 0.0],
    [axis_length, 0.0, 0.0],
    [0.0, axis_length, 0.0],
    [0.0, 0.0, -axis_length]
], dtype=np.float32)

# ================== 辅助函数：角点排序 ==================
def order_points(pts):
    """将找到的4个点按照 左上, 右上, 右下, 左下 的顺序排列"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # 左上：x+y 最小
    rect[2] = pts[np.argmax(s)] # 右下：x+y 最大
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # 右上：y-x 最小
    rect[3] = pts[np.argmax(diff)] # 左下：y-x 最大
    return rect

# ================== 3. 实时视频流处理 ==================
cap = cv2.VideoCapture(0) # 打开默认摄像头
# 尝试设置分辨率与内参匹配
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("按 'q' 键退出程序")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # --- 图像预处理，寻找黑框 ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 高斯滤波去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 二值化（提取黑色部分）你可以根据光线调整 100 这个阈值
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        # 面积过滤，太小的噪点不要
        if cv2.contourArea(cnt) < 2000:
            continue
            
        # 多边形拟合
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # 如果拟合出来是4个点，我们就认为找到了矩形
        if len(approx) == 4:
            # 拿到 2D 像素坐标 (imagePoints)
            pts = approx.reshape(4, 2)
            # 排序：严格对应 objectPoints 的顺序
            image_points = order_points(pts)
            
            # --- 核心：solvePnP 位姿解算 ---
            success, rvec, tvec = cv2.solvePnP(
                object_points, 
                image_points, 
                camera_matrix, 
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # --- 投影：projectPoints 画三维坐标轴 ---
                projected_axis_points, _ = cv2.projectPoints(
                    axis_points, rvec, tvec, camera_matrix, dist_coeffs
                )
                
                # 转换数据格式以适应画线函数
                p = np.int32(projected_axis_points).reshape(-1, 2)
                origin = tuple(p[0])
                
                # 画出矩形边框（可选，用于确认检测正常）
                cv2.polylines(frame, [np.int32(image_points)], True, (0, 255, 255), 2)
                
                # 画 XYZ 三维坐标轴 (粗细为3)
                # X轴 - 红色 (BGR: 0, 0, 255)
                cv2.line(frame, origin, tuple(p[1]), (0, 0, 255), 3)
                # Y轴 - 绿色 (BGR: 0, 255, 0)
                cv2.line(frame, origin, tuple(p[2]), (0, 255, 0), 3)
                # Z轴 - 蓝色 (BGR: 255, 0, 0)
                cv2.line(frame, origin, tuple(p[3]), (255, 0, 0), 3)
                
                # 给原点画个小圆圈
                cv2.circle(frame, origin, 5, (255, 255, 255), -1)

    # 显示画面
    cv2.imshow('solvePnP 3D Pose', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()