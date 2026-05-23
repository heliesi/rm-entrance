import cv2
import numpy as np

def detect_station(image_path):
    image = cv2.imread(image_path)
    output_image = image.copy()
    # 1. 转灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化阈值提升至 235，从源头滤除不够亮的纸张和微弱反光

    _, binary2 = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    # 2. 寻找所有轮廓 

    contours, _ = cv2.findContours(binary2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:

        # 最小面积阈值提升至 350
        if cv2.contourArea(contour) < 350:
            continue
        # 3. 多边形拟合 

        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # 4. 核心特征筛选：六边形

        if len(approx) == 6:
            # 5. 最小框选 
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box) 

            cv2.drawContours(output_image, [box], 0, (0, 0, 255), 3)

    cv2.imwrite("param_tuned_result.jpg", output_image)
    cv2.imshow("img", output_image)
    cv2.waitKey(0)


# --- 运行测试 ---
detect_station('opencv2.png')