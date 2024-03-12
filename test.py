from PIL import Image, ImageDraw
import cv2
import numpy as np

# 打开图像
image = Image.open('/home/ps/honghesong/test_perspective/saved_images/test_lines.jpg')

image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# 定义四个点的坐标
points = np.array([(50, 50), (350, 50), (350, 200), (50, 200)], np.int32)

mask = np.zeros(image_cv.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [points], (255))

# 在原图上根据掩膜图像抠出多边形区域
image_crop = cv2.bitwise_and(image_cv, image_cv, mask=mask)

#二值化的图
_,binary = cv2.threshold(image_crop,127,255,cv2.THRESH_BINARY)

#将二值化的图转为灰度图
gray_image = cv2.cvtColor(binary,cv2.COLOR_BGR2GRAY)

# 运行Harris角点检测算法
# corners = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)
# corners = cv2.dilate(corners, None)  # 对角点进行膨胀操作，增加角点的突出性

corners = cv2.goodFeaturesToTrack(gray_image, 100, 0.01, 10)  # 使用Good Features to Track方法检测角点

#凸包算法筛选出最外面的4个点
hull = cv2.convexHull(corners)
# 绘制凸包
for point in hull:
    x, y = point[0]
    cv2.circle(image_cv, (int(x), int(y)), 5, (0,0,255), -1)

# 绘制角点
# threshold = 0.1  # 设定角点的阈值
# image_with_corners = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
# image_with_corners[corners > threshold * corners.max()] = [0, 0, 255]  # 标记角点的位置为红色

cv2.imwrite('./saved_images/hull.jpg',image_cv)
