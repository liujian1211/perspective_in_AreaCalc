import cv2
# from chessboard2 import transformed_cor
import numpy as np
import math

# M = np.float32([[5.53458814e+00, 2.79305947e+00, -3.90165559e+03],
#      [7.43505510e-01, 1.18716757e+01, -7.19985889e+03],
#      [3.34179619e-03, 2.64147306e-02, 1.00000000e+00]])

#chessboard1.jpg的M
# M=np.float32([[-2.81181842e+01 ,-2.38852247e+00 , 5.13690986e+04],
#  [ 1.43381431e+00 ,-3.14792058e+01 , 9.39616533e+04],
#  [ 4.38416386e-03 , 8.60911017e-02 , 1.00000000e+00]])

M = np.float32([[ 2.87298890e+01 , 1.44402299e+01, -5.12253878e+04],
 [-8.19973819e-01 , 4.35288109e+01 ,-9.38148644e+04],
 [ 4.38383343e-03 , 8.60848571e-02 , 1.00000000e+00]])

scale = 100

def transformed_cor(M, p1, p2, p3, p4):
    p1 = p1 + (1,)
    p2 = p2 + (1,)
    p3 = p3 + (1,)
    p4 = p4 + (1,)

    ret1 = np.dot(M, p1)
    ret2 = np.dot(M, p2)
    ret3 = np.dot(M, p3)
    ret4 = np.dot(M, p4)

    p1_x = round(ret1[0] / ret1[2], 2)
    p1_y = round(ret1[1] / ret1[2], 2)

    p2_x = round(ret2[0] / ret2[2], 2)
    p2_y = round(ret2[1] / ret2[2], 2)

    p3_x = round(ret3[0] / ret3[2], 2)
    p3_y = round(ret3[1] / ret3[2], 2)

    p4_x = round(ret4[0] / ret4[2], 2)
    p4_y = round(ret4[1] / ret4[2], 2)

    point1 = ()
    point2 = ()
    point3 = ()
    point4 = ()

    point1 = point1 + (p1_x, p1_y)
    point2 = point2 + (p2_x, p2_y)
    point3 = point3 + (p3_x, p3_y)
    point4 = point4 + (p4_x, p4_y)
    return point1, point2, point3, point4,

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def area_of_quadrilateral(p1, p2, p3, p4):
    a = distance(p1, p2)
    b = distance(p2, p3)
    c = distance(p3, p4)
    d = distance(p4, p1)
    s = (a + b + c + d) / 2
    return round(math.sqrt((s - a) * (s - b) * (s - c) * (s - d)), 2)

image = cv2.imread('D:/test_perspective/chessboard_correct1.jpg')

#chessboard1.jpg的外围矩形
# obj_p1 = (1694, 2873)
# obj_p2 = (2185, 2864)
# obj_p3 = (2202, 3282)
# obj_p4 = (1691, 3282)

#chessboard2.jpg的外围矩形
# obj_p1 = (1740, 2358)
# obj_p2 = (2139, 2358)
# obj_p3 = (2149, 2638)
# obj_p4 = (1743, 2638)

#chessboard3.jpg的外围矩形
obj_p1 = (1792, 2012)
obj_p2 = (2153, 1998)
obj_p3 = (2162, 2229)
obj_p4 = (1805, 2238)

ret_p1, ret_p2, ret_p3, ret_p4 = transformed_cor(M, obj_p1, obj_p2, obj_p3, obj_p4)
print(f'瓶盖变换前的4个顶点坐标为：{obj_p1},{obj_p2},{obj_p3},{obj_p4}')
print(f'瓶盖变换后的4个顶点坐标为：{ret_p1},{ret_p2},{ret_p3},{ret_p4}')
area_bottle = area_of_quadrilateral(ret_p1,ret_p2,ret_p3,ret_p4)
print(f'瓶盖的像素面积为{area_bottle}')
print(f'瓶盖的实际面积为{round(area_bottle/scale,2)}')

cv2.circle(image, obj_p1, 10, (0, 0, 255), -1)
cv2.circle(image, obj_p2, 10, (0, 0, 255), -1)
cv2.circle(image, obj_p3, 10, (0, 0, 255), -1)
cv2.circle(image, obj_p4, 10, (0, 0, 255), -1)

cv2.line(image, obj_p1, obj_p2, (0, 255, 0), 3)
cv2.line(image, obj_p2, obj_p3, (0, 255, 0), 3)
cv2.line(image, obj_p3, obj_p4, (0, 255, 0), 3)
cv2.line(image, obj_p4, obj_p1, (0, 255, 0), 3)

cv2.putText(image,'Real Area of bounding box:'+str(round(area_bottle/scale,2))+'cm**2',(800,3500),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=3,color=(0,0,255),thickness=10)

cv2.namedWindow('bottle', cv2.WINDOW_NORMAL)
cv2.imshow('bottle', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
