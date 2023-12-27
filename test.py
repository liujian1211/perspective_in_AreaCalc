import os
<<<<<<< HEAD
import ast
=======

>>>>>>> ce0564bf044b5106cd8982755bd248a3b7a341e6
import numpy as np
import cv2
import time
import json
import math

scale = 100


def get_image_name():
    square_size = 20
    img1 = cv2.imread('D:/test_perspective/test_videos_pics/correct.jpg')
    #**************改为手动添加点2023-11-15**************
    src_points = np.float32([
        [893, 826],
        [1267, 842],
        [1320, 887],
        [842, 863]
    ]) #从左上角为起点开始顺时针


    # dst_points = np.float32([
    #         [0, 0],  #左上角
    #         [320, 0], #右上角
    #         [320, 240], #右下角
    #         [0, 240] #左下角
    #     ])

    dst_points = np.float32([
        [893, 826],  # 变换后的左上角，应该与原图的左上角一致
        [893+320, 826],  # 右上角
        [893+320, 826+240],  # 右下角
        [893, 826+240]  # 左下角
    ])
    #单位：cm,4张棋盘格叠加

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    # warped_image = cv2.warpPerspective(img1, M, (square_size * (32 - 2), square_size * (24 - 2)))
    warped_image = cv2.warpPerspective(img1, M, (1920,1080))
    cv2.imwrite('./saved_images/img1_warped.jpg',warped_image)
    np.savetxt('./saved_M/'  + 'test.txt', M)
    json_data = {"errorCode": 0, "errorMessage": "生成校准矩阵完毕"}
    json_str = json.dumps(json_data)
    print('校准完毕')
    return json_str

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

def cal_area(p1,p2,p3,p4):
    area1 = abs(0.5 * ( (p1[0]*p2[1]) + (p2[0]*p4[1]) + (p4[0]*p1[1]) - (p2[0]*p1[1]) - (p4[0]*p2[1]) - (p1[1]*p4[1])))
    area2 = abs(0.5 * ( (p2[0]*p3[1]) + (p3[0]*p4[1]) + (p4[0]*p2[1]) - (p3[0]*p2[1]) - (p4[0]*p3[1]) - (p2[0]*p4[1])))
    return round(area1+area2,2)

def cal_ret_area(p1,p2,p3,p4):
    w = min(abs(p2[0]-p1[0]),abs(p3[0]-p4[0]))
    h = min(abs(p3[1]-p2[1]),abs(p4[1]-p1[1]))
    #p1和p2中y值最大的点，p3和p4中y值最小的点，组成的矩形作为面积
    # lt = p1 if p1[1]>p2[1] else p2
    # rb = p3 if p3[1]<p4[1] else p4
    # w = abs(rb[0] - lt[0])
    # h = abs(rb[1] - lt[1])
    scale_area = 1 - abs(w / h - 1)  # 该公式用来表示与1的接近程度，w/h越接近1，scale越大，即视线越近，面积应该乘以的系数就越大
    print(f'scale_area为{scale_area}')
    print(f'变换后的w为{w},h为{h}')
    return round(w * h * scale_area, 2)

def correct_result():
    # correct_p1 = request.args.get('correct_p1')  # p1:'321.72,241.92'
    # correct_p2 = request.args.get('correct_p2')
    # correct_p3 = request.args.get('correct_p3')
    # correct_p4 = request.args.get('correct_p4')
    #********ret3井盖坐标********
    correct_p1 = '900, 860'
    correct_p2 = '1037, 860'
    correct_p3 = '1037,889'
    correct_p4 = '900,889'
    # ********ret3井盖坐标********

    M = np.loadtxt('./saved_M/' +'test.txt')
    print(f'M的值为{M}')

    square_size=20
    img1 = cv2.imread('D:/test_perspective/crop_imgs/larger_test.jpg')
    # warped_image = cv2.warpPerspective(img1, M, (square_size * (32 - 2), square_size * (24 - 2)))
    warped_image = cv2.warpPerspective(img1, M, (1920,1080))

    correct_p1_x = float(correct_p1.split(',')[0])
    correct_p1_y = float(correct_p1.split(',')[1])

    correct_p2_x = float(correct_p2.split(',')[0])
    correct_p2_y = float(correct_p2.split(',')[1])

    correct_p3_x = float(correct_p3.split(',')[0])
    correct_p3_y = float(correct_p3.split(',')[1])

    correct_p4_x = float(correct_p4.split(',')[0])
    correct_p4_y = float(correct_p4.split(',')[1])

    # 得到变换前的4个角点坐标
    src_p1 = (correct_p1_x, correct_p1_y)
    src_p2 = (correct_p2_x, correct_p2_y)
    src_p3 = (correct_p3_x, correct_p3_y)
    src_p4 = (correct_p4_x, correct_p4_y)
    ret_p1, ret_p2, ret_p3, ret_p4 = transformed_cor(M, src_p1, src_p2, src_p3, src_p4)

    cv2.circle(warped_image,(int(ret_p1[0]),int(ret_p1[1])),5,(0,255,0),-1)
    cv2.circle(warped_image, (int(ret_p2[0]),int(ret_p2[1])), 5, (0, 255, 0), -1)
    cv2.circle(warped_image, (int(ret_p3[0]),int(ret_p3[1])), 5, (0, 255, 0), -1)
    cv2.circle(warped_image, (int(ret_p4[0]),int(ret_p4[1])), 5, (0, 255, 0), -1)
    cv2.imwrite('./saved_images/manhole3_warped.jpg', warped_image)

    print(f'棋盘角点变换前的4个顶点坐标为：{src_p1},{src_p2},{src_p3},{src_p4}')
    print(f'棋盘角点变换后的4个顶点坐标为：{ret_p1},{ret_p2},{ret_p3},{ret_p4}')

    # area_corners = cal_area(ret_p1,ret_p2,ret_p3,ret_p4) #计算像素面积
    area_corners = cal_ret_area(ret_p1,ret_p2,ret_p3,ret_p4)
    true_area_corners = round(area_corners,2) #计算实际面积，保留2位小数
    print(f'棋盘角点的像素面积为{area_corners}')
    print(f'棋盘角点的实际面积为{true_area_corners}平方厘米')
    json_data = {"content": true_area_corners, "errorCode": 0, "errorMessage": '校验成功'}
    json_str = json.dumps(json_data)
    return json_str

if __name__ == '__main__':
    # get_image_name()
<<<<<<< HEAD
    # correct_result()
    m_str = "[[ 1.20771149e-01, -3.20408340e-01, 4.70644005e+01], [-9.09599780e-03, -1.61812770e-01, 8.00876169e+01], [-1.44767448e-04, -1.38279140e-03, 1.00000000e+00]]"
    m_list = ast.literal_eval(m_str)
    M = np.array(m_list)

    # 打印转换后的NumPy数组
    print(M)
=======
    correct_result()
>>>>>>> ce0564bf044b5106cd8982755bd248a3b7a341e6
