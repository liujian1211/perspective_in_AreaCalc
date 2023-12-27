import os
from flask import Flask, request
import numpy as np
import cv2
from PIL import Image
import requests
import json
<<<<<<< HEAD
import  sqlite3
import pickle
import ast
=======
>>>>>>> ce0564bf044b5106cd8982755bd248a3b7a341e6

app = Flask(__name__)

scale = 100

@app.route('/Upload_Image', methods=['POST'])
def upload_image():
    # 接收的图片或字符串
    image_file = request.files.get('file')  # 上传的图片文件
    if image_file:
        file_name = image_file.filename
        image_file.save('./saved_images/' + file_name)
        return file_name
    else:
        return '没有上传成功'

@app.route('/Get_Image_Name', methods=['GET'])
def get_image_name():
    car_number = request.args.get('car_number')  #获取车牌号
    device_number = request.args.get('device_number')  #获取设备号
    image_name = request.args.get('image_name')  #获取第一张图，此时返回的是url
    p1 = request.args.get('p1')  #p1:'321.72,241.92'
    p2 = request.args.get('p2')
    p3 = request.args.get('p3')
    p4 = request.args.get('p4')

    p1_x = float(p1.split(',')[0])
    p1_y = float(p1.split(',')[1])

    p2_x = float(p2.split(',')[0])
    p2_y = float(p2.split(',')[1])

    p3_x = float(p3.split(',')[0])
    p3_y = float(p3.split(',')[1])

    p4_x = float(p4.split(',')[0])
    p4_y = float(p4.split(',')[1])

    square_size = 20  # 可以控制缩放比例

    if image_name:
        # print(f'p1的值为{p1}')
        # print(f'p2的值为{p2}')
        # print(f'p3的值为{p3}')
        # print(f'p4的值为{p4}')

        response = requests.get(image_name, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw)
<<<<<<< HEAD
        # image.save('./saved_images/first_img.jpg')
        # img1 = cv2.imread('./saved_images/first_img.jpg')
=======
        image.save('./saved_images/first_img.jpg')

        img1 = cv2.imread('./saved_images/first_img.jpg')
>>>>>>> ce0564bf044b5106cd8982755bd248a3b7a341e6

        #**************改为手动添加点2023-11-15**************
        src_points = np.float32([
            [p1_x, p1_y],
            [p2_x, p2_y],
            [p3_x, p3_y],
            [p4_x, p4_y]
        ]) #从左上角为起点开始顺时针

        dst_points = np.float32([
                [0, 0],  #左上角
<<<<<<< HEAD
                [440, 0], #右上角
                [440, 240], #右下角
                [0, 240] #左下角
            ])
        #单位：cm,2张棋盘格叠加

        # ************************************** M保存在数据库中 **************************************
        # 连接到SQLite数据库
        conn = sqlite3.connect('database.db')

        #创建一个游标对象
        cursor = conn.cursor()

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        # warped_image = cv2.warpPerspective(img1, M, (square_size * (16 - 2), square_size * (12 - 2)))
        # cv2.imwrite('./saved_images/img1_warped.jpg',warped_image)

        #将M转为字符串格式
        m_str = np.array2string(M, separator=',')

        #创建表数据表（如果不存在）：表名M，列名car_number , m_value
        cursor.execute("CREATE TABLE IF NOT EXISTS M (car_number, m_value)")

        # 插入透视变换矩阵M的值到表中,根据车牌号将各自的m_str插入到表中。
        cursor.execute("INSERT INTO M (car_number,m_value) VALUES (?,?)", (car_number,m_str))

        # 保存更改
        conn.commit()

        # 关闭数据库连接
        conn.close()
        # ************************************** M保存在数据库中 **************************************

        # np.savetxt('./saved_M/' + car_number + '.txt', M) #舍弃掉原来的保存在本地的做法
=======
                [320, 0], #右上角
                [320, 240], #右下角
                [0, 240] #左下角
            ])
        #单位：cm,4张棋盘格叠加

        # dst_points = np.float32([
        #     [0,0],
        #     [160,0],
        #     [160,120],
        #     [0,120]
        # ])
        #单位：cm,1张棋盘格

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_image = cv2.warpPerspective(img1, M, (square_size * (16 - 2), square_size * (12 - 2)))
        cv2.imwrite('./saved_images/img1_warped.jpg',warped_image)
        np.savetxt('./saved_M/' + car_number + '.txt', M)
>>>>>>> ce0564bf044b5106cd8982755bd248a3b7a341e6
        json_data = {"errorCode": 0, "errorMessage": "生成校准矩阵完毕"}
        json_str = json.dumps(json_data)
        print('校准完毕')
        return json_str
        # **************改为手动添加点2023-11-15**************

        # response = requests.get(image_name, stream=True)
        # response.raise_for_status()
        # image = Image.open(response.raw)
        # image.save('./saved_images/first_img.jpg')
        #
        # #若采用网络读取
        # img1 = cv2.imread('./saved_images/first_img.jpg')
        #
        # # 定义棋盘格的行数和列数
        # rows = 16
        # cols = 16
        #
        # # 棋盘格的交叉点数目
        # num_corners = (cols - 1) * (rows - 1)
        #
        # # 设置棋盘格方块的大小
        # square_size = 20  # 可以控制缩放比例
        #
        # # 计算棋盘格的交叉点坐标
        # obj_points = np.zeros((num_corners, 3), np.float32)
        # obj_points[:, :2] = np.mgrid[0:cols - 1, 0:rows - 1].T.reshape(-1, 2) * square_size
        #
        # # 转换为灰度图像
        # gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        #
        # ret, corners = cv2.findChessboardCorners(gray, (cols - 1, rows - 1), None)
        #
        # if ret:
        #     # 亚像素级角点检测
        #     corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
        #                                 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        #
        #     # 绘制角点
        #     cv2.drawChessboardCorners(img1, (cols - 1, rows - 1), corners2, ret)
        #
        #     # cv2.imwrite('./img1_corners.jpg',img1)
        #
        #     # 定义棋盘格的目标点和图像点
        #     src_points = np.float32([corners2[0], corners2[cols - 2], corners2[corners2.shape[0] - cols + 1],
        #                              corners2[corners2.shape[0] - 1]])
        #
        #     dst_points = np.float32([
        #         [0, 140],
        #         [0, 0],
        #         [140, 140],
        #         [140, 0]
        #     ])
        #
        #     # 计算透视变换矩阵
        #     M = cv2.getPerspectiveTransform(src_points, dst_points)
        #     warped_image = cv2.warpPerspective(img1, M, (square_size * (cols - 2), square_size * (rows - 2)))
        #     # cv2.imwrite('./img1_warped.jpg',warped_image)
        #     np.savetxt('./saved_M/'+car_number+'.txt',M)
        #     json_data = {"errorCode": 0, "errorMessage": "找到棋盘格图片"}
        #     json_str = json.dumps(json_data)
        #     print('校准完毕')
        #     return json_str
        # else:
        #     json_data = {"errorCode": 1, "errorMessage": "未找到棋盘格图片"}
        #     json_str = json.dumps(json_data)
        #     print('未找到棋盘格')
        #     return json_str
        #     # return '未找到棋盘格'
    else:
        json_data = {"errorCode": 1, "errorMessage": "上传第一张图片失败"}
        json_str = json.dumps(json_data)
        print('上传第一张图片失败')
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

@app.route('/Correct_Result', methods=['GET'])
def correct_result():
    test_image_name = request.args.get('test_image_name')
    car_number = request.args.get('car_number')

    correct_p1 = request.args.get('correct_p1')  # p1:'321.72,241.92'
    correct_p2 = request.args.get('correct_p2')
    correct_p3 = request.args.get('correct_p3')
    correct_p4 = request.args.get('correct_p4')

<<<<<<< HEAD
    # ************************************** 在数据库中根据车牌号提取出M **************************************
    #连接到SQLite数据库
    conn = sqlite3.connect('database.db')

    #创建一个游标对象
    cursor = conn.cursor()

    # 根据车牌号检索保存的透视变换矩阵M的值
    cursor.execute("SELECT m_value FROM M WHERE car_number = ?", (car_number,))

    # 从游标中获取透视变换矩阵M的值
    m_str = cursor.fetchone()[0]

    #将m_str转为numpy格式的M
    m_list = ast.literal_eval(m_str)
    M = np.array(m_list)

    if M.any():
        print(f'M的值为{M}')
    # if car_number:
    #     M = np.loadtxt('./saved_M/' + car_number +'.txt')
    #     print(f'M的值为{M}')
=======
    if car_number:
        M = np.loadtxt('./saved_M/' + car_number +'.txt')
        print(f'M的值为{M}')
>>>>>>> ce0564bf044b5106cd8982755bd248a3b7a341e6
    else:
        json_data = {"errorCode": 1, "errorMessage": "获取车牌号失败"}
        json_str = json.dumps(json_data)
        print('获取车牌号失败')
        return json_str

<<<<<<< HEAD
    #关闭数据库连接
    cursor.close()
    conn.close()
    # ************************************** 在数据库中根据车牌号提取出M **************************************

=======
>>>>>>> ce0564bf044b5106cd8982755bd248a3b7a341e6
    if test_image_name:
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
        print(f'棋盘角点变换前的4个顶点坐标为：{src_p1},{src_p2},{src_p3},{src_p4}')
        print(f'棋盘角点变换后的4个顶点坐标为：{ret_p1},{ret_p2},{ret_p3},{ret_p4}')

        area_corners = cal_area(ret_p1,ret_p2,ret_p3,ret_p4) #计算像素面积
        true_area_corners = round(area_corners/scale,2) #计算实际面积，保留2位小数
        print(f'棋盘角点的像素面积为{area_corners}')
        print(f'棋盘角点的实际面积为{str(round(area_corners/scale,2))}')
        json_data = {"content": true_area_corners, "errorCode": 0, "errorMessage": '校验成功'}
        json_str = json.dumps(json_data)
        return json_str

        # response = requests.get(test_image_name, stream=True)
        # response.raise_for_status()
        # image = Image.open(response.raw)
        # image.save('./saved_images/second_img.jpg')
        #
        # # 若采用网络地址读取
        # img2 = cv2.imread('./saved_images/second_img.jpg')


        # # 定义棋盘格的行数和列数
        # rows = 16
        # cols = 16
        #
        # # 棋盘格的交叉点数目
        # num_corners = (cols - 1) * (rows - 1)
        #
        # # 设置棋盘格方块的大小
        # square_size = 20  # 可以控制缩放比例
        # # warped_image = cv2.warpPerspective(img2, M, (square_size * (cols - 2), square_size * (rows - 2)))
        # # cv2.imwrite('./img2_warped.jpg',warped_image)
        #
        # # 计算棋盘格的交叉点坐标
        # obj_points = np.zeros((num_corners, 3), np.float32)
        # obj_points[:, :2] = np.mgrid[0:cols - 1, 0:rows - 1].T.reshape(-1, 2) * square_size
        #
        # # 转换为灰度图像
        # gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        #
        # ret, corners = cv2.findChessboardCorners(gray, (cols - 1, rows - 1), None)
        # if ret:
        #     # 亚像素级角点检测
        #     corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
        #                                 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        #
        #     # 绘制角点
        #     cv2.drawChessboardCorners(img2, (cols - 1, rows - 1), corners2, ret)
        #
        #     # cv2.imwrite('./img2_corners.jpg',img2)
        #
        #     # 定义棋盘格的目标点和图像点
        #     src_points = np.float32([corners2[0], corners2[cols - 2], corners2[corners2.shape[0] - cols + 1],
        #                              corners2[corners2.shape[0] - 1]])
        #
        #     coordinates = src_points[:,0,:]
        #     tuple_coordinates = tuple(map(tuple,coordinates.tolist()))
        #     #得到变换前的4个角点坐标
        #     src_p1 = tuple_coordinates[0]
        #     src_p2 = tuple_coordinates[1]
        #     src_p3 = tuple_coordinates[2]
        #     src_p4 = tuple_coordinates[3]
        #     ret_p1, ret_p2, ret_p3, ret_p4 = transformed_cor(M, src_p1, src_p2, src_p3, src_p4)
        #     print(f'棋盘角点变换前的4个顶点坐标为：{src_p1},{src_p2},{src_p3},{src_p4}')
        #     print(f'棋盘角点变换后的4个顶点坐标为：{ret_p1},{ret_p2},{ret_p3},{ret_p4}')
        #
        #     area_corners = cal_area(ret_p1,ret_p2,ret_p3,ret_p4) #计算像素面积
        #     true_area_corners = round(area_corners/scale,2) #计算实际面积
        #     print(f'棋盘角点的像素面积为{area_corners}')
        #     print(f'棋盘角点的实际面积为{str(round(area_corners/scale,2))}')
        #     json_data = {"content": true_area_corners, "errorCode": 0, "errorMessage": '校验成功'}
        #     json_str = json.dumps(json_data)
        #     return json_str
        #
        # else:
        #     json_data = {"errorCode": 1, "errorMessage": "未找到第二张图片的棋盘格"}
        #     json_str = json.dumps(json_data)
        #     print('未找到第二张图片的棋盘格，请重新上传第二张测试图')
        #     return json_str
    else:
        json_data = {"errorCode": 1, "errorMessage": "第二个图上传失败"}
        json_str = json.dumps(json_data)
        print('第二个图上传失败')
        return json_str

if __name__ == '__main__':
    app.run('10.0.1.34', 8888, debug=False)  # 本机IP
