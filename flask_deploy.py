import os
from flask import Flask, request,send_file
import numpy as np
import cv2
from PIL import Image,ImageDraw
import requests
import json
import  sqlite3
import pickle
import ast
from io import BytesIO
import base64
import urllib.parse

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

#角点检测
def corners_det(image,points,corners_quality,euclidean_dist):
    # corners_quality 和 euclidean_dist的数据类型为str

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mask = np.zeros(image_cv.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], (255))

    # 在原图上根据掩膜图像抠出多边形区域
    image_crop = cv2.bitwise_and(image_cv, image_cv, mask=mask)

    #二值化的图
    _,binary = cv2.threshold(image_crop,127,255,cv2.THRESH_BINARY)

    #将二值化的图转为灰度图
    gray_image = cv2.cvtColor(binary,cv2.COLOR_BGR2GRAY)

    # 使用Good Features to Track方法检测角点
    corners = cv2.goodFeaturesToTrack(gray_image, 4, float(corners_quality), float(euclidean_dist))  
    #后期把0.1改为corners_quality，20改为euclidean_dist

    #凸包算法筛选出最外面的4个点
    hull = cv2.convexHull(corners)

    # 绘制凸包
    for point in hull:
        x, y = point[0]
        cv2.circle(image_cv, (int(x), int(y)), 5, (0,0,255), -1) 

    cv2.polylines(image_cv,[points],isClosed=True,color=(0,255,0),thickness=2)

    # 将图像从BGR转换为RGB
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    # 将图像转换为PIL格式
    image_pil = Image.fromarray(image_rgb)

    return image_pil,hull

@app.route('/Get_Image_Name', methods=['GET'])
def get_image_name():
    car_number = request.args.get('car_number')  #获取车牌号
    device_number = request.args.get('device_number')  #获取设备号
    image_name = request.args.get('image_name')  #获取第一张图，此时返回的是url

    corners_quality = request.args.get("corners_quality") #滑动条的形式获取角点质量，值越大，点越少，反之越多
    euclidean_dist = request.args.get("euclidean_dist") #滑动条的形式获取欧式距离,值越大，点越疏，反之越密

    p1 = request.args.get('p1')  
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

    if image_name: #image_name为网络图片格式，即https://devtdc.iquanzhan.com/himsFile/upload/flvcut/flv_video_13515277111_221.png

        response = requests.get(image_name, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw)

        points = np.array([(p1_x/3.75,p1_y/3.6),(p2_x/3.75,p2_y/3.6),(p3_x/3.75,p3_y/3.6),(p4_x/3.75,p4_y/3.6)],np.int32)
        image_lines,corners = corners_det(image=image,points=points,corners_quality=corners_quality,euclidean_dist=euclidean_dist)  #角点检测
        # image_lines.save("./saved_images/test_lines.jpg")
        
        #将绘制了线段的图像转换为字节流
        image_byte = BytesIO() 
        image_lines.save(image_byte,format='JPEG') 
        
        # #将图像字节流编码为Base64字符串，以便后期传给json
        image_base64 = base64.b64encode(image_byte.getvalue()).decode('utf-8')

        # 将字符串形式的图像数据转换为URL格式
        image_url = 'data:image/jpeg;base64,' + image_base64

        src_points = corners

        if src_points.shape[0] == 4:
            dst_points = np.float32([
                    [0, 0],  #左上角
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
            json_data = {"errorCode": 0, "errorMessage": "生成校准矩阵完毕","content":image_url}
            json_str = json.dumps(json_data)
            print('校准完毕')
            return json_str
        else:
            json_data = {"errorCode": 1, "errorMessage": "检测到不足4个点","content":image_url}
            json_str = json.dumps(json_data)
            return json_str

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

    corners_quality = request.args.get("corners_quality") #滑动条的形式获取角点质量，值越大，点越少，反之越多
    euclidean_dist = request.args.get("euclidean_dist") #滑动条的形式获取欧式距离,值越大，点越疏，反之越密

    correct_p1 = request.args.get('correct_p1')  
    correct_p2 = request.args.get('correct_p2')
    correct_p3 = request.args.get('correct_p3')
    correct_p4 = request.args.get('correct_p4')

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
    else:
        json_data = {"errorCode": 1, "errorMessage": "获取车牌号失败"}
        json_str = json.dumps(json_data)
        print('获取车牌号失败')
        return json_str

    #关闭数据库连接
    cursor.close()
    conn.close()
    # ************************************** 在数据库中根据车牌号提取出M **************************************

    if test_image_name:

        response = requests.get(test_image_name, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw)

        correct_p1_x = float(correct_p1.split(',')[0])
        correct_p1_y = float(correct_p1.split(',')[1])

        correct_p2_x = float(correct_p2.split(',')[0])
        correct_p2_y = float(correct_p2.split(',')[1])

        correct_p3_x = float(correct_p3.split(',')[0])
        correct_p3_y = float(correct_p3.split(',')[1])

        correct_p4_x = float(correct_p4.split(',')[0])
        correct_p4_y = float(correct_p4.split(',')[1])

        points = np.array([(correct_p1_x/3.75,correct_p1_y/3.6),(correct_p2_x/3.75,correct_p2_y/3.6),
                           (correct_p3_x/3.75,correct_p3_y/3.6),(correct_p4_x/3.75,correct_p4_y/3.6)],np.int32)
        image_lines,corners = corners_det(image=image,points=points,corners_quality=corners_quality,euclidean_dist=euclidean_dist)  #角点检测

        #将绘制了线段的图像转换为字节流
        image_byte = BytesIO() 
        image_lines.save(image_byte,format='JPEG') 
        
        # #将图像字节流编码为Base64字符串，以便后期传给json
        image_base64 = base64.b64encode(image_byte.getvalue()).decode('utf-8')

        # 将字符串形式的图像数据转换为URL格式
        image_url = 'data:image/jpeg;base64,' + image_base64
        
        if corners.shape[0] == 4:
            src_p1 = tuple(corners[0][0])
            src_p2 = tuple(corners[1][0])
            src_p3 = tuple(corners[2][0])
            src_p4 = tuple(corners[3][0])

            print(f'棋盘角点变换前的4个顶点坐标为:{src_p1},{src_p2},{src_p3},{src_p4}')

            # 得到变换前的4个角点坐标
            # src_p1 = (correct_p1_x, correct_p1_y)
            # src_p2 = (correct_p2_x, correct_p2_y)
            # src_p3 = (correct_p3_x, correct_p3_y)
            # src_p4 = (correct_p4_x, correct_p4_y)
            ret_p1, ret_p2, ret_p3, ret_p4 = transformed_cor(M, src_p1, src_p2, src_p3, src_p4)
            
            print(f'棋盘角点变换后的4个顶点坐标为:{ret_p1},{ret_p2},{ret_p3},{ret_p4}')

            area_corners = cal_area(ret_p1,ret_p2,ret_p3,ret_p4) #计算像素面积
            true_area_corners = round(area_corners/scale,2) #计算实际面积，保留2位小数
            print(f'棋盘角点的实际面积为{str(round(area_corners/scale,2))}')

            json_data_child = {"true_area_corners":true_area_corners, "image_url":image_url}
            json_data = {"content": json_data_child, "errorCode": 0, "errorMessage": '校验成功'}
            json_str = json.dumps(json_data)
            return json_str
        else:
            json_data = {"content":image_url,"errorCode": 1, "errorMessage": "角点不足四个点"}
            json_str = json.dumps(json_data)
            print('角点不足四个点')
            return json_str

    else:
        json_data = {"errorCode": 1, "errorMessage": "第二个图上传失败"}
        json_str = json.dumps(json_data)
        print('第二个图上传失败')
        return json_str

if __name__ == '__main__':
    app.run('10.0.1.120', 8888, debug=False)  # 本机IP
