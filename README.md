- 基于flask与前端交互，实现现场图片在手机和后端的收发
- 根据用户在手机上通过点击四个点，确定ROI，利用掩膜抠出ROI
- 利用二值化、角点检测、透视变化传统CV方法获取矫正布的正视图，并结合矫正布的实际尺寸计算转换矩阵
- 保存转换系数至数据库，后续根据识别框的像素坐标，结合转换系数计算得到识别对象的实际坐标
- 根据现场图像加入系数，修正识别对象的实际坐标
 ![9c5d22e5f21548d2adfdf96baf3b50f](https://github.com/user-attachments/assets/04e6c032-640e-4b9f-a3f1-3b1ad7ce033a)
