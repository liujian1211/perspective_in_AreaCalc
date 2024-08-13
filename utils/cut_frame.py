import cv2

def extract_frames(video_path, output_dir, frame_interval):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)
    # 获取视频的帧率信息
    fps = video.get(cv2.CAP_PROP_FPS)
    # 设置帧间隔
    frame_interval = int(fps) * frame_interval

    # 循环读取视频帧
    count = 0
    while True:
        success, frame = video.read()

        # 如果没有成功读取帧，则退出循环
        if not success:
            break

        # 每隔frame_interval帧保存一次帧
        if count % frame_interval == 0:
            # 拼接帧文件名
            output_path = output_dir + "/frame_" + str(count) + ".jpg"
            # 保存帧为JPEG图片
            cv2.imwrite(output_path, frame)

        count += 1

    # 释放视频文件句柄
    video.release()

# 使用示例
video_path = "D:/test_perspective/test_videos_pics/useful.mp4"  # 视频文件路径
output_path = "D:/test_perspective/dataset/images"  # 存储帧图像的文件夹路径
frame_interval = 1

extract_frames(video_path, output_path, frame_interval)




