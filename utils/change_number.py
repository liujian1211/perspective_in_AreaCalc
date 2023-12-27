import os

folder_path = 'D:/yolov5-master-test/datasets/manhole/labels/train'
# folder_path = 'D:/yolov5-master-test/datasets/manhole/labels/val'

# 获取文件夹中所有txt文件的路径
txt_files = [file for file in os.listdir(folder_path) if file.endswith(".txt")]

for txt_file in txt_files:
    file_path = os.path.join(folder_path, txt_file)
    with open(file_path, "r") as f:
        lines = f.readlines()

    with open(file_path, "w") as f:
        for line in lines:
            # 获取行中的每个单词
            words = line.split()
            if len(words) > 0:
                # 将第一个数字改为0
                words[0] = "0"
            # 将修改后的行写入文件
            f.write(" ".join(words) + "\n")