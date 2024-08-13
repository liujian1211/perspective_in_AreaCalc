import os

image_file_path = 'D:/test_perspective/dataset/images'
label_file_path = 'D:/test_perspective/dataset/labels'

image_files = os.listdir(image_file_path)
label_files = os.listdir(label_file_path)

valid_labels = set()

#*******************删掉多余的label*******************
# for image_file in image_files:
#     image_name = os.path.splitext(image_file)[0]
#     valid_labels.add(image_name)

# for label_file in label_files:
#     label_name = os.path.splitext(label_file)[0]
#     if label_name not in valid_labels:
#         os.remove(os.path.join(label_file_path,label_file))
#         print(f'删除标签文件{label_file}')
# *******************删掉多余的label*******************

#*******************删掉多余的image*******************
for label_file in label_files:
    label_name = os.path.splitext(label_file)[0]
    valid_labels.add(label_name)

for image_file in image_files:
    image_name = os.path.splitext(image_file)[0]
    if image_name not in valid_labels:
        os.remove(os.path.join(image_file_path,image_file))
        print(f'删除多余的图片文件{image_file}')
