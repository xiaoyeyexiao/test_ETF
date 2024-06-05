#import vedio_to_image
import json
import numpy as np
import cv2
#import process_data
#import identify_nose
import multi_ostu

"""将视频截取为图片"""
# # 视频路径
# video_path = "/home/yechentao/Programs/Test/vedio/gray10.mp4"
# # 图片保存路径
# output_directory = "/home/yechentao/Programs/Test/image/gray10"
# # 每隔多少帧截取一次
# frame_interval = 10
# # 是否只截取一张图片
# isoneimage = False
# # 将视频截取为图片
# vedio_to_image.video_to_image(video_path, output_directory, frame_interval, isoneimage)

"""识别鼻子"""
# 读取图像
image = cv2.imread('tt1.jpg')
# 转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 识别
# identify_nose.identify_nose(gray_image)

"""应用多级大津法提取出人脸"""
# classese表示需要划分的类别数量，= 4 表示3级大津阈值算法
classes = 5
# 这里用原图来提取，若先进行预处理，则同样经过预处理后的背景会影响算法效果
thresholds = multi_ostu.multi_ostu(image, classes)
print("thresholds: ", thresholds)
mask = image > thresholds[len(thresholds) - 1]
# 将人脸以外的区域涂成黑色
image[~mask] = 0

cv2.imshow("Result Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""呼吸波形图"""
# # 指定包含JSON文件的文件夹路径
# json_folder_path = '/home/yechentao/Programs/Test/newJson'
# # 指定包含图像文件的文件夹路径
# image_folder_path = '/home/yechentao/Programs/Test/grayimage'
# # 计算鼻孔处的像素变化值
# process_data.process(json_folder_path, image_folder_path)
