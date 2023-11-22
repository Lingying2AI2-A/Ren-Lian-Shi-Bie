import cv2  # 导入opencv库
import numpy as np  #导入numpy库并将其命名为np

# 输入训练图片
# 将训练图片的路径放入列表images中
images=[]
images.append(cv2.imread('C:\\py Fengxing0tian\\Ren Lian Shi Bie\\img01.jpg',cv2.IMREAD_GRAYSCALE))  # 将图片路径和读取图片的方式作为参数，将读取的图片转换为灰度图并存入images列表的第一个元素
images.append(cv2.imread('C:\\py Fengxing0tian\\Ren Lian Shi Bie\\img02.jpg',cv2.IMREAD_GRAYSCALE))  # 将图片路径和读取图片的方式作为参数，将读取的图片转换为灰度图并存入images列表的第二个元素

# 新增两张训练图片
images.append(cv2.imread('C:\\py Fengxing0tian\\Ren Lian Shi Bie\\img03.jpg',cv2.IMREAD_GRAYSCALE))  # 将图片路径和读取图片的方式作为参数，将读取的图片转换为灰度图并存入images列表的第三个元素
images.append(cv2.imread('C:\\py Fengxing0tian\\Ren Lian Shi Bie\\img04.jpg',cv2.IMREAD_GRAYSCALE))  # 将图片路径和读取图片的方式作为参数，将读取的图片转换为灰度图并存入images列表的第四个元素

# 定义训练图片的标签，每个图片对应一个标签，标签列表存储了所有图片的标签
labels=[0,0,1,1]  # 这里将标签设置为0,0

# 创建LBPHFaceRecognizer对象，用于训练人脸识别模型
recognizer=cv2.face.LBPHFaceRecognizer_create()

# 对训练图片和标签进行训练，得到人脸识别模型
recognizer.train(images,np.array(labels))
