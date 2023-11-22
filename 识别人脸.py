import cv2

# 读取图片
img = cv2.imread(
    "C:\\py Fengxing0tian\\Ren Lian Shi Bie\\img00.jpg"
)

# 将图片从BGR色彩空间转换为RGB色彩空间
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 加载人脸检测器
faceCascade = cv2.CascadeClassifier(
    "C:\\Users\\L2876\\anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml"
)

# 检测人脸
faces = faceCascade.detectMultiScale(
    gray_img,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(32, 32)
)

# 在图像上绘制矩形框
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

# 显示结果
cv2.imshow('face', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
