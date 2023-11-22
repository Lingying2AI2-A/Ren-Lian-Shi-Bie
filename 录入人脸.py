import cv2
import os

# 调用内置摄像头
cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(
    'C:\\Users\\L2876\\anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
)

face_id = input('\n enter user id:')
print('\n  看着镜头等待......')

count = 0

while True:
    sucess, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray_img, 1.3, 5
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0))
        count =count+1

        # 保存图片
        cv2.imwrite(
            "C:\\py Fengxing0tian\\Ren Lian Shi Bie\\Facedata\\" + str(face_id) + '.' + str(count) + '.jpg', gray_img[y: y + h, x: x + w]
        )

        cv2.imshow('face', img)

    k = cv2.waitKey(1)
    if k == 27:
        break
    elif count >= 100:
        break

# 关闭摄像头
cap.release()
cv2.destroyAllWindows()