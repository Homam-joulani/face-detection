import cv2

face_cas = cv2.CascadeClassifier("venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

img = cv2.imread("Recorse/Elon Musk.jpg")

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cas.detectMultiScale(imgGray, 1.1, 4)

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h),(255,0,0),2)


cv2.imshow("result",img)
cv2.waitKey(0)