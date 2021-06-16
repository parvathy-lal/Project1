import cv2
import time
import imutils
import PoseModule as pm

cap = cv2.VideoCapture(0)
pTime = 0
detector = pm.poseDetector()
while True:
    success, img = cap.read()
    img = imutils.resize(img, 500)
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    print(lmList)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
