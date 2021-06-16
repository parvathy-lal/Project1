import cv2
import time
import imutils
import PoseModule as pm
from win10toast import ToastNotifier

fitToEllipse = False
cap = cv2.VideoCapture('Pose Videos/video7.mp4')
time.sleep(2)
fgbg = cv2.createBackgroundSubtractorMOG2()
j = 0

pTime = 0
detector = pm.poseDetector()

while (1):
    ret, img = cap.read()
    img = imutils.resize(img, 500)
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    print(lmList)
    # Convert each frame to gray scale and subtract the background
    try:
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(img)

        # Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:

            # List to hold all areas
            areas = []

            for contour in contours:
                ar = cv2.contourArea(contour)
                areas.append(ar)

            max_area = max(areas, default=0)

            max_area_index = areas.index(max_area)

            cnt = contours[max_area_index]

            M = cv2.moments(cnt)

            x, y, w, h = cv2.boundingRect(cnt)

            cv2.drawContours(fgmask, [cnt], 0, (255, 255, 255), 3, maxLevel=0)

            if h < w:
                j += 1

            if j > 10:
                #print("FALL")
                cv2.putText(fgmask, 'FALL', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                hr = ToastNotifier()
                hr.show_toast(title='Alert',msg='Sudden Fall')

            if h > w:
                j = 0
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.imshow('video', img)

            if cv2.waitKey(33) == 27:
                break
    except Exception as e:
        break
#cv2.destroyAllWindows()