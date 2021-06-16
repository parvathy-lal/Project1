# Pose Estimation & Alert Generation for Sudden Fall

## Overview
This is an AI/ML model which estimates different body postures and detect the occuring of sudden fall. Then generates an alert. 

Here we are using the Single Pose Estimation which is the simpler and faster of the two algorithms. Its ideal use case is for when there is only one person in the image. Post Estimation can be done by **Human Pose Skeleton** represents which is a set of coordinates that can be connected to describe the pose of the person. 

## Installation 

- If you don't have it already, install Pycharm Community 2021.1.1  or later, following the instructions on the website

### Usage
- Open Pycharm Community, and from the **Welcome** screen, select New project.
- Name it as **Pose Estimation Project**
- Now install the required packages in Python Interpreter Files-> Settings-> Python Interpreter
- Install, **mediapipe, opencv-python, imutilis, win10toast**
- IN this project, we have 4 Python files.
1. PoseEst_Basics
2. PoseModule
3. Webcam_Pose
4. AlertOnFall

## Key Points
Our model takes a processed video as the input and outputs the information about keypoints. The keypoints detected are indexed by a part ID, with a confidence score between 0.0 and 1.0. The confidence score indicates the probability that a keypoint exists in that position.


The various body joints detected by our model are tabulated below:
|Id|Part|
|------|----------|
|0|Nose|
|1|leftEye|
|2|rightEye|
|3|leftEar|
|4|rightEar|
|5|leftShoulder|
|6|rightShoulder|
|7|leftElbow|
|8|rightElbow|
|9|leftWrist|
|10|rightWrist|
|11|leftHip|
|12|rightHip|
|13|leftKnee|
|14|rightKnee|
|15|leftAnkle|
|16|rightAnkle|

![](https://i1.wp.com/www.marktechpost.com/wp-content/uploads/2020/08/Screenshot-2020-08-25-at-10.01.08-PM.png?fit=1039%2C620&ssl=1)

### **PoseEst_Basics.py**
Here our model detect and mark the keypoints on the body parts. These marked points can be connected together to form the **Human Pose Skeleton**.

```
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
cap = cv2.VideoCapture('Pose Videos/video1.mp4')
```
Whenever the condition is True,
``` 
img = cap.read()
results = pose.process(imgRGB)
mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)

for id, lm in enumerate(results.pose_landmarks.landmark):
    h, w, c = img.shape
    print(id, lm)
    cx, cy = int(lm.x*w),int(lm.y*h)
    cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)

cv2.imshow("Image", img)
cv2.waitKey(1)
```

![](https://learnopencv.com/wp-content/uploads/2018/05/OpenPose.jpg)

### **Pose_Module.py**
In this module, we are trying to find out the pose and the position. For that, we have to create a class named '_poseDetector_'.
```
def __init__(self, mode = False, upBody = False, smooth = True,detectionCon = 0.5, trackCon = 0.5):
    self.mode = mode
    self.upBody = upBody
    self.smooth = smooth
    self.detectionCon = detectionCon
    self.trackCon = trackCon
    self.mpDraw = mp.solutions.drawing_utils
    self.mpPose = mp.solutions.pose
    self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)
```

Then we create two functions namely, '_findPose_' and '_findPosition_', in which _findPose_ returns  _img_ and _findPosition_ returns a list of landmarks.
```
[[0, 393, 151], [1, 398, 142], [2, 402, 141], [3, 406, 141], [4, 388, 143], [5, 385, 144], [6, 382, 144], [7, 414, 144], [8, 382, 148], [9, 402, 159], [10, 389, 161], [11, 444, 187], [12, 377, 209], [13, 452, 243], [14, 369, 276], [15, 415, 252], [16, 348, 334], [17, 405, 261], [18, 341, 355], [19, 392, 249], [20, 344, 356], [21, 393, 249], [22, 349, 347], [23, 460, 332], [24, 409, 338], [25, 497, 413], [26, 386, 420], [27, 531, 497], [28, 384, 497], [29, 529, 512], [30, 388, 509], [31, 545, 523], [32, 369, 525]]
...........................................................................................................................................................................................................................................................................................................................

[[0, 343, 174], [1, 340, 164], [2, 340, 164], [3, 341, 164], [4, 337, 163], [5, 334, 163], [6, 331, 162], [7, 331, 164], [8, 317, 164], [9, 339, 182], [10, 333, 182], [11, 331, 212], [12, 269, 215], [13, 323, 277], [14, 254, 289], [15, 339, 321], [16, 265, 360], [17, 339, 330], [18, 271, 379], [19, 341, 329], [20, 280, 374], [21, 342, 327], [22, 280, 368], [23, 333, 343], [24, 294, 353], [25, 355, 426], [26, 293, 432], [27, 320, 492], [28, 289, 500], [29, 307, 500], [30, 282, 510], [31, 345, 524], [32, 305, 529]]
```
![](https://cdn-images-1.medium.com/max/600/1*H2ViR54BACV0patPZmhHnw.gif)

### **WebCam_Pose.py**
In this module, we are using webcam for capturing the video to estimate the poses 

```
cap = cv2.VideoCapture(0)
cv2.imshow("Webcam", img)
```
The output will be:

[[0, 348, 257], [1, 371, 224], [2, 386, 224], [3, 399, 223], [4, 327, 224], [5, 313, 224], [6, 298, 224], [7, 411, 233], [8, 279, 239], [9, 376, 291], [10, 321, 292], [11, 440, 358], [12, 195, 374], [13, 472, 515], [14, 144, 542], [15, 381, 365], [16, 130, 642], [17, 364, 333], [18, 112, 681], [19, 367, 317], [20, 136, 668], [21, 366, 331], [22, 151, 654], [23, 386, 667], [24, 228, 663], [25, 366, 889], [26, 225, 893], [27, 349, 1092], [28, 219, 1090], [29, 346, 1126], [30, 212, 1124], [31, 335, 1162], [32, 240, 1163]]

................................................................................................................................................................................................................................................................................................................................
[[0, 264, 214], [1, 285, 183], [2, 299, 183], [3, 313, 183], [4, 248, 185], [5, 236, 186], [6, 223, 187], [7, 335, 205], [8, 216, 209], [9, 293, 253], [10, 241, 254], [11, 421, 371], [12, 137, 380], [13, 476, 563], [14, 72, 565], [15, 447, 693], [16, 88, 694], [17, 449, 742], [18, 84, 744], [19, 433, 724], [20, 102, 724], [21, 425, 709], [22, 111, 707], [23, 377, 748], [24, 180, 748], [25, 354, 1031], [26, 186, 1029], [27, 344, 1288], [28, 190, 1288], [29, 344, 1326], [30, 189, 1326], [31, 322, 1374], [32, 204, 1376]]

![](https://www.tensorflow.org/images/lite/models/pose_estimation.gif)


### **Alert_On_Fall.py**
Here comes our exact target. We tried to send an alert on the occuring of sudden fall. For fall detection, we find out the boundaries and marked it as a green rectangle. When the boundary values get changed, it will detect the falling and boundaries will be started showing as a red rectangle.

```
x, y, w, h = cv2.boundingRect(cnt)
cv2.drawContours(fgmask, [cnt], 0, (255, 255, 255), 3, maxLevel=0)
if h < w:
    j += 1
if j > 10:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
```

When an anomaly detected, an alert will be generated. For alert, we have to install **_win10toast_** package and alert formed as,

``` 
hr = ToastNotifier()
hr.show_toast(title='Alert',msg='Sudden Fall')
```

Output will be,
[[0, 460, 43], [1, 464, 41], [2, 465, 41], [3, 466, 41], [4, 464, 41], [5, 464, 40], [6, 464, 40], [7, 471, 43], [8, 468, 41], [9, 464, 49], [10, 464, 48], [11, 479, 62], [12, 478, 61], [13, 469, 91], [14, 477, 90], [15, 431, 111], [16, 455, 106], [17, 424, 116], [18, 455, 112], [19, 421, 114], [20, 443, 108], [21, 424, 113], [22, 456, 111], [23, 514, 104], [24, 513, 106], [25, 484, 119], [26, 482, 132], [27, 498, 162], [28, 498, 159], [29, 507, 166], [30, 504, 162], [31, 483, 173], [32, 487, 167]]

.....................................................................................................................................................................................................................................................................
[]
[]
[]
[]
[]
[]
[]
[]
[]
[]
[]
[]
[]
[]
[]
[]
[]
[]
[]
[]
[]
[]
[]
[]
[]
