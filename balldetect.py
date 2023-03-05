import cv2
import numpy as np
import queue, threading, time

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()
# capture frames from a camera
ip_front = 'http://192.168.227.245:4747/mjpegfeed?640x480'
cap = VideoCapture(ip_front)
img = None
count = 0
rate = 10

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

highRed = 166
lowRed = 99
highBlue = 73
lowBlue = 0
lowGreen = 17
highGreen = 88
while True:
    img = cap.read()
    time.sleep(.5)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # if count % rate == 0:

    #     img = new_img
    #     img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    #     lower = np.array([lowRed,lowGreen,lowBlue])
    #     upper = np.array([highRed,highGreen,highBlue])
        
        
    #     mask = cv2.inRange(img, lower, upper)
    #     mask = cv2.erode(mask, None, iterations=2)
    #     mask = cv2.dilate(mask, None, iterations=2)
    #     img = cv2.bitwise_and(img,img, mask= mask)
    #     contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     if len(contours) == 0:
    #         continue
    #     blob = max(contours, key=lambda el: cv2.contourArea(el))
    #     M = cv2.moments(blob)
    #     center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    #     cv2.circle(img, center, 2, (255,255,0), -1)
        # contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=150, param2=0.8)
        # # ensure at least some circles were found
        # if circles is not None:
        #     # convert the (x, y) coordinates and radius of the circles to integers
        #     circles = np.round(circles[0, :]).astype("int")
        #     # loop over the (x, y) coordinates and radius of the circles
        #     for (x, y, r) in circles:
        #         # draw the circle in the output image, then draw a rectangle
        #         # corresponding to the center of the circle
        #         cv2.circle(img, (x, y), r, (0, 255, 0), 4)
        #         cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    #     # blurred = cv2.GaussianBlur(img, (11, 11), 0)
    #     # hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    #     # mask = cv2.inRange(hsv, greenLower, greenUpper)
    #     # mask = cv2.erode(mask, None, iterations=2)
    #     # mask = cv2.dilate(mask, None, iterations=2)
    #     bodies,rejectLevels, levelWeights = face_cascade.detectMultiScale3(
    #         img,
    #         scaleFactor=1.1,
    #         minNeighbors=20,
    #         minSize=(24, 24),
    #         # maxSize=(96,96),
    #         flags = cv2.CASCADE_SCALE_IMAGE,
    #         outputRejectLevels = True
    #     )
    #     print(rejectLevels)
    #     print(levelWeights)

    #     i = 0
    #     font = cv2.FONT_ITALIC
    #     for (x,y,w,h) in bodies:
    #         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         #cv2.putText(image,str(i)+str(":")+str(np.log(levelWeights[i][0])),(x,y), font,0.5,(255,255,255),2,cv2.LINE_AA)
    #         cv2.putText(img,str(levelWeights[i]),(x,y), font,0.5,(255,255,255),2,cv2.LINE_AA)
    #         i = i+1

    #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=150, param2=0.8)
    #     # # ensure at least some circles were found
    #     # if circles is not None:
    #     #     # convert the (x, y) coordinates and radius of the circles to integers
    #     #     circles = np.round(circles[0, :]).astype("int")
    #     #     # loop over the (x, y) coordinates and radius of the circles
    #     #     for (x, y, r) in circles:
    #     #         # draw the circle in the output image, then draw a rectangle
    #     #         # corresponding to the center of the circle
    #     #         cv2.circle(img, (x, y), r, (0, 255, 0), 4)
    #     #         cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    cv2.imshow('img',img)

    cv2.waitKey(1)

    count += 1