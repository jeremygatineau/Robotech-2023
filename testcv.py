# OpenCV program to detect face in real time
# import libraries of python OpenCV
# where its functionality resides
import cv2
from enum import Enum
import time

# load the required trained XML classifiers
# https://github.com/Itseez/opencv/blob/master/
# data/haarcascades/haarcascade_frontalface_default.xml
# Trained XML classifiers describes some features of some
# object we want to detect a cascade function is trained
# from a lot of positive(faces) and negative(non-faces)
# images.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class Mode(Enum):
	FIND_PERSON = 0
	NOTHING = 1
	MOVE_TO_PERSON = 2
	LOOK_FOR_BALL = 3
	MOVE_TO_BALL = 4
	SIDE_TO_SIDE = 5



# (left_motor, right_motor, drop_ball, open_lid)
def signal(sig):
	return

# https://github.com/Itseez/opencv/blob/master
# /data/haarcascades/haarcascade_eye.xml
# Trained XML file for detecting eyes
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# capture frames from a camera
ip_front = 'http://192.168.30.4:4747/mjpegfeed?640x480'
ip_back = 'http://192.168.30.213:4747/mjpegfeed?640x480'
ip_side = 'http://192.168.30.188:4747/mjpegfeed?640x480'
# cap1 = cv2.VideoCapture()
# cap2 = cv2.VideoCapture()
# cap3 = cv2.VideoCapture()
rate = 10

img = None
count = 0
close_area_threshold = 400
center_thresh = 20
curr_mode = Mode.FIND_PERSON

while True:
	match curr_mode:
		case Mode.NOTHING:
			continue
		case Mode.FIND_PERSON:
			cap = cv2.VideoCapture(ip_front)
			signal((1, -1, 0, 0))
			while True:
				ret, new_img = cap.read()
				if count % rate == 0:
					img = new_img
					img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

					# convert to gray scale of each frames
					gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

					# Detects faces of different sizes in the input image
					faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30), flags =cv2.CASCADE_SCALE_IMAGE)
					if len(faces) > 0:
						signal((0, 0, 0, 0))
					max_x, max_y, max_w, max_h = (0, 0, 0, 0)
					for (x,y,w,h) in faces:
						if w * h > max_w * max_h:
							max_x, max_y, max_w, max_h = (x, y, w, h)

					cv2.rectangle(img,(max_x,max_y),(max_x+max_w,max_y+max_h),(255,255,0),2)

					if x + w/2 > 240 + center_thresh:
						signal((1, -1, 0, 0))
					elif x + w/2 < 240 - center_thresh:
						signal((-1, 1, 0, 0))
					else:
						curr_mode = Mode.MOVE_TO_PERSON
						cap.release()
						break
				count += 1
			continue
		case Mode.LOOK_FOR_BALL:
			cap = cv2.VideoCapture(ip_back)
			signal((1, -1, 0, 0))
			while True:
				ret, new_img = cap.read()
				if count % rate == 0:
					img = new_img
					img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

					# Detects faces of different sizes in the input image
					blurred = cv2.GaussianBlur(img, (11, 11), 0)
					hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

					faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30), flags =cv2.CASCADE_SCALE_IMAGE)
					if len(faces) > 0:
						signal((0, 0, 0, 0))
					max_x, max_y, max_w, max_h = (0, 0, 0, 0)
					for (x,y,w,h) in faces:
						if w * h > max_w * max_h:
							max_x, max_y, max_w, max_h = (x, y, w, h)

					cv2.rectangle(img,(max_x,max_y),(max_x+max_w,max_y+max_h),(255,255,0),2)

					if x + w/2 > 240 + center_thresh:
						signal((1, -1, 0, 0))
					elif x + w/2 < 240 - center_thresh:
						signal((-1, 1, 0, 0))
					else:
						curr_mode = Mode.MOVE_TO_BALL
						cap.release()
						break
				count += 1
			continue
		case Mode.MOVE_TO_PERSON:
			signal((1, 1, 0, 0))
			time.sleep(5) #modify this to go forward
			signal((1, -1, 0, 0)) # spin 180 degrees
			time.sleep(0.5)
			signal((0, 0, 0, -1)) # drop ball
			time.sleep(0.5)
			signal((1, 1, 0, 0)) # go for a bit
			time.sleep(0.5)
			signal((1, -1, 0, 0)) # turn around 180
			time.sleep(1)
			signal((0, 0, 1, 0))
			time.sleep(10)
			signal((0, 0, 0, 0))
			curr_mode = Mode.LOOK_FOR_BALL
			continue
		case Mode.MOVE_TO_BALL:
			signal((1, 1, 0, 1))
			time.sleep(5) #modify this to go forward until we hit ball
			signal((0, 0, 0, 0))
			time.sleep(0.5)
			signal((1, -1, 0, 0)) #turn around 180
			time.sleep(1) 
			signal((1, 1, 0, 0))
			time.sleep(5) # go back the same amount of time 
			signal((0, 0, 0, 0))
			curr_mode = Mode.FIND_PERSON
			continue
		case Mode.SIDE_TO_SIDE:
			continue

# while cap.isOpened():
# 	match 
# 	

# 	if ret:
# 		# cv2.imwrite('frame{:d}.jpg'.format(count), frame)
		
# 		if len(actions) > 0:
# 			if actions[0][1] == 0:
# 				act = actions.pop(0)
# 				if act[0][0][3] == 1:
# 					waitForBallIn()
# 			else:
# 				signal(actions[0][0])
# 				actions[0][1] -= 1

# 		if count % rate == 0:
# 			img = new_img
# 			img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# 			# convert to gray scale of each frames
# 			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 			# Detects faces of different sizes in the input image
# 			faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30), flags =cv2.CASCADE_SCALE_IMAGE)

# 			max_x, max_y, max_w, max_h = (0, 0, 0, 0)
# 			for (x,y,w,h) in faces:
# 				if w * h > max_w * max_h:
# 					max_x, max_y, max_w, max_h = (x, y, w, h)
# 				# print(x, y, w, h)
# 				# To draw a rectangle in a face
				
# 				# roi_gray = gray[y:y+h, x:x+w]
# 				# roi_color = img[y:y+h, x:x+w]

# 				# # Detects eyes of different sizes in the input image
# 				# eyes = eye_cascade.detectMultiScale(roi_gray)

# 				# #To draw a rectangle in eyes
# 				# for (ex,ey,ew,eh) in eyes:
# 				# 	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)

# 			cv2.rectangle(img,(max_x,max_y),(max_x+max_w,max_y+max_h),(255,255,0),2)

# 			if x + w > 240:
# 				appendAction((1, 0.5, 0, 0), 100)
# 				appendAction((1, 1, 0, 0), 1)

# 			if w * h > close_area_threshold:
# 				actions = []
# 				appendAction((0, 0, 1, 0), 10) # Drop ball
# 				appendAction((-1, 1, 0, 0), 50) # tinker with this to turn around
# 				appendAction((1, 1, 0, 0), 1000) # walking forward
# 				appendAction((-1, 1, 0, 0), 50) # tinker with this to turn around
# 				appendAction((0, 0, 0, 1), 1) # Open lid
				




# 		# Display an image in a window
# 		cv2.imshow('img',img)
# 		count += 1 # i.e. at 30 fps, this advances one second
# 		# cap.set(cv2.CAP_PROP_POS_FRAMES, count)
# 		cv2.waitKey(1)
# 	else:
# 		cap.release()
# 		break

# # De-allocate any associated memory usage
# cv2.destroyAllWindows()