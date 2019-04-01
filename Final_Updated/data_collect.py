# USAGE
# python skindetector.py
# python skindetector.py --video video/skin_example.mov

# import the necessary packages
from pyimagesearch import imutils
import numpy as np
import argparse
import cv2
import time
from keras.preprocessing import image

img_path_fist = "fist/fist"
img_path_hand = "hand/hand"
img_path_one = "one/one"



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help = "path to the (optional) video file")
args = vars(ap.parse_args())

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'

#0,48,80
#20,255,255

lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

# if a video path was not supplied, grab the reference
# to the gray
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping over the frames in the video
i = 1
time.sleep(5)
while i < 1500:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a
	# frame, then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame, convert it to the HSV color space,
	# and determine the HSV pixel intensities that fall into
	# the speicifed upper and lower boundaries
	frame = imutils.resize(frame, width = 400)
	converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	skinMask = cv2.inRange(converted, lower, upper)

	# apply a series of erosions and dilations to the mask
	# using an elliptical kernel
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
	skinMask = cv2.erode(skinMask, kernel, iterations = 2)

	skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
	

	# blur the mask to help remove noise, then apply the
	# mask to the frame

	
	skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
	skin = cv2.bitwise_and(frame, frame, mask = skinMask)
	im = skin
	imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray,127,255,0)
	im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	if (len(contours) > 0):
        	#print(contours[0])
        	c = max(contours, key = cv2.contourArea)
        	x,y,w,h = cv2.boundingRect(c)
        	print(x,y,w,h)
        	cv2.rectangle(skin,(x,y),(x+w,y+h),(0,255,0),2)
        	cv2.drawContours(skin, contours, 0, (0,255,0), 3)
        	roi = skin[y:y+h, x:x+w]
        	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        	th1 = cv2.resize(gray, (150,150))

	# show the skin in the image along with the mask
	#cv2.imshow("images", np.hstack([frame, skin]))
                #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                #th1 = cv2.resize(gray, (150,150))
        #gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
        #th1 = cv2.resize(gray, (150, 150))

	cv2.imshow("images", skin)
	th1 = image.img_to_array(th1)
	npa = np.array([th1])
        
	#cv2.imwrite('None/none' + str(i) + '.png',th1)

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
	i += 1
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
