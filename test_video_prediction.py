# USAGE
# python test_prediction.py --image examples

# import OpenCV before mxnet to avoid a segmentation fault
import cv2

# import the necessary packages
from config import age_gender_deploy as deploy
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.preprocessing import CropPreprocessor
from pyimagesearch.utils import AgeGenderHelper
from imutils.video import VideoStream
from imutils.face_utils import FaceAligner
from imutils import face_utils
from imutils import paths
import numpy as np
import mxnet as mx
import argparse
import pickle
import imutils
import json
import dlib
import os
import random 
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-r", "--resolution", default="1920x1080",
	help="resolution of the screen defautl 1920x1080 ")
ap.add_argument("-a", "--alpha", type=float, default=0.5,
	help="alpha transparency of the overlay (smaller is more transparent)")
args = vars(ap.parse_args())


def visAge(agePreds, le):
    # initialize the canvas and sort the predictions according
    # to their probability
    print(agePreds)

    idxs = np.argsort(agePreds)[::-1]
    print(idxs)
    # construct the text for the prediction
    #ageLabel = le.inverse_transform(j) # Python 2.7
    ageLabel = le.inverse_transform(idxs[0]).decode("utf-8")
    ageLabel = ageLabel.replace("_", "-")
    ageLabel = ageLabel.replace("-inf", "+")
    age = "{}".format(ageLabel)
    return age


def visGender(genderPreds, le):
    idxs = np.argsort(genderPreds)[::-1]
    gender = le.inverse_transform(idxs[0])
    gender = "Male" if gender == 0 else "Female"
    text_gender = "{}".format(gender)
    return text_gender

def load_image(path):
	imagePaths = list(paths.list_files(path,validExts=(".jpg")))
	print(imagePaths)
	random.shuffle(imagePaths)
	
	if imagePaths == []: print("need more gifs at :" + path)
	imagePath = imagePaths[0]
	return imagePath

def grab_frames_video(url):
	frames = []
	video =  cv2.VideoCapture(url)
	while True:
		grab , frame = video.read()
		if grab == True:
			frames.append(frame)
		else:
			return frames

def insert_animation(image,frame,position="center",alpha=args["alpha"]):
	watermark = frame

	(wH, wW) = watermark.shape[:2]
	(h, w) = image.shape[:2]
	overlay =  np.zeros((h, w, 3), dtype="uint8")
	print("watermark :"+str(watermark.shape[:2]))
	print("frame window  shape" + str(image.shape[:2]))
	if position =="right_low" :
		overlay[h - wH - 10:h - 10, w - wW - 10:w - 10] = watermark
	if position == "center" :
		init_y = int(h/2) - int(wH/2)
		if init_y < 0:
			init_y = 0 
		print("init_y :"+str(init_y))
	
		overlay[init_y :init_y+int(wH) , int(w/2) - int(wW/2) :int(w/2)+int(wW/2)] = watermark
	if position == "center_left" :
		overlay[int(h/2) - int(wH/2) :int(h/2)+int(wH/2) , int(w/2)-int(wW) :int(w/2)] = watermark

 
	# blend the two images together using transparent overlays
	output = image.copy()
	cv2.addWeighted(overlay, alpha, output, 1.0, 0, output)
 
	# write the output image to disk
	return output

def show_local_img(img,montage_copy):
	frame_montage = montage_copy.copy()
	framex = img
	frame_to_show = imutils.resize(framex, width=880, height=880)
	(h, w) = frame_to_show.shape[:2]
	h_montage = frame_montage.shape[0]
	if h > h_montage:
		print("crop image at :" + str(h_montage))
		frame_to_show=frame_to_show[0:h_montage,0:w]
	x_montage =  insert_animation(frame_montage,frame_to_show,alpha=1.0)
	cv2.imshow("Montage",x_montage)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		return



# load the label encoders and mean files
print("[INFO] loading label encoders and mean files...")
ageLE = pickle.loads(open(deploy.AGE_LABEL_ENCODER, "rb").read())
genderLE = pickle.loads(open(deploy.GENDER_LABEL_ENCODER, "rb").read())
ageMeans = json.loads(open(deploy.AGE_MEANS).read())
genderMeans = json.loads(open(deploy.GENDER_MEANS).read())

# load the models from disk
print("[INFO] loading models...")
agePath = os.path.sep.join([deploy.AGE_NETWORK_PATH,
	deploy.AGE_PREFIX])
genderPath = os.path.sep.join([deploy.GENDER_NETWORK_PATH,
	deploy.GENDER_PREFIX])
ageModel = mx.model.FeedForward.load(agePath, deploy.AGE_EPOCH)
genderModel = mx.model.FeedForward.load(genderPath,
	deploy.GENDER_EPOCH)

# now that the networks are loaded, we need to compile them
print("[INFO] compiling models...")
ageModel = mx.model.FeedForward(ctx=[mx.gpu(0)],
	symbol=ageModel.symbol, arg_params=ageModel.arg_params,
	aux_params=ageModel.aux_params)
genderModel = mx.model.FeedForward(ctx=[mx.gpu(0)],
	symbol=genderModel.symbol, arg_params=genderModel.arg_params,
	aux_params=genderModel.aux_params)

# initialize the image pre-processors
sp = SimplePreprocessor(width=256, height=256,
	inter=cv2.INTER_CUBIC)
cp = CropPreprocessor(width=227, height=227, horiz=True)
ageMP = MeanPreprocessor(ageMeans["R"], ageMeans["G"],
	ageMeans["B"])
genderMP = MeanPreprocessor(genderMeans["R"], genderMeans["G"],
	genderMeans["B"])
iap = ImageToArrayPreprocessor(dataFormat="channels_first")

# initialize dlib's face detector (HOG-based), then create the
# the facial landmark predictor and face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(deploy.DLIB_LANDMARK_PATH)
fa = FaceAligner(predictor)

 # if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	camera = VideoStream(src=0, usePiCamera=False).start()

# otherwise, load the video
else:
	# camera = VideoStream(src=args["video"], usePiCamera=False).start()	
	# camera = VideoStream(src="./blackorwhite.mp4", usePiCamera=False).start()	
	camera = cv2.VideoCapture("./blackorwhite.mp4")
# Draw main window 
# keep looping
cv2.namedWindow('Montage', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Montage', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
WIN_max_x = round(int(args["resolution"].split('x')[0])/1)
WIN_max_y = round(int(args["resolution"].split('x')[1])/1)
#montage = build_montages([], (256, 256), (WIN_max_x, WIN_max_y))[0]
montage = np.zeros((WIN_max_y, WIN_max_x, 3), dtype="uint8")
montage_copy = montage.copy()# loop over the image paths
image_to_show_path = "images/{}".format("unknow")
advertise = cv2.imread(load_image(image_to_show_path))
while True:

	# load the image from disk, resize it, and convert it to
	# grayscale
	# grab the current frame
	g,frame = camera.read()
	cv2.imshow("Frame",frame)
	cv2.imshow("Montage",montage)
	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
	print(g)
	print("[INFO] processing ")
	image = imutils.resize(frame, width=800)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)

	if len(rects) == 0 :
		show_local_img(advertise,montage_copy)
	# loop over the face detections
	else:
		if len(rects) > 0:
			rect = rects[0]
			# determine the facial landmarks for the face region, then
			# align the face
			shape = predictor(gray, rect)
			face = fa.align(image, gray, rect)

			# resize the face to a fixed size, then extract 10-crop
			# patches from it
			face = sp.preprocess(face)
			patches = cp.preprocess(face)

			# allocate memory for the age and gender patches
			agePatches = np.zeros((patches.shape[0], 3, 227, 227),
				dtype="float")
			genderPatches = np.zeros((patches.shape[0], 3, 227, 227),
				dtype="float")

			# loop over the patches
			for j in np.arange(0, patches.shape[0]):
				# perform mean subtraction on the patch
				agePatch = ageMP.preprocess(patches[j])
				genderPatch = genderMP.preprocess(patches[j])
				agePatch = iap.preprocess(agePatch)
				genderPatch = iap.preprocess(genderPatch)

				# update the respective patches lists
				agePatches[j] = agePatch
				genderPatches[j] = genderPatch

			# make predictions on age and gender based on the extracted
			# patches
			agePreds = ageModel.predict(agePatches)
			genderPreds = genderModel.predict(genderPatches)

			# compute the average for each class label based on the
			# predictions for the patches
			agePreds = agePreds.mean(axis=0)
			genderPreds = genderPreds.mean(axis=0)

			# visualize the age and gender predictions
			age = visAge(agePreds, ageLE)
			gender = visGender(genderPreds,genderLE)
			image_to_show_path = "images/{}/{}".format(gender,age)
			print(image_to_show_path)
			img = load_image(image_to_show_path)
			advertise_gender = cv2.imread(img)
			# draw the bounding box around the face
			clone = image.copy()
			(x, y, w, h) = face_utils.rect_to_bb(rect)
			cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

			# show the output image
			# cv2.imshow("Input", clone)
			# cv2.imshow("Face", face)
			show_local_img(advertise_gender,montage_copy)

			time.sleep(5)

			print(image_to_show_path)
	# cv2.imshow("Gender Probabilities", advertise)
	# cv2.imshow("Frame",frame)
	# 	# if the 'q' key is pressed, stop the loop
	# if cv2.waitKey(1) & 0xFF == ord("q"):
	# 	break