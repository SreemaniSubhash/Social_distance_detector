# import packages
from Detect import social_distancing_config as config
from Detect.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

# constructing the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# loading YOLO trained on COCO dataset
print("loading YOLO...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()

# Check the output of getUnconnectedOutLayers()
unconnected_layers = net.getUnconnectedOutLayers()
print("Unconnected layers:", unconnected_layers)

# Check if unconnected_layers is a scalar or a list
if isinstance(unconnected_layers, int):
    # If it's a scalar, convert it to a list
    unconnected_layers = [unconnected_layers]

# Use a loop to get the correct indices
ln = [ln[i - 1] for i in unconnected_layers]

# initialize the video stream and pointer to output video file
print("accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

# loop over the frames
while True:
	# read frame
	(grabbed, frame) = vs.read()
	if not grabbed:
		break

	# resize the frame and then detect
	frame = imutils.resize(frame, width=700)
	results = detect_people(frame, net, ln,
		personIdx=LABELS.index("person"))

	violate = set()

	# ensuring atleast two people are detected from a frame
	if len(results) >= 2:
		# extract all centroids and compute the Euclidean distances between all pairs of the centroids
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")

		# loop over the upper triangular of the distance matrix
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# check to see if the distance between any two centroid pairs is less than the configured number of pixels
				if D[i, j] < config.MIN_DISTANCE:
					# update our violation set with the indexes of
					# the centroid pairs
					violate.add(i)
					violate.add(j)

	# loop over the results
	for (i, (prob, bbox, centroid)) in enumerate(results):
		# extract the bounding box and centroid coordinates and initialize the color of biunding boxes
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)

		# update to red color for index pair exists within the violation set
		if i in violate:
			color = (0, 0, 255)

		# drawing a bounding box around the person and the centroid coordinates of the person,
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 1)

	# displaying total no. of violations on screen
	text = "Social Distancing Violations: {}".format(len(violate))
	cv2.putText(frame, text, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

	# display
	if args["display"] > 0:
		# showing output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# 'q' to quit
		if key == ord("q"):
			break

	# if an output video file path has been supplied
	if args["output"] != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

	if writer is not None:
		writer.write(frame)
