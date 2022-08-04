import numpy as np
import tensorflow as tf
import cv2
from scipy.spatial import distance as dist
import dlib

TRAINED_MODEL_DIR = ""
PATH_TO_CKPT = r'mask.pb'
PATH_TO_CKPT2 =   r'frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'labelmap.pbtxt'


def load_inference_graph(PATH_TO_CKPT):

	print('=======> Loading frozen graph into memory')
	detection_graph = tf.compat.v1.Graph()

	with detection_graph.as_default():
		od_graph_def = tf.compat.v1.GraphDef()
		with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')
			sess = tf.compat.v1.Session(graph=detection_graph)
			print(detection_graph)
			print(sess)
		print('=======> Detection graph loaded')
		return detection_graph, sess


def detect_objects(image_np, detection_graph, sess):

	image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
	detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
	detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
	detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
	num_detections = detection_graph.get_tensor_by_name('num_detections:0')
	image_np_expanded = np.expand_dims(image_np, axis=0)

	(boxes, scores, classes, num) = sess.run(
  		[detection_boxes, detection_scores,
    	 detection_classes, num_detections],
    	feed_dict={image_tensor: image_np_expanded})

	return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)


def draw_box_on_face(frame, score_thresh, scores, boxes, classes, im_width, im_height, image_np):
	num_face_detect = 0
	color = None
	color0 = (0,255,0)
	color1 = (255,0,0)
	color2 = (255,255,0)
	print(sum(boxes))
	face_detector = dlib.get_frontal_face_detector()
	frame1 = cv2.flip(frame, 1)
	gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
	faces = face_detector(gray)
	num_face_detect = len(faces)
	if num_face_detect ==0:
		cv2.putText(image_np, "NO PEOPLE ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	for i in range(num_face_detect):
		if scores[i] > score_thresh:
			item = ''
			if classes[i]==1:
				item = 'Mask'
				color = color0
			elif classes[i]==2:
				item = 'No Mask'
				color = color1

			(x_min, x_max, y_min, y_max) = (boxes[i][1]*im_width, boxes[i][3]*im_width,
											boxes[i][0]*im_height, boxes[i][2]*im_height)

			p1 = (int(x_min), int(y_min))
			p2 = (int(x_max), int(y_max))

			cv2.rectangle(image_np, p1, p2, color, 3, 1)

			cv2.putText(image_np, 'Face '+str(i)+': '+item, (int(x_min), int(y_min)-5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	return num_face_detect

def alert_check(image_np, im_width, im_height, p1, p2, point_dict):
	if len(point_dict.items()) > 1:

		if point_dict[0][1][0] < point_dict[1][0][0]:
			c1 = (point_dict[0][1][0], (point_dict[0][0][1]+point_dict[0][1][1])//2)
			c2 = (point_dict[1][0][0], (point_dict[1][0][1]+point_dict[1][1][1])//2)

			cv2.line(image_np, c1, c2, (0,0,255), 2, 8)
			distance = dist.euclidean(c1, c2)
			dist_inch = distance/100
			pt = (((c1[0]+c2[0])//2)-10, ((c1[1]+c2[1])//2)-10)
			cv2.putText(image_np, '%0.2f ft'%(dist_inch), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
			if dist_inch<1.35:
				cv2.putText(image_np, "MAINTIAN SOCIAL DISTANCE", (10, 30),
							cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2)

		elif point_dict[1][1][0] < point_dict[0][0][0]:
			c1 = (point_dict[1][1][0], (point_dict[1][0][1]+point_dict[1][1][1])//2)
			c2 = (point_dict[0][0][0], (point_dict[0][0][1]+point_dict[0][1][1])//2)

			cv2.line(image_np, c1, c2, (0,0,255), 2, 8)
			distance = dist.euclidean(c1, c2)
			dist_inch = distance/101.76
			pt = (((c1[0]+c2[0])//2)-10, ((c1[1]+c2[1])//2)-10)
			cv2.putText(image_np, '%0.2f ft'%(dist_inch), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
			if dist_inch < 1.35:
				cv2.putText(image_np, "MAINTAIN SOCAIL DISTANCE ", (10, 30),

							cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2)




def draw_box_on_person(num_persons, score_thresh, scores, boxes, classes, im_width, im_height, image_np):
	
	color = None
	color0 = (255,255,255)
	point_dict = {}
	for i in range(num_persons):

		if scores[i] > score_thresh:
			item = ''
			if classes[i]==1:
				item = 'ID '
				color = color0
			
			(x_min, x_max, y_min, y_max) = (boxes[i][1]*im_width, boxes[i][3]*im_width,
											boxes[i][0]*im_height, boxes[i][2]*im_height)
			p1 = (int(x_min), int(y_min))
			p2 = (int(x_max), int(y_max))

			cv2.rectangle(image_np, p1, p2, color, 3, 1)

			cv2.putText(image_np, item+str(i), (int(x_min), int(y_min)-5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			point_dict[i] = (p1, p2)
			alert_check(image_np, im_width, im_height, p1, p2, point_dict)