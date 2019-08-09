from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import os
from knn import predict
import numpy as np
import pickle


detection_graph, sess = detector_utils.load_inference_graph()

# body mpi 
protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

nPoints = 15

with open('pca.pkl', 'rb') as file:
    myPCA = pickle.load(file)


def getImages():
    images = []

    for _, _, i in os.walk('testdata'):
        for img in i:
            images.append('testdata/'+str(img))
    return images


def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, 
                      im_height, image_np):
    # print(num_hands_detect)
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width,
                                          boxes[i][3] * im_width,
                                          boxes[i][0] * im_height,
                                          boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 2, 2)

            test = image_np[int(top): int(bottom), int(left):int(right)]
            test = cv2.resize(test, (128, 128))
            test = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
            label = predict(test)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if i == 0:
                cv2.putText(image_np, str(label), (100, 100), font, 1,
                            (255, 0, 0), 2, cv2.LINE_AA)
            if i == 1:
                cv2.putText(image_np, str(label), (150, 100), font, 1,
                            (255, 0, 0), 2, cv2.LINE_AA)


def annotation(video_source=0, width=320, height=180,
               score_thresh=0.2, display=1, fps=1):

    # cap = cv2.VideoCapture(video_source)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    start_time = datetime.datetime.now()
    num_frames = 0
    # im_width, im_height = (cap.get(3), cap.get(4))

    im_width, im_height = (1920, 1080)

    # max number of hands we want to detect/track
    num_hands_detect = 2

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    images = getImages()

    for image in images:
        # Expand dimensions since the model expects images to
        # have shape: [1, None, None, 3]
        # ret, image_np = cap.read()
        image_np = cv2.imread(image)

        # image_np = cv2.flip(image_np, 1)
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")
        boxes, scores = detector_utils.detect_objects(image_np,
                                                      detection_graph, sess)
        # draw bounding boxes on frame
        draw_box_on_image(num_hands_detect, score_thresh,
                          scores, boxes, im_width, im_height,
                          image_np)
        if (display > 0):
            cv2.imshow('Single-Threaded Detection',
                       cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


annotation()

# python3 detect_single_threaded.py -src "./Khmer-Chol-Chnam-Thmay.webm"
#  -fps 1 -sth 0.1
