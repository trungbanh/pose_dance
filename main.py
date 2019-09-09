from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import os
import json
from knn import predict
import numpy as np
import pickle
from pose2d import get_pose
from classify import peceptron


detection_graph, sess = detector_utils.load_inference_graph()

# body mpi 
protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

nPoints = 15

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

with open('lda.pkl', 'rb') as file:
    lda = pickle.load(file)

def getImages():
    images = []

    for _, _, i in os.walk('testdata'):
        for img in i:
            images.append('testdata/'+str(img))
    return images


def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, 
                      im_height, image_np):
    # print(num_hands_detect)

    hands = ['None', 'None']
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
            label = predict(test,lda)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            if i == 0:
                # cv2.putText(image_np, str(label), (100, 100), font, 1,
                #             (255, 0, 0), 2, cv2.LINE_AA)
                hands[0] = str(label[0])
            if i == 1:
                # cv2.putText(image_np, str(label), (150, 100), font, 1,
                #             (255, 0, 0), 2, cv2.LINE_AA)
                hands[1] = str(label[0])
    return hands


def annotation(video_source=0, width=320, height=180,
               score_thresh=0.2, display=1, fps=1):

    jsonfile = open('jsontest.json', 'a+')

    im_width, im_height = (1920, 1080)
    # max number of hands we want to detect/track
    num_hands_detect = 2
    # cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    model = peceptron()

    # images = getImages()
    # for image in images:
    video = cv2.VideoCapture("Khmer-Chol-Chnam-Thmay.webm")
    num = 0
    while (True):
        # Expand dimensions since the model expects images to
        # image_np = cv2.imread(image)
        ret, image_np = video.read()
        
        num = num+1
        if num >= 128 and num % 3 == 0:
            print(num)

            pose, foots = get_pose(image_np, net)

            # ret, results, neighbours, dist = knn.findNearest(np.array([pose], np.float32), 3)
            result = model.predict([pose])

            # cv2.putText(image_np, str(results),(100,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            try:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")
            boxes, scores = detector_utils.detect_objects(image_np,
                                                        detection_graph, sess)
            # draw bounding boxes on frame
            hands = draw_box_on_image(num_hands_detect, score_thresh,
                                    scores, boxes, im_width, im_height,
                                    image_np)
            # if (display > 0):
            #     cv2.imshow('Single-Threaded Detection',
            #                 cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            #     if cv2.waitKey(0) & 0xFF == ord('q'):
            #         cv2.destroyAllWindows()
            #         break
            frame = {
                "number": num,
                "foots": foots,
                "pose": int(result[0]),
                "hands": hands
            }
            json.dump(frame,jsonfile)
            jsonfile.write(",\n")
            
    video.release()
    print ("okey chưa")

annotation()


# python3 detect_single_threaded.py -src "./Khmer-Chol-Chnam-Thmay.webm"
#  -fps 1 -sth 0.1
