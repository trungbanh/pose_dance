import cv2
import numpy as np
import os

LABEL = {
    "ca_2_tay_chap_vao_hong_phai": 1,
    "chap_tay": 2,
    "ngam_hoa": 3,
    "phai_thang_trai_dang_hoa": 4,
    "tay_phai_hong_tay_trai_hoa": 5,
    "xoay_nguoi": 6
}


def cut_frame(path):
    img = cv2.imread(path, 0)
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    newimg = np.array(img)
    mewimg = newimg[y-10:y+h+10, x-10:x+w+10]
    cv2.imwrite(path, mewimg)


def get_data():
    labels = []
    data = []
    for _, d, _ in os.walk('./bodyposture/'):
        for path in d:
            for r, _, f in os.walk('./bodyposture/'+path):
                for img in f:
                    # print(r+'/'+img)
                    labels.append(LABEL[r.split('/')[2]])
                    img = cv2.imread(r+'/'+img, 0)
                    img = cv2.resize(img, (250, 700))
                    img = np.array(img)
                    data.append(img.flatten())
    return np.array(data, np.float32), np.array(labels)


data, labels = get_data()

labels = np.reshape(labels, (1, -1))

print(data.shape)

# print(labels.shape)
knn = cv2.ml.KNearest_create()
knn.train(data, cv2.ml.ROW_SAMPLE, labels)


testchoi = cv2.imread('testchoi.png', 0)

testchoi = np.array(testchoi, np.float32).flatten()
ret, results, neighbours, dist = knn.findNearest(np.array([testchoi]), 3)


print("result:  {}\n".format(results))
print("neighbours:  {}\n".format(neighbours))
print("distance:  {}\n".format(dist))
