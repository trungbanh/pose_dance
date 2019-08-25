import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import pickle
CLASSIFY = {
    'fruit': 1,
    'lead': 2,
    'pray': 3
}


def read(path='./knowledge/') -> np:
    """
        **read all image**
        :param path: is a path (defaut is '.knowledge')
        :type: file: str

        :return list of image and class
        :rtype numpy, list
    """

    images = []
    labels = []

    for _, d, _ in os.walk('./knowledge/'):
        for path in d:
            for r, _, f in os.walk('./knowledge/'+path):
                for img in f:
                    img = cv2.imread(r+'/'+img, 0)
                    img = cv2.resize(img, (128, 128))
                    img = np.array(img).flatten()
                    images.append(img)
                    labels.append(CLASSIFY[r.split('/')[2]])

    return np.array(images), labels


def predict(image,lda):
    data = image.reshape(1, -1)
    testResponse = lda.predict(data)
    return testResponse

