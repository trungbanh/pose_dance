import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import pickle
CLASSIFY = {
    'fruit': 1,
    'lead': 2,
    'pray': 3,
    'rice': 4
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


def PCA(images):
    with open('pca.pkl', 'rb') as file:
        myPCA = pickle.load(file)
    data = myPCA.transform(images)

    return data


# def SVM_Classifycaition(X, y):
#     svm = cv2.ml.SVM_create()
#     # Set SVM type
#     svm.setType(cv2.ml.SVM_C_SVC)
#     # Set SVM Kernel to Radial Basis Function (RBF)
#     svm.setKernel(cv2.ml.SVM_RBF)
#     # Set parameter C
#     svm.setC(1)
#     # Set parameter Gamma
#     svm.setGamma(1)
#     # Train SVM on training data
#     svm.train(np.float32(X), cv2.ml.ROW_SAMPLE, np.int32(y))
#     # Save trained model
#     svm.save("hand.yml")
#     # print(type(np.float32(X)))
#     # print(np.float32(X).shape)
def KNN(X, y):
    nbrs = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree').fit(X, y)
    with open("knn.pkl", "wb") as file:
        pickle.dump(nbrs, file)


def predict(image):
    with open('knn.pkl', 'rb') as file:
        knn = pickle.load(file)
    image = image.reshape(1, -1)
    data = PCA(image)
    testResponse = knn.predict(data)
    # print(testResponse)
    return testResponse
