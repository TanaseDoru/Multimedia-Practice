import cv2
from matplotlib import pyplot as plt
import numpy as np
import random
import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
import numpy as np 
from sklearn.cluster import DBSCAN 
from sklearn.datasets import make_blobs 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles
import tensorflow_datasets as tfds
import tensorflow as tf
import csv
import pandas as pd
from mpl_toolkits.basemap import Basemap
from pylab import rcParams
#tf.enable_eager_execution()
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import tensorflow_datasets as tfds
import tensorflow as tf
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.metrics import accuracy_score


# Empty for now
def ex1():
    # a.
    xVals = np.arange(5,79,2)
    f = [x**2 + 4*x + 7 for x in xVals]
    plt.plot(xVals, f)
    #b. 
    eVals = np.arange(0.05, 0.1, 0.05 / len(xVals))
    print("evals: ", eVals)
    g = [x**2 + 4*(x+e) + 7 for x,e in zip(xVals, eVals)]
    plt.plot(xVals, g)
    plt.show()

def ex2():
    img = cv2.imread('sudoku.png') #BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,9,7)
    result = cv2.PSNR(img, th)
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.subplot(122)
    plt.imshow(th, cmap='gray')
    plt.show()
    print("PSNR: ", result)
    


def ex3():
    trainCleanImages = [cv2.imread(file) for file in glob.glob("./train/clean/*.png")]
    trainMessyImages = [cv2.imread(file) for file in glob.glob("./train/messy/*.png")]
    testImages = [cv2.imread(file) for file in glob.glob("./test/*.png")]
    trainCleanImages = [cv2.cvtColor(file, cv2.COLOR_BGR2GRAY) for file in trainCleanImages]
    trainMessyImages = [cv2.cvtColor(file, cv2.COLOR_BGR2GRAY) for file in trainMessyImages]
    testImages = [cv2.cvtColor(file, cv2.COLOR_BGR2GRAY) for file in testImages]
    # print(len(trainCleanImages))
    def display_image(index):
        plt.imshow(trainMessyImages[index], cmap='gray')
        plt.show()

    # 299x299

    numTrain = len(trainCleanImages + trainMessyImages)
    numTest = len(testImages)
    cleanTrainData = np.zeros((len(trainCleanImages),256)).astype(int)
    for idx in range(len(trainCleanImages)):
        hist, bins = np.histogram(trainCleanImages[idx], 256, [0, 256])
        cleanTrainData[idx] = hist
    messyTrainData = np.zeros((len(trainMessyImages),256)).astype(int)
    for idx in range(len(trainMessyImages)):
        hist, bins = np.histogram(trainMessyImages[idx], 256, [0, 256])
        messyTrainData[idx] = hist
    X_train = np.concatenate((cleanTrainData,messyTrainData),axis=0)
    testData = np.zeros((len(testImages),256)).astype(int)
    for idx in range(len(testImages)):
        hist, bins = np.histogram(testImages[idx], 256, [0, 256])
        testData[idx] = hist
    X_test = testData
    y_train = np.zeros(len(trainCleanImages) + len(trainMessyImages)).astype(int)
    y_train[len(trainCleanImages):] = 1
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    minPoints = 3
    eps = 20
    dbscan = DBSCAN(eps, min_samples = minPoints)
    clusters = dbscan.fit_predict(X_train)
    # print(clusters)
    #b.
    dbscan = DBSCAN(eps, min_samples=minPoints)
    clusters = dbscan.fit_predict(X_test)
    # print(clusters)
    counter = 0
    y_test = [0, 0, 1, 0, 1, 1, 0, 1 ,1 ,0]
    for i in range(len(X_test)):
        if clusters[i] == y_test[i]:
            counter = counter + 1
    
    print("Accuray: ", counter / len(X_test))




def ex4():
    print("hello")

# ex1()
# ex2()
# ex3()
ex4()


# plt.show()
