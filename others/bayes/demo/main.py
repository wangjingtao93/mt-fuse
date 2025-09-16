import datetime
starttime = datetime.datetime.now()

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os
import cv2

X = []
Y = []

for i in range(0, 10):
    #遍历文件夹，读取图片
    for f in os.listdir("./photo/%s" % i):
        #打开一张图片并灰度化
        Images = cv2.imread("./photo/%s/%s" % (i, f)) 
        image=cv2.resize(Images,(256,256),interpolation=cv2.INTER_CUBIC)
        hist = cv2.calcHist([image], [0,1], None, [256,256], [0.0,255.0,0.0,255.0]) 
        X.append(((hist/255).flatten()))
        Y.append(i)
X = np.array(X)
Y = np.array(Y)
#切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

from sklearn.naive_bayes import BernoulliNB
clf0 = BernoulliNB().fit(X_train, y_train)
predictions0 = clf0.predict(X_test)
print (classification_report(y_test, predictions0))
endtime = datetime.datetime.now()
print (endtime - starttime)
