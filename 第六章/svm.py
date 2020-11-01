import pickle
import gzip
import pandas as pd
import numpy as np
from sklearn import svm
import time
import os
import cv2 as cv
import random

# author-王辉
# github-https://github.com/ZEROICEWANG/ML.git
# date-2020.11.1
# e-mail-1141125530@qq.com

##SVM
def get_data():
    DataF = pd.read_excel('./西瓜数据集3.0a.xlsx')
    DataF = DataF.values.tolist()
    random.shuffle(DataF)
    DataF = np.array(DataF)
    data = DataF[:11, :-1]
    label = DataF[:11, -1]
    test = DataF[11:, :-1]
    test_label = DataF[11:, -1]
    test = test.tolist()
    for i in range(len(test)):
        test[i] = np.array(test[i]).reshape((1, data.shape[1]))

    return data, label, test, test_label


def svm_train(C, gamma):
    training_data, label, test_data, test_label = get_data()
    model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
    model.fit(training_data, label)
    predictions = [int(model.predict(data)) for data in test_data]
    num_correct = np.mean([int(a == y) for a, y in zip(predictions, test_label)])
    print("preci  %f" %num_correct)


if __name__ == "__main__":
    C = 200
    gamma = 0.003
    for i in range(10):
        svm_train(C, gamma)
