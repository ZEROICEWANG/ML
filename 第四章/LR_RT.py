import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import pandas as pd
import torch
import random
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def is_number(n):
    is_number = True
    try:
        num = float(n)
        # 检查 "nan"
        is_number = num == num  # 或者使用 `math.isnan(num)`
    except ValueError:
        is_number = False
    return is_number


def get_data():
    DataF = pd.read_excel('./西瓜数据集3.0.xlsx')
    DataF = DataF.values.tolist()
    for i in range(len(DataF)):
        DataF[i].append(i)

    random.shuffle(DataF)
    DataF = np.array(DataF)
    ids = DataF[:, -1]
    DataF = DataF[:, :-1]
    dic = {}
    lis = []
    for i in range(len(DataF)):
        if DataF[i, -1] in lis:
            continue
        else:
            lis.append(DataF[i, -1])
            dic[lis[-1]] = len(lis) - 1
    for i in range(len(DataF)):
        DataF[i, -1] = dic[DataF[i, -1]]
    datas = []
    for i in range(len(DataF[0])):
        data = DataF[:, i]
        if not is_number(data[i]):
            data = pd.get_dummies(data).values
        else:
            data = np.array(data.reshape((DataF.shape[0], 1)), dtype=np.float)
        datas.append(data)
    data = datas[0]
    for i in range(1, len(datas)):
        data = np.hstack((data, datas[i]))

    DataF = torch.tensor(data, dtype=torch.float)
    data = DataF[:11, :-1]
    label = DataF[:11, -1]
    test = DataF[11:, :-1]
    test_label = DataF[11:, -1]
    return data, label, test, test_label, ids[:11]


def activate(x):
    return 1 / (1 + torch.exp(-x))


def LR(data, label):
    dim = data.shape[1]
    W = torch.rand((dim, 1)) - 0.5
    alpha = 0.05
    for epoch in range(100):
        loss = 0
        for i in range(data.shape[0]):
            f = torch.mm(data[i, :].reshape((1, dim)), W)
            f = activate(f)
            delta = (f - label[i]) * data[i, :]
            W = W - alpha * delta.t_()
            alpha -= 1e-7
            loss += label[i] * torch.log(f) + (1 - label[i]) * torch.log(1 - f)
        if epoch % 10 == 0:
            print('[epoch %d] loss:%f' % (epoch, -loss / data.shape[0]))

    f = torch.mm(data, W)
    f = activate(f)

    f[f > 0.5] = 1
    f[f <= 0.5] = 0

    return f[:, 0], W


# dic:{count:[W,left count->neg,right count->pos]}
# count=-1:neg
# count=-2:pos
def Tree(data, label, dic, key, ids):
    f, W = LR(data, label)
    neg_id = ids[f == 0]
    pos_id = ids[f == 1]
    dic[key] = [W, key + 1, key + 2, neg_id, pos_id]
    label_neg = label[f == 0]
    label_pos = label[f == 1]
    if label_neg.max() == label_neg.min():
        dic[key][1] = -label_neg[0] - 1
    else:
        Tree(data[f == 0, :], label_neg, dic, key + 1,neg_id)
    if label_pos.max() == label_pos.min():
        dic[key][2] = -label_pos[0] - 1
    else:
        Tree(data[f == 1, :], label_pos, dic, key + 2,pos_id)


def predict(test_data, label, dic):
    result = []
    key = 0
    for i in range(test_data.shape[0]):
        data = test_data[i, :].reshape((1, test_data.shape[1]))
        while True:
            W = dic[key][0]
            f = torch.mm(data, W)
            f = activate(f)
            if f > 0.5:
                f = 1
            else:
                f = 0
            temp_key = dic[key][f + 1]
            if temp_key < 0:
                result.append(-temp_key.item() - 1)
                break
            key = temp_key
    pre = np.mean([int(i == j.item()) for i, j in zip(result, label)])
    print(dic)
    print(pre)
    print(result)
    print(label)


if __name__ == '__main__':
    train_data, train_label, test_data, test_label, ids = get_data()
    dic = {}
    Tree(train_data, train_label, dic, 0, ids)
    predict(test_data, test_label, dic)
