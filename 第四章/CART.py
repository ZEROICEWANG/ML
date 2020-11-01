import pandas as pd
import torch
import numpy as np

# author-王辉
# github-https://github.com/ZEROICEWANG/ML.git
# date-2020.11.1
# e-mail-1141125530@qq.com

##基尼决策树
def train_encoder(data):
    classes = 0
    collected_class = []
    dic = {}
    for i in data:
        if i in collected_class:
            continue
        else:
            dic[i] = classes
            classes += 1
            collected_class.append(i)
    for i in range(len(data)):
        data[i] = dic[data[i]]
    return data, dic


def test_encoder(data, dic):
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i, j] = dic[j][data[i, j]]
    return data


def get_data():
    train_path = './训练集.xlsx'
    test_path = './测试集.xlsx'
    DataF = pd.read_excel(train_path)
    train_data = DataF.values
    dic = []
    for i in range(len(train_data[0, :])):
        train_data[:, i], part_dic = train_encoder(train_data[:, i])
        dic.append(part_dic)
    DataF = pd.read_excel(test_path)
    test_data = DataF.values
    test_data = test_encoder(test_data, dic)
    train_data = np.array(train_data, dtype=np.int)
    test_data = np.array(test_data, dtype=np.int)
    return train_data, test_data


def cacu_Gini(data, left_point):
    Gini = []
    for point in left_point:
        Gini.append(0)
        for i in range(data[:, point].max() + 1):
            p1 = 0
            p2 = 0
            for j in range(len(data)):
                if data[j, point] == i:
                    p1 += data[j, -1]
                    p2 += (1 - data[j, -1])
            sums = p1 + p2
            p1 /= sums + 1e-16
            p2 /= sums + 1e-16
            gini = 1 - p1 ** 2 - p2 ** 2
            Gini[-1] += sums / len(data) * gini
    return Gini


def Tree(data: np.array, left_point: list, used_point: dict, dic_index, dic_position):
    if np.max(data[:, -1]) == np.min(data[:, -1]):
        used_point[dic_index][dic_position] = -data[0, -1]-1
        return
    if len(data) == 0:
        return
    count = 0
    for index in left_point:
        if np.max(data[:, index]) == np.min(data[:, index]):
            count += 1
    if count == len(left_point):
        return
    Gini_index = cacu_Gini(data, left_point)
    index = np.argmin(Gini_index)
    point = left_point[index]
    left_point.pop(index)
    used_point[dic_index][dic_position] = point
    used_point[point] = [-1 for i in range(data[:, point].max() + 1)]
    for i in range(data[:, point].max() + 1):
        Tree(data[data[:, point] == i, :], left_point, used_point, point, i)
    return


def predict(dic, test_data):
    result = []
    label = []
    for i in range(len(test_data)):
        point = dic[-1][0]
        while True:
            data = test_data[i, point]
            new_point = dic[point][data]
            if new_point < 0:
                result.append(-dic[point][data]-1)
                label.append([test_data[i, -1]])
                break
            point = new_point
    pre=np.mean([int(i==j) for i,j in zip(result,label)])
    print(pre)


if __name__ == '__main__':
    train_data, test_data = get_data()
    left_point = [i for i in range(train_data.shape[1] - 1)]
    used_point = {-1: [0]}
    Tree(train_data, left_point, used_point, -1, 0)
    print(used_point)
    predict(used_point,test_data)
