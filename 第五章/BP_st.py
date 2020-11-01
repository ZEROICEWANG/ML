# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

# author-王辉
# github-https://github.com/ZEROICEWANG/ML.git
# date-2020.11.1
# e-mail-1141125530@qq.com

##标准BP算法


def is_number(n):
    is_number = True
    try:
        num = float(n)
        # 检查 "nan"
        is_number = num == num  # 或者使用 `math.isnan(num)`
    except ValueError:
        is_number = False
    return is_number


class Data(object):

    def __init__(self):
        self.label = None
        self.data = None
        self.test_label = None
        self.test_data = None

    def get_data(self):
        DataF = pd.read_excel('./西瓜数据集3.0.xlsx')
        DataF = DataF.values.tolist()
        random.shuffle(DataF)
        DataF = np.array(DataF)
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

        DataF=np.array(data,dtype=np.float)
        data = DataF[:11, :-1]
        label = DataF[:11, -1]
        test = DataF[11:, :-1]
        test_label = DataF[11:, -1]
        self.data = data
        self.label = label.reshape((label.shape[0],1))
        self.test_data = test
        self.test_label = test_label.reshape((test_label.shape[0],1))


def sigmod(z):
    return 1.0 / (1.0 + np.exp(-z))


class mlp(object):
    def __init__(self, lr=0.001, delay=1e-7, lda=0.0, te=1e-5, epoch=100, batch=800, size=None):
        self.learningRate = lr
        self.delay = delay
        self.lambda_ = lda
        self.thresholdError = te
        self.maxEpoch = epoch
        self.size = size
        self.batch = batch
        self.input = None
        self.label = None
        self.result = None
        self.mid_result = None
        self.error_array = None
        self.correct_list = None
        self.W = []
        self.b = []
        self.init()

    def init(self):
        for i in range(len(self.size) - 1):
            self.W.append(np.array(np.random.uniform(-0.5, 0.5, size=(self.size[i + 1], self.size[i]))))
            self.b.append(np.array(np.random.uniform(-0.5, 0.5, size=(self.size[i + 1], 1))))

    def feed_data(self, data, label):
        self.input = data
        self.label = label

    def save_weight(self):
        W = np.array(self.W)
        b = np.array(self.b)
        np.savez('weight', W, b)

    def load_wight(self):
        Data = np.load('weight.npz', allow_pickle=True)
        self.W = Data['arr_0'].tolist()
        self.b = Data['arr_1'].tolist()

    def forwardPropagation(self, item=None):
        a = [item]
        for wIndex in range(len(self.W)):
            a.append(sigmod(np.dot(self.W[wIndex], a[-1]) + self.b[wIndex]))
        return a

    def cacu_result(self):
        self.result = []
        self.mid_result = []
        self.correct_list = []
        self.error_array = []
        for index in random.sample(range(self.input.shape[0]), self.batch):
            all_result = self.forwardPropagation(self.input[index, :].reshape((self.size[0], 1)))
            self.result.append(all_result[-1])
            self.mid_result.append(all_result)
            self.error_array.append(0.5 * (self.label[index] - all_result[-1]) ** 2)
            self.correct_list.append(index)

    def backPropagation(self):
        deltas = []
        for i in range(len(self.correct_list)):
            label = self.label[self.correct_list[i], :]
            a = self.mid_result[i]
            delta = [(a[-1] - label) * a[-1] * (1.0 - a[-1])]
            for j in range(len(self.W) - 1):
                abc = np.multiply(a[-2 - j], 1 - a[-2 - j])
                cba = np.multiply(np.dot(self.W[-1 - j].T, delta[-1]), abc)
                delta.append(cba)
            deltas.append(delta)
        deltaW = [0 for i in range(len(delta))]
        deltaB = [0 for i in range(len(delta))]
        for i in range(len(self.correct_list)):
            a = self.mid_result[i]
            delta = deltas[i]
            for j in range(len(delta)):
                deltaW[j] += np.dot(delta[j], a[-2 - j].T)
                deltaB[j] += delta[j]
        for i in range(len(deltaW)):
            deltaw = deltaW[i] / len(self.correct_list)
            deltab = deltaB[i] / len(self.correct_list)
            self.W[-1 - i] = self.W[-1 - i] - self.learningRate * deltaw
            self.b[-1 - i] = self.b[-1 - i] - self.learningRate * deltab

    def train(self):
        for ep in range(self.maxEpoch):
            self.cacu_result()
            tt = sum(self.error_array) / self.batch
            if tt < self.thresholdError:
                break
            self.backPropagation()
            self.learningRate -= self.delay
            #print("epoch {0}: ".format(ep), tt)

        self.save_weight()

    def predict(self, data, label):
        results = []
        for i in range(data.shape[0]):
            result = self.forwardPropagation(item=data[i, :].reshape((self.size[0], 1)))[-1]
            if result > 0.5:
                result = 1
            else:
                result = 0
            results.append(result)
        sums = sum(int(a == b) for a, b in zip(results, label))
        print('pre:precise:%.5f' % ((sums) / len(label)))
        return


if __name__ == "__main__":
    for i in range(10):
        data = Data()
        data.get_data()
        model = mlp(lr=0.001, delay=1e-7, lda=0.0, te=1e-5, epoch=500, batch=1, size=[data.data.shape[1], 100, 1])
        model.feed_data(data.data, data.label)
        model.train()
        model.load_wight()
        model.predict(data.data, data.label)
