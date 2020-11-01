import numpy as np
import matplotlib.pyplot as plt
import random
import time
from roc import ROC, get_microFPR, cacu_AUC


# author-王辉
# github-https://github.com/ZEROICEWANG/ML.git
# date-2020.11.1
# e-mail-1141125530@qq.com

##BP算法--adam优化器，多分类

class Data(object):

    def __init__(self):
        self.data_path = r'./wine/wine.npz'
        self.label = None
        self.data = None
        self.train_data = None
        self.train_label = None
        self.test_label = None
        self.test_data = None
        self.train_rate = 0.7

    def get_data(self):
        Data = np.load(self.data_path)
        data = Data['arr_0']
        self.data = data[:, :-1]
        dims = self.data.shape[1]
        self.label = data[:, -1]
        self.data = (self.data - np.min(self.data, axis=0).reshape((1, dims))) / (
                np.max(self.data, axis=0) - np.min(self.data, axis=0)).reshape((1, dims))
        self.label = np.array(self.label).reshape(len(self.label), 1)
        label = np.zeros((len(self.label), int(max(self.label) + 1)))
        for i in range(len(self.label)):
            label[i, int(self.label[i])] = 1
        self.label = label

    def div_data(self):
        number = self.data.shape[0]
        train_size = int(number * self.train_rate)
        self.train_data = self.data[:train_size, :]
        self.train_label = self.label[:train_size, :]
        self.test_data = self.data[train_size:, :]
        self.test_label = self.label[train_size:, :]


def sigmod(z):
    return 1.0 / (1.0 + np.exp(-z))


def dsigmod(item):
    return item * (1 - item)


def softmax(z):
    result = np.exp(z) / np.sum(np.exp(z))
    return result


def LRelu(item):
    item = item.copy()
    for i in range(item.shape[0]):
        if item[i, 0] > 1:
            item[i, 0] = item[i, 0] * 0.01 + 1
        elif item[i, 0] < 0:
            item[i, 0] = item[i, 0] * 0.01
    return item


def dLRelu(item):
    item = item.copy()
    for i in range(item.shape[0]):
        if item[i, 0] > 1:
            item[i, 0] = 0.01
            continue
        elif item[i, 0] > 0:
            item[i, 0] = 1
            continue
        else:
            item[i, 0] = 0.01
    return item


class mlp(object):
    def __init__(self, lr=0.001, te=1e-5, epoch=100, batch=800, size=None):
        self.learningRate = lr
        self.rho1 = 0.9
        self.rho2 = 0.999
        self.delta = 1e-8
        self.sw = [0 for i in range(len(size) - 1)]
        self.sb = [0 for i in range(len(size) - 1)]
        self.rw = [0 for i in range(len(size) - 1)]
        self.rb = [0 for i in range(len(size) - 1)]
        self.thresholdError = te
        self.maxEpoch = epoch
        self.size = size
        self.batch = batch
        self.input = None
        self.label = None
        self.result = None
        self.mid_result = None
        self.adam=False
        self.errors = 0
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

    def forwardP(self, item=None):
        a = [item]
        for wIndex in range(len(self.W) - 1):
            a.append(LRelu(np.dot(self.W[wIndex], a[-1]) + self.b[wIndex]))
        a.append(softmax(np.dot(self.W[len(self.W) - 1], a[-1]) + self.b[len(self.W) - 1]))
        return a

    def cacu_result(self, start, end):
        self.result = []
        self.mid_result = []
        self.errors = 0
        for index in range(start, end):
            label = self.label[index, :]
            all_result = self.forwardP(self.input[index, :].reshape((self.size[0], 1)))
            self.result.append(all_result[-1])
            self.mid_result.append(all_result)
            self.errors += np.sum(-label.reshape(self.size[-1], 1) * np.log(all_result[-1] + 1e-16))

    def backP(self, t, start, end):
        deltas = []
        for i in range(start, end):
            label = self.label[i, :]
            a = self.mid_result[i - start]
            delta = [-label.reshape(self.size[-1], 1) + a[-1]]
            for j in range(len(self.W) - 1):
                d = np.multiply(np.dot(self.W[-1 - j].T, delta[-1]), dLRelu(a[-2 - j]))
                delta.append(d)
            deltas.append(delta)
        deltaW = [0 for i in range(len(delta))]
        deltaB = deltaW.copy()
        for i in range(start, end):
            a = self.mid_result[i - start]
            delta = deltas[i - start]
            for j in range(len(delta)):
                deltaW[j] += np.dot(delta[j], a[-2 - j].T)
                deltaB[j] += delta[j]
        for i in range(len(deltaW)):
            if not self.adam:
                deltaw = deltaW[i] / (end-start)
                deltab = deltaB[i] / (end-start)
                self.W[-1 - i] = self.W[-1 - i] - self.learningRate * deltaw
                self.b[-1 - i] = self.b[-1 - i] - self.learningRate * deltab
            else:
                deltaw = deltaW[i] / (end - start)
                self.sw[i] = self.sw[i] * self.rho1 + (1 - self.rho1) * deltaw
                self.rw[i] = self.rw[i] * self.rho2 + (1 - self.rho2) * deltaw * deltaw
                s_revise = self.sw[i] / (1 - self.rho1 ** t)
                r_revise = self.rw[i] / (1 - self.rho2 ** t)
                self.W[-1 - i] = self.W[-1 - i] - self.learningRate * s_revise / (self.delta + r_revise ** 0.5)

                deltab = deltaB[i] / (end - start)
                self.sb[i] = self.sb[i] * self.rho1 + (1 - self.rho1) * deltab
                self.rb[i] = self.rb[i] * self.rho2 + (1 - self.rho2) * deltab * deltab
                s_revise = self.sb[i] / (1 - self.rho1 ** t)
                r_revise = self.rb[i] / (1 - self.rho2 ** t)
                self.b[-1 - i] = self.b[-1 - i] - self.learningRate * s_revise / (self.delta + r_revise ** 0.5)

    def train(self):
        count = 0
        for ep in range(self.maxEpoch):
            for i in range(0, self.label.shape[0], self.batch):
                count += 1
                if i + self.batch > self.label.shape[0]:
                    end = self.label.shape[0]
                else:
                    end = self.batch + i
                self.cacu_result(i, end)
                self.backP(count, i, end)
                e = self.errors / (end - i)
                print('[epoch:%d] [%d/%d loss=%.5f]' % (ep, end, self.label.shape[0], e))
                if e < self.thresholdError:
                    self.save_weight()
                    return
            self.save_weight()
        self.save_weight()

    def predict(self, data, label):
        results = []
        pre_rate = []
        one_label = []
        resultsf = []
        for i in range(data.shape[0]):
            result = self.forwardP(item=data[i, :].reshape((self.size[0], 1)))[-1]
            results.append(np.argmax(result))
            one_label.append(np.argmax(label[i, :]))
            resultsf.append(result)
        ACC = sum(int(a == b) for a, b in zip(results, one_label)) / len(one_label)
        P, R, F1, TPR, FPR = get_microFPR(results, one_label)
        pre_rate.append(ACC)
        print('pre:precise:%.5f' % max(pre_rate))
        print(ACC, P, R, F1)
        roc = np.array(ROC(np.array(resultsf), one_label))
        cacu_AUC(roc)
        plt.plot(roc[:, 0], roc[:, 1])
        plt.title('mlp_multiclass')
        plt.show()
        return


if __name__ == "__main__":
    path = './wine/wine.npz'
    start = time.time()
    data = Data()
    data.data_path = path
    data.get_data()
    data.div_data()
    model = mlp(lr=0.001, te=1e-5, epoch=100, batch=20, size=[data.train_data.shape[1], 5, data.train_label.shape[1]])
    model.adam=False
    model.feed_data(data.train_data, data.train_label)
    model.train()
    model.load_wight()
    model.predict(data.test_data, data.test_label)
    print('time cost:%.5f' % (time.time() - start))
