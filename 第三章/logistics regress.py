import torch
import pandas as pd
import random


def activate(x):
    return 1 / (1 + torch.exp(-x))


def get_data():
    DataF = pd.read_excel('./西瓜数据集3.0a.xlsx')
    DataF = DataF.values.tolist()
    random.shuffle(DataF)
    DataF = torch.tensor(DataF)
    data = DataF[:11, :-1]
    label = DataF[:11, -1]
    test = DataF[11:, :-1]
    test_label = DataF[11:, -1]

    return data.cuda(), label.cuda(),test.cuda(),test_label.cuda()


def LR(data, label,test_data,test_label):
    dim = data.shape[1]
    W = torch.rand((dim, 1)).cuda() - 0.5
    alpha = 0.05
    for epoch in range(1000):
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

    f = torch.mm(test_data, W)
    f = activate(f)

    f[f > 0.5] = 1
    f[f <= 0.5] = 0
    print(sum([int(i == j) for i, j in zip(f, test_label)]) / test_data.shape[0])
    print(f)
    print(label)

if __name__ == '__main__':
    data, label,test_data,test_label = get_data()
    data = torch.cat((data, torch.ones((data.shape[0], 1)).cuda()), dim=1)
    test_data = torch.cat((test_data, torch.ones((test_data.shape[0], 1)).cuda()), dim=1)
    LR(data, label,test_data,test_label)
