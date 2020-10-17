import torch
import pandas as pd
import matplotlib.pyplot as plt


def get_data():
    DataF = pd.read_excel('./西瓜数据集3.0a.xlsx')
    DataF = DataF.values
    pos_data, pos_label, neg_data, neg_label = [], [], [], []
    for i in range(DataF.shape[0]):
        if DataF[i, -1] == 1:
            pos_data.append(DataF[i, :-1])
            pos_label.append(DataF[i, -1])
        else:
            neg_data.append(DataF[i, :-1])
            neg_label.append(DataF[i, -1])
    return torch.tensor(pos_data).cuda(), torch.tensor(pos_label).cuda(), \
           torch.tensor(neg_data).cuda(), torch.tensor(neg_label).cuda()


def Fisher(data):
    dim = data[0].shape[1]
    u0 = torch.mean(data[0], dim=0).reshape((dim, 1))
    u1 = torch.mean(data[1], dim=0).reshape((dim, 1))
    Sw = 0
    for i in range(data[0].shape[0]):
        Sw += torch.mm((data[0][i, :].reshape((dim, 1)) - u0), (data[0][i, :].reshape((dim, 1)) - u0).t_())
    for i in range(data[1].shape[0]):
        Sw += torch.mm((data[1][i, :].reshape((dim, 1)) - u1), (data[1][i, :].reshape((dim, 1)) - u1).t_())
    w = torch.mm(torch.inverse(Sw), (u0 - u1))
    return w, u0, u1


if __name__ == '__main__':
    p_data, p_label, n_data, n_label = get_data()
    w, u0, u1 = Fisher([p_data, n_data])
    u0 = torch.mm(u0.t_(),w)
    u1 = torch.mm(u1.t_(),w)
    p_data = torch.mm(p_data, w)
    n_data = torch.mm(n_data, w)
    plt.figure(0)
    plt.plot(p_data.tolist(), [0 for i in range(len(p_label))], '.', color='r')
    plt.plot(n_data.tolist(), [0 for i in range(len(n_label))], '.', color='b')
    plt.plot([u0.item()], [0], '*', color='r')
    plt.plot([u1.item()], [0], '*', color='b')
    plt.show()
