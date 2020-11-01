from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# author-王辉
# github-https://github.com/ZEROICEWANG/ML.git
# date-2020.11.1
# e-mail-1141125530@qq.com

##SVR
def get_data():
    DataF = pd.read_excel('./西瓜数据集3.0a.xlsx')
    DataF = DataF.values.tolist()
    random.shuffle(DataF)
    DataF = np.array(DataF)
    data = DataF[:11, 0].reshape(11, 1)
    label = DataF[:11, 1]
    test = DataF[11:, 0].reshape(6, 1)
    test_label = DataF[11:, 1]

    return data, label, test, test_label


def train(data, label):
    # 自动选择合适的参数
    svr = GridSearchCV(SVR(),
                       param_grid={"kernel": ("linear", 'rbf'), "C": np.logspace(-3, 3, 7),
                                   "gamma": np.logspace(-3, 3, 7)})
    svr.fit(data, label)
    return svr


def predict(svr, data, label):
    y_pre = svr.predict(data)  # 对结果进行可视化：
    plt.plot(data, y_pre,'.', c='r', label='SVR_fit')
    plt.plot(data, label,'.', c='b', label='label')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('SVR versus Kernel Ridge')
    plt.legend()
    plt.show()
    print(svr.best_params_)


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = get_data()
    svr = train(train_data, train_label)
    predict(svr, test_data, test_label)
