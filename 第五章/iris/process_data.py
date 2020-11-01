import numpy as np
import random
import pandas as pd

path = './iris.txt'
data = []
dic = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
with open(path, 'r') as f:
    lines_list = f.readlines()
    for line in lines_list:
        line_list=line.strip('\n').split(',')
        data.append([])
        for val in line_list[:-1]:
            data[-1].append(float(val))
        data[-1].append(dic[line_list[-1]])
random.shuffle(data)
data=np.array(data)
np.savez('iris.npz',data)
DataF=pd.DataFrame(data)
writer=pd.ExcelWriter('./iris.xls')
DataF.to_excel(excel_writer=writer)
writer.close()