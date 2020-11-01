import numpy as np
import random
import pandas as pd

path='wine.txt'
data=[]
with open(path,'r') as f:
    lines=f.readlines()
    for line in lines:
        data.append([])
        line=line.strip('\n').split(',')
        for val in line[1:]:
            data[-1].append(float(val))
        data[-1].append(float(line[0])-1)
random.shuffle(data)
data=np.array(data)
np.savez('wine.npz',data)
DataF=pd.DataFrame(data)
writer=pd.ExcelWriter('./wine.xls')
DataF.to_excel(excel_writer=writer)
writer.close()
