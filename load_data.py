# coding=utf-8
import pickle
import numpy as np 
import random
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


# ============load_data==========
# 讀取檔案全部資料
# filename:檔名

# return:a_dict1 該檔全部資料
# ====================================
def load_data(filename):
    with open(filename, 'rb') as file:
        a_dict1 = pickle.load(file)

    return a_dict1



# ============cut_data_len==========
# 從data中擷取所需欄位之值不重複之數量
#data:所有資料
# field:欄位

# return:其不重複值的數量
# ====================================
def cut_data_len(data, field):
    result = []
    for d in data:
        if field == 'reviewerID':
            result.append(d[0])
        elif field == 'asin':
            result.append(d[1])
        else:
            result.append(d[2])

    return int(len(np.unique(result)))



