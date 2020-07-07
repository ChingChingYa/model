import matplotlib.pyplot as plt
import numpy as np
from load_data import load_data, cut_data_len
from sklearn.model_selection import train_test_split
from pmf import PMF
from sklearn.utils import shuffle
from hcan_train import Config
import pickle
import torch
from torch.autograd import Variable

test_list = []
test_rmse = []
config = Config()
aspect_vec = []
uv = []
train_uv = []


# PMF parameter
ratio = 0.8
lambda_U = 0.01
lambda_V = 0.01
latent_size = 3
learning_rate = 3e-5  # 3e-5
iterations = 1000

lambda_value_list = []
lambda_value_list.append([0.01, 0.01])

if __name__ == "__main__":
    # 讀取商品評分檔
    alldata = load_data('./data/prouduct_rating_data_Musical_Instruments.pickle')
    
    # 分割train 80%/test 10%/val 10%
    train, test = train_test_split(alldata, test_size=0.2, random_state=1)# alldata:整體資料/test_size=0.2:測試資料20%/random_state:隨機參數保證每次分割都一樣
    test, val = train_test_split(test, test_size=0.2, random_state=1)
    train = Variable(torch.LongTensor(train))
    test = Variable(torch.LongTensor(test))
    val = Variable(torch.LongTensor(val))

    # cut_data_len:計算用戶數量、商品數量
    num_users = cut_data_len(alldata, 'reviewerID')
    num_items = cut_data_len(alldata, 'asin')

    fp = open("log.txt", "a")
    fp.write("dataset:" + "Digital_Music"+"\n")
    fp.write("ratio:" + str(ratio)+"\n")
    fp.write("latent_factor:" + str(latent_size)+"\n")
    fp.write("learning_rate:" + str(learning_rate)+"\n")
    
    # initialization pmf
    for lambda_value in lambda_value_list:
        lambda_U = lambda_value[0]
        lambda_V = lambda_value[1]
        
        pmf_model = PMF(U=None, V=None, lambda_U=lambda_U,
                        lambda_V=lambda_V, latent_size=latent_size,
                        momentum=0.8, learning_rate=learning_rate,
                        iterations=iterations)

        s = ('parameters are:ratio={:f}, reg_u={:f}, reg_v={:f}, latent_size={:d},'
             + 'learning_rate={:f}, iterations={:d}')
        print(s.format(ratio, lambda_U, lambda_V, latent_size,
              learning_rate, iterations))
        print("batch size", 32)
        print("latent_size", 3)
        U = None
        V = None
        uv = None

        fp.write("=============================== Lambda Value ============="+"\n")
        fp.write("lambda_U:" + str(lambda_U)+"\n")
        fp.write("lambda_V:" + str(lambda_V)+"\n")
        # rmse均方根誤差
        rmse_minus = 1 # 存放本次pmf的test rmse
        rmse_temp = 0 # 存放上次pmf的test rmse
        round_value = 0 # 計算目前交互訓練次數


        while rmse_minus > 0.001:
            print("=============================== Round =================================")
            print(round_value)
            fp.write("=============================== Round ================================="+"\n")
            fp.write(str(round_value)+"\n")

            # 判斷是否為第一次
            if round_value == 0:
                valid_uv, uv, U, V, train_loss, val_rmse = pmf_model(num_users=num_users, num_items=num_items,
                                                                     train_data=train, valid_data=val, aspect_vec=None, U=U, V=V, uv=uv, flag=round_value, 
                                                                     lambda_U=lambda_U, lambda_V=lambda_V) # go to pmf.py-forward
            else:
                valid_uv, uv, U, V, train_loss, val_rmse = pmf_model(num_users=num_users, num_items=num_items, 
                                                                     train_data=train, valid_data=val, aspect_vec=aspect_vec, U=U, V=V, uv=uv, flag=round_value,
                                                                     lambda_U=lambda_U, lambda_V=lambda_V) # go to pmf.py-forward
        
            aspect_vec.clear()
            # 更新uv哈達瑪乘積
            train_uv.clear()
            train_uv = uv

            print('testing PMFmodel.......')
            test_uv, preds = pmf_model.predict(data=test)
            test_rmse = pmf_model.RMSE(preds, test[:, 2])

            print("=============================== RMSE =================================")
            print('test rmse:{:f}'.format(test_rmse.float()))
            fp.write("================================ RMSE =========================="+"\n")
            fp.write(str('test rmse:{:f}'.format(test_rmse.float()))+"\n")
      
            # abs 絕對值 計算上次根本次rmse之間誤差
            rmse_minus = abs(rmse_temp - test_rmse.float())
            
            rmse_temp = test_rmse
            print(rmse_minus)

            # 判斷是否收斂至小
            if rmse_minus > 0.001:

                # 將uv哈德瑪乘積傳入HCAN模型中，進行訓練
                aspect_vec, aspects = config.Setup(train_uv, valid_uv, test_uv) # go to hcan_train.py-Setup
                
                # 輸出aspect vector
                pickle.dump((aspects),
                            open('./data/aspect_vec'+str(round_value)+'.pickle', 'wb'))  
                # aspect_vec.append(sentiment_labels)

            round_value = round_value + 1

    fp.close()
