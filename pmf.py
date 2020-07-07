import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from numpy.random import RandomState
import copy
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import matplotlib.pyplot as plt

# https://github.com/mcleonard/pmf-pytorch/blob/master/Probability%20Matrix%20Factorization.ipynb


class PMF(torch.nn.Module):
    def __init__(self, U, V, lambda_U=1e-2, lambda_V=1e-2, latent_size=5,
                 momentum=0.8, learning_rate=0.001, iterations=1000):
        super().__init__()
        torch.manual_seed(91) 
        self.lambda_U = lambda_U
        self.lambda_V = lambda_V
        # momentum動量  避免曲折過於嚴重
        self.momentum = momentum
        # k數量
        self.latent_size = latent_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        # 用戶的潛在因子矩陣
        self.U = None
        # 商品的潛在因子矩陣
        self.V = None
        # 評分矩陣
        self.R = None
        # 指示函數
        self.I = None
        # uv哈達瑪乘積
        self.uv = None
        # 用戶數量、商品數量
        self.count_users = None
        self.count_items = None
        # gpu or cpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       

    # ======================計算RMSE=========================
    # predicts:pmf模型所預測之評分
    # truth:真實評分

    # return:RMSE數值
    # ======================================================
    def RMSE(self, predicts, truth):
        truth = torch.IntTensor([int(ele) for ele in truth])
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(predicts.float(), truth.float()))
        return loss


    # ======================計算目標式=========================
    # aspect:hcan模型預測之情感向量aspect_vec

    # return:目標式數值
    # ======================================================
    def loss(self, aspect):
        # the loss function of the model
        # 判斷是不是第一次
        if(aspect.shape[0]==0):
            loss = (torch.sum(self.I*(self.R-torch.mm(self.U, self.V.t()))**2)
                    + self.lambda_U*torch.sum(self.U.pow(2))
                    + self.lambda_V*torch.sum(self.V.pow(2)))
        else:
            aspect.requires_grad = True
            b = ((self.uv-aspect)**2)          
            loss = (torch.sum(self.I*(self.R-torch.mm(self.U, self.V.t()))**2)
                    + self.lambda_U*torch.sum(self.U.pow(2))
                    + self.lambda_V*torch.sum(self.V.pow(2))+torch.sum(b))
           
        return loss


    # ======================評分預測=========================
    # data:商品評分檔(reviewerID/asin/overall)

    # return:uv哈達瑪乘積/preds_value_array(pmf所預測之評分)
    # ======================================================
    def predict(self, data):
        tmp = torch.LongTensor([0, 1, 2]) #latent factor數量
        index_data = torch.IntTensor([[int(ele[0]), int(ele[1])] for ele in data]) #擷取reviewerID/asin
        u_index = torch.LongTensor([torch.take(i, torch.LongTensor([0])) for i in index_data]) #取得其reviewerID用戶的的潛在因子矩陣的index
        v_index = torch.LongTensor([torch.take(i, torch.LongTensor([1])) for i in index_data]) #取得其asin商品的潛在因子矩陣的index
        u_features = [torch.gather(self.U[i.item()], 0, tmp) for i in u_index] #取得其reviewerID用戶的潛在因子矩陣U
        v_features = [torch.gather(self.V[i.item()], 0, tmp) for i in v_index] #取得其asin商品的潛在因子矩陣V
        a = [torch.mul(u, v) for u, v in zip(u_features, v_features)] #UV哈達瑪乘積
        preds_value_array = torch.DoubleTensor([torch.sum(i, 0) for i in a]) #得出pmf所預測之評分
        uv = [i.detach().numpy() for i in torch.stack(a)] #轉型態tensor to array 
        return uv, preds_value_array

    # ======================更新uv哈達瑪乘積==================
    # data:商品評分檔(reviewerID/asin/overall)

    # return:none
    # ======================================================
    def uvcount(self, data):
        tmp = torch.LongTensor([0, 1, 2]) #latent factor數量
        index_data = torch.IntTensor([[int(ele[0]), int(ele[1])] for ele in data])  #擷取reviewerID/asin
        u_index = torch.LongTensor([torch.take(i, torch.LongTensor([0])) for i in index_data]) #取得其reviewerID用戶的的潛在因子矩陣的index
        v_index = torch.LongTensor([torch.take(i, torch.LongTensor([1])) for i in index_data]) #取得其asin商品的潛在因子矩陣的index
        u_features = [torch.gather(self.U[i.item()], 0, tmp) for i in u_index] #取得其reviewerID用戶的潛在因子矩陣U
        v_features = [torch.gather(self.V[i.item()], 0, tmp) for i in v_index] #取得其asin商品的潛在因子矩陣V
        a = [torch.mul(u, v) for u, v in zip(u_features, v_features)] #UV哈達瑪乘積
        self.uv = torch.stack(a) #更新uv哈達瑪乘積

    



        # ======================更新uv哈達瑪乘積==================
    
    
    # ======================pmf forward==================
    # num_users:消費者數量
    # num_items:商品數量
    # train_data:訓練資料
    # valid_data:驗證資料
    # aspect_vec:hcan輸出aspect情感向量
    # U:用戶的潛在因子矩陣
    # V:商品的潛在因子矩陣
    # uv:uv哈達瑪乘積
    # flag:計算是否為第一次訓練
    # lambda_U/lambda_V:pmf正規化參數

    # return:
    # valid_uv:驗證資料uv哈達瑪乘積list
    # uv:訓練資料uv哈達瑪乘積list
    # u_list:訓練資料用戶的潛在因子矩陣
    # v_list:訓練資料商品的潛在因子矩陣
    # loss_list/len(train_data):平均訓練資料loss
    # valid_rmse_list/len(valid_data):平均驗證資料loss
    # ======================================================
    
    def forward(self, num_users, num_items, train_data=None, valid_data=None,
                aspect_vec=None, U=None, V=None, uv=None, flag=0,
                lambda_U=0.01, lambda_V=0.01):

        train_data = Variable(torch.LongTensor(train_data)) #訓練資料轉成tensor
        valid_data = Variable(torch.LongTensor(valid_data)) #驗證資料轉成tensor
        self.lambda_U = lambda_U
        self.lambda_V = lambda_V


        aspect = [] #儲存hcan輸出aspect情感向量
        # 如果不是第一次
        if flag != 0:
            for i, a in enumerate(aspect_vec):
                for j in range(len(a)):
                    aspect.append(aspect_vec[i][j])

        # 初始化評分矩陣大小，將訓練資料評分輸入評分矩陣中
        if self.R is None:
            self.R = torch.zeros(num_users, num_items, dtype=torch.float64) # num_users:消費者數量/num_items:商品數量
            for i, ele in enumerate(train_data):
                self.R[int(ele[0]), int(ele[1])] = float(ele[2])
            self.I = copy.deepcopy(self.R) #將self.R複製到self.I
            self.I[self.I != 0] = 1 # 有評分為1 為評分為0

        if self.U is None and self.V is None:
            self.count_users = np.size(self.R, 0)
            self.count_items = np.size(self.R, 1)
            # Random 初始化
            # 用戶的潛在因子矩陣 = 消費者數量 x latent factor數量
            self.U = torch.rand(self.count_users, self.latent_size,
                                dtype=torch.float64, requires_grad=True)
            # 商品的潛在因子矩陣 = 商品數量 x latent factor數量                   
            self.V = torch.rand(self.count_items, self.latent_size,
                                dtype=torch.float64, requires_grad=True)
            # uv哈達瑪乘積 = 消費者數量 x 商品數量
            self.uv = torch.rand(self.count_users, self.count_items,
                                 dtype=torch.float64, requires_grad=True)

        else:
            self.U = torch.DoubleTensor(U)
            self.V = torch.DoubleTensor(V)
            self.uv = torch.DoubleTensor(uv)
            self.V.requires_grad = True
            self.U.requires_grad = True
            self.uv.requires_grad = True
        
        
        # ==============梯度下降法=====================
        # 梯度下降法更新U V uv
        # lr:學習率
        # momentum:動量
        # ==========================================
        optimizer = torch.optim.SGD([self.U, self.V, self.uv],
                                    lr=self.learning_rate,
                                    momentum=self.momentum)
 
        loss_list = [] #訓練資料的loss
        valid_rmse_list = [] #驗證資料的loss

        # ====================training=======================
        # 進行self.iterations次
        # 每一epoch訓練一筆資料
        # 更新uv哈達瑪乘積
        # 獲得訓練資料loss/驗證資料loss/驗證資料rmse
        # =====================================================
        for step, epoch in enumerate(range(self.iterations)):
            optimizer.zero_grad()
            self.uvcount(train_data)
            loss = self.loss(torch.DoubleTensor(aspect))
            loss_list.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()

            valid_uv, valid_predicts = self.predict(valid_data)
            valid_rmse = self.RMSE(valid_predicts, valid_data[:, 2])
            valid_rmse_list.append(valid_rmse.detach().numpy())

            if step % 50 == 0:
                print(f'Step {step}\tLoss: {loss:.4f}(train)\tRMSE: {valid_rmse:.4f}(valid)')

        u_list = [i.detach().numpy() for i in self.U] #將self.U tensor轉型成array
        v_list = [i.detach().numpy() for i in self.V] #將self.V tensor轉型成array
        uv = [i.detach().numpy() for i in self.uv] #將self.uv tensor轉型成array
        
        loss_list = np.sum(loss_list) #加總訓練資料loss
        valid_rmse_list = np.sum(valid_rmse_list) #加總驗證資料loss

        return valid_uv, uv, u_list, v_list, loss_list/len(train_data), valid_rmse_list/len(valid_data)       
