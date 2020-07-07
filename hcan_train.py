from models import Proto_CNN as HCAN
from load_data import load_data
from gensim.models import KeyedVectors
import torch
import time
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn import metrics
import numpy as np
import torch.utils.data as Data
from torchsummary import summary
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchviz import make_dot       
from graphviz import Digraph

# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html


class Config(object):
    def __init__(self):
        self.model_name = 'HCAN'
        # Aspect數量
        self.num_classes = 4 
        # 詞表
        self.vocab_path = './voc/word2vec2.pickle'
        # 設備
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = 0.4                                              
        # 词表大小
        self.n_vocab = 41779
        self.num_epochs = 10                                         
        self.batch_size = 32                                           
        self.learning_rate = 3e-5 
        # self.learning_rate = 0.01                         
        self.embed_dim = 300
        # 字向量维度
        self.filter_sizes = [3, 3]
        # 卷积核尺寸
        self.hidden_dim = 80
        self.sent_maxlen = 10
        self.word_maxlen = 34

        # 讀取文字評論檔(文本+評分)
        print("Loading data...")
        alldata = load_data('./data/review_comment_data_Musical_Instruments.pickle')
        # self.x_train_data:文本訓練資料
        # self.x_val_data:文本驗證資料
        # self.x_test_data:文本評論測試資料
        # self.y_train_data:評分訓練資料
        # self.y_val_data:評分驗證資料
        # self.y_test_data:評分測試資料
        self.x_train_data, self.x_test_data = train_test_split(alldata[0], test_size=0.2, random_state=1)
        self.x_test_data, self.x_val_data = train_test_split(self.x_test_data, test_size=0.2, random_state=1)
        self.y_train_data, self.y_test_data = train_test_split(alldata[1], test_size=0.2, random_state=1)
        self.y_test_data, self.y_val_data = train_test_split(self.y_test_data, test_size=0.2, random_state=1)


    # =================測試模型==========================
    # model:HACN模型
    # test:文本評論測試資料
    # rating:評分測試資料
    # pmf_muti:pmf模型輸出uv哈達瑪乘積

    # return:
    # test_loss / len(test): 測試資料平均loss
    # test_acc / len(test): 測試資料平均acc
    # ===================================================
    def test(self, model, test, rating, pmf_muti):
        input_batch = Variable(torch.LongTensor(test)).to(self.device) #文本評論測試資料轉成tensor型態
        rating_batch = Variable(torch.LongTensor(rating)).to(self.device) #評分測試資料轉成tensor型態
        sentiment_batch = Variable(torch.FloatTensor(pmf_muti)).to(self.device) #模型輸出uv哈達瑪乘積轉成tensor型態
        test_loss = 0 #測試資料loss
        test_acc = 0 #測試資料acc準確率
        torch_dataset = Data.TensorDataset(input_batch, rating_batch, sentiment_batch)
        test_loader = Data.DataLoader(dataset=torch_dataset, 
                                      batch_size=self.batch_size)


        # ====================training=======================
        # 每一epoch訓練一個batch size大小的資料
        # batch_x:文本評論測試資料
        # batch_y:評分測試資料
        # batch_z:模型輸出uv哈達瑪乘積
        # 進入hcan的forward，獲得hcan最終輸出
        # pmf_vec:為模型的輸出aspect情感向量
        # rating_vec:為模型的輸出文檔的評分預測
        # aspect_vec:為模型的輸出aspect特徵向量
        # =====================================================
        for i, (batch_x, batch_y, batch_z) in enumerate(test_loader):

            pmf_vec, rating_vec, aspect_vec = model(batch_x)  # 進入hcan的forward
            loss = model.loss(batch_z, batch_y, pmf_vec, rating_vec) # 計算loss
            acc = model.accuracy(rating_vec, batch_y) # 計算acc
            test_loss += loss.item()
            test_acc += acc.item()

        return test_loss / len(test), test_acc / len(test)
    


    # =================訓練模型==========================
    # model:HACN模型
    # tain:文本評論訓練資料
    # rating:評分訓練資料
    # pmf_muti:pmf模型輸出uv哈達瑪乘積

    # return:
    # train_loss / len(tain):訓練資料平均loss
    # train_acc / len(tain):訓練資料平均acc
    # pmf:hcan模型的輸出aspect情感向量
    # aspect_vec:can模型的輸出aspect特徵向量
    # ===================================================
    def train(self, model, tain, rating, pmf_muti):

        input_batch = Variable(torch.LongTensor(tain)).to(self.device) #文本評論訓練資料轉成tensor型態
        rating_batch = Variable(torch.LongTensor(rating)).to(self.device) #評分訓練資料轉成tensor型態
        sentiment_batch = Variable(torch.FloatTensor(pmf_muti)).to(self.device) #模型輸出uv哈達瑪乘積轉成tensor型態

        torch_dataset = Data.TensorDataset(input_batch, rating_batch, sentiment_batch)
        # 優化器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
        # Train the model
        train_loss = 0 #訓練資料loss
        train_acc = 0 #訓練資料acc準確率
        pmf = []
        train_loader = Data.DataLoader(dataset=torch_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True)
   
        # ====================training=======================
        # 每一epoch訓練一個batch size大小的資料
        # batch_x:文本評論訓練資料
        # batch_y:真實評分訓練資料
        # batch_z:模型輸出uv哈達瑪乘積
        # 進入hcan的forward，獲得hcan最終輸出
        # pmf_vec:為模型的輸出aspect情感向量
        # rating_vec:為模型的輸出文檔的評分預測
        # aspect_vec:為模型的輸出aspect特徵向量
        # =====================================================
        for i, (batch_x, batch_y, batch_z) in enumerate(train_loader):
            optimizer.zero_grad()
            pmf_vec, rating_vec, aspect_vec = model(batch_x) # go to models.py-Proto.forward
            pmf.append(pmf_vec)
           
            loss = model.loss(batch_z, batch_y, pmf_vec, rating_vec) # 分別計算loss
            acc = model.accuracy(rating_vec, batch_y) # 計算評分的準確率

            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_acc += acc.item()

        # Adjust the learning rate
        scheduler.step()

        return (train_loss / len(tain), train_acc / len(tain), pmf, aspect_vec)



    # =================初始化模型==========================
    # pmf_muti_train:pmf uv train資料
    # pmf_muti_val:pmf uv val資料
    # pmf_muti_test:pmf uv test資料

    # return:
    # pmf_vec:hcan模型的輸出aspect特徵向量
    # aspect_vec:hcan模型的輸出aspect情感向量
    # ===================================================
    def Setup(self, pmf_muti_train, pmf_muti_val, pmf_muti_test):
 
        pmf_muti_train = [i.astype(np.float32) for i in pmf_muti_train]
        pmf_muti_val = [i.astype(np.float32) for i in pmf_muti_val]
        pmf_muti_test = [i.astype(np.float32) for i in pmf_muti_test]

        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样

        # 讀取詞表
        wvmodel = KeyedVectors.load_word2vec_format(self.vocab_path)

        # 初始化HACN
        model = HCAN(self.embed_dim, self.hidden_dim, self.filter_sizes,
                     self.sent_maxlen, self.word_maxlen, self.num_classes,
                     self.dropout, self.n_vocab, wvmodel.vectors).to(self.device)

        # =====================================================
        # 總共進行num_epochs次訓練，再分別進行train和vaild

        # train
        # train_loss: 訓練資料loss
        # train_acc: 訓練資料acc準確率
        # pmf_vec: 訓練資料aspect 情感向量
        # aspect_vec: 訓練資料aspect 特徵向量

        # vaild
        # valid_loss: 驗證資料loss
        # valid_acc: 驗證資料acc準確率

        # 最後模型才進行test
        # test_loss: 測試資料loss
        # test_acc: 測試資料acc準確率
        # =====================================================
        for epoch in range(self.num_epochs):
            start_time = time.time()
            # train
            train_loss, train_acc, pmf_vec, aspect_vec = self.train(model, self.x_train_data, self.y_train_data, pmf_muti_train) # go to Config.train
            # vaild
            valid_loss, valid_acc = self.test(model, self.x_val_data, self.y_val_data, pmf_muti_val) # go to Config.test



            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
            print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
            print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
            

        # test
        print('Checking the results of test dataset...')
        test_loss, test_acc = self.test(model, self.x_test_data, self.y_test_data, pmf_muti_test) # go to Config.test
        print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')

        # 將tensor型態轉成list
        pmf_vec = [i.detach().cpu().numpy() for i in pmf_vec]
        aspect_vec = [i.detach().cpu().numpy() for i in aspect_vec]

        # 輸出
        return pmf_vec, aspect_vec