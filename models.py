import torch
import torch.nn as nn
import numpy
import torch.nn.functional as F
# import torch.nn.utils as utils
# import torch.nn.init as init
from layers import ConvMultiheadSelfAttWord as CMSA_Word
from layers import ConvMultiheadTargetAttnWord as CMTA_Word
from layers import ConvMultiheadSelfAttSent as CMSA_Sent
from layers import ConvMultiheadTargetAttSent as CMTA_Sent


class Proto(nn.Module):
    def __init__(self, num_emb, input_dim, pretrained_weight):
        super(Proto, self).__init__()
        # num_emb   單詞數量
        # input_dim   維度300
        self.id2vec = nn.Embedding(num_emb, input_dim, padding_idx=1)
        # 讀取詞表中的Embedding
        self.id2vec.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.id2vec.requires_grad = False

    
    # ==================accuracy========================
    # 計算評分的準確率
    # rating_vec:為模型的輸出文檔的評分預測
    # rating_batch:評分測試資料

    # return acc:hcan模型準確率
    # ==================================================
    def accuracy(self, rating_vec, rating_batch):
        predict = [torch.argmax(i, 1) for i in rating_vec] #獲得hcan所預測之評分
        equals = [torch.eq(p, l) for p, l in zip(predict, rating_batch)] #判斷預測與真實評分是否一樣
        acc = torch.sum(torch.FloatTensor(equals)) 
        return acc


    # ==================loss========================
    # 計算hcan的loss
    # sentiment_batch:模型輸出uv哈達瑪乘積
    # rating_batch:真實評分資料
    # pmf_vec:為模型的輸出aspect情感向量
    # rating_vec:為模型的輸出文檔的評分預測

    # return acc:評分與哈達瑪乘積Loss加總
    # ==================================================
    def loss(self, sentiment_batch, rating_batch, pmf_vec, rating_vec):
        loss_fn = torch.nn.MSELoss(reduction='mean')
        entroy = nn.CrossEntropyLoss()
        loss_sentiment = 0
        loss_rating = 0


        # 計算UV loss (MSE)
        for x, y in zip(pmf_vec, sentiment_batch):
            loss_sentiment += loss_fn(x, y)


        # 計算評分 loss (CrossEntropy)
        for x, y in zip(rating_vec, rating_batch):
            a = entroy(x, y)
            loss_rating += torch.mean(a)
        loss_all = loss_sentiment + loss_rating
        return loss_all


    # ======================hcan forward================== 
    # input_batch: 文本評論資料

    # return:
    # pmf_vec:為模型的輸出aspect情感向量
    # rating_vec:為模型的輸出文檔的評分預測
    # aspect_vec:為模型的輸出aspect特徵向量
    # ======================================================
    def forward(self, input_batch):
        pmf_vec, rating_vec, aspect_vec = self.predict(input_batch) # go to Proto_CNN.predict
        return pmf_vec, rating_vec, aspect_vec



# ==================hcan 模型架構========================
# input_dim: 輸入維度
# hidden_dim: 隱藏層大小
# kernel_dim: fliter大小
# sent_maxlen: 最大句子長度
# word_maxlen: 最大單詞數量
# num_classes: #latent factor數量
# dropout_rate: dropout數值
# num_emb: 詞表單詞數量
# pretrained_weight: 單詞預訓練值
# ==================================================
class Proto_CNN(Proto):
    def __init__(self, input_dim, hidden_dim, kernel_dim,
                 sent_maxlen, word_maxlen, num_classes,
                 dropout_rate, num_emb, pretrained_weight):
        super(Proto_CNN, self).__init__(num_emb, input_dim, pretrained_weight)
        
        # 加入單詞位置向量
        self.positions_word = nn.Parameter(torch.randn(sent_maxlen, word_maxlen, input_dim))
        stdv = 1. / self.positions_word.size(1) ** 0.5
        self.positions_word.data.uniform_(-stdv, stdv)

        # 加入句子位置向量
        self.positions_sent = nn.Parameter(torch.randn(sent_maxlen, input_dim))
        stdv = 1. / self.positions_sent.size(1) ** 0.5
        self.positions_sent.data.uniform_(-stdv, stdv)

        # 單詞階層
        self.cmsa_word = CMSA_Word(input_dim, kernel_dim[0])
        self.cmta_word = CMTA_Word(input_dim, kernel_dim[1])
        
         # 句子階層
        self.cmsa_sent = CMSA_Sent(input_dim, kernel_dim[0])
        self.cmta_sent = CMTA_Sent(input_dim, kernel_dim[1])

        self.dropout = nn.Dropout(dropout_rate)
        self.cls1 = nn.Linear(input_dim, 1)
        self.cls2 = nn.Linear(input_dim, 5)
        self.cls3 = nn.Linear(3, 1)
        # nn.Linear是用于设置网络中的全连接层的
        nn.init.xavier_normal_(self.cls1.weight)
        nn.init.xavier_normal_(self.cls2.weight)
        nn.init.xavier_normal_(self.cls3.weight)



    # ==================predict=======================
    # x:WE文本評論資料

    # return 
    # pmf_vec:模型的輸出aspect情感向量
    # rating_vec:模型的輸出文檔的評分預測
    # aspect_vec:模型的輸出aspect特徵向量
    # ================================================
    def predict(self, x):
        input = self.id2vec(x)
        # input [batch_size, sent_maxlen, word_maxlen, embedding_size]
        
        # =====================word to Sent=========================
        # 加入單詞階層位置向量
        input = self.dropout(input + self.positions_word)
        new = torch.reshape(input, [-1, 34, 300])
        # new [batch_size*sent_maxlen, word_maxlen, embedding_size]
        
        # 放入單詞階層卷積Mutihead self-Attention
        hidden = self.cmsa_word(new.permute(0, 2, 1)) # go to layers.py-ConvMultiheadSelfAttWord.forward
        # hidden [batch_size*sent_maxlen, embedding_size, word_maxlen]

        # 放入單詞階層卷積Mutihead Target-Attention
        hidden = self.cmta_word(hidden) # go to layers.py-ConvMultiheadTargetAttnWord.forward
        # hidden [batch_size*sent_maxlen, embedding_size, 1]

        sent_vec = torch.reshape(hidden.squeeze(-1), [-1, 10, 300])
        # squeeze降维
        # sent_vec [batch_size, sent_maxlen, embedding_size]


        # =====================Sent to doc=========================
        # 加入句子階層位置向量
        sent_vec = self.dropout(sent_vec + self.positions_sent)
        # sent_vec [batch_size, sent_maxlen, embedding_size]

        # 放入句子階層卷積Mutihead self-Attention
        hidden = self.cmsa_sent(sent_vec.permute(0, 2, 1)) # go to layers.py-ConvMultiheadSelfAttSent.forward
        # hidden [batch_size, embedding_size, sent_maxlen]

        # 放入句子階層卷積Mutihead Target-Attention
        sentiment_vec, aspect_vec = self.cmta_sent(hidden) # go to layers.py-ConvMultiheadTargetAttSent.forward
        # sentiment_vec [batch_size, embedding_size, latent factor數量]
        # aspect_vec [latent factor數量 , embedding_size]
        sentiment_vec = torch.abs(sentiment_vec)

        sentiment_vec1 = self.cls1(sentiment_vec.permute(0, 2, 1)) 
        # sentiment_vec1 [batch_size, latent factor數量, 1]
        logits = (sentiment_vec1.squeeze(2))
        # logits [batch_size, latent factor數量]
        # squeeze去掉维数为1的的维度

        sentiment_vec2 = self.cls2(sentiment_vec.permute(0, 2, 1)) 
        # sentiment_vec2 [batch_size, latent factor數量, rating數量(1~5)5]
        rating_vec = (self.cls3(sentiment_vec2.permute(0, 2, 1))).squeeze(-1)
        # rating_vec [batch_size, 5]

        pmf_vec = logits
        rating_vec = [F.softmax(i, 0) for i in rating_vec] 
        rating_vec = [i.unsqueeze(0) for i in rating_vec] #文檔評分預測
    
        

        return pmf_vec, rating_vec, aspect_vec
 

