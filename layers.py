import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import copy

# =====================Conv Multihead Attention=================
# 初始化
# input_dim:輸入維度
# kernel_dim:filter大小
# multihead_cnt:multihead投影次數
# conv_cnt:cnn數量
#  ==============================================================
class ConvMultiheadAttention(Module):
    
    def __init__(self, input_dim, kernel_dim, multihead_cnt, conv_cnt):
        super(ConvMultiheadAttention, self).__init__()
        self.input_dim = input_dim
        self.multihead_cnt = multihead_cnt

        self.convs = nn.ModuleList([nn.Conv1d(input_dim, input_dim, kernel_dim, stride=1,padding=1)
                                    for _ in range(conv_cnt)])
        for w in self.convs:
            nn.init.xavier_normal_(w.weight)


    # =====================self attention=================
    # 加入self-attention Attention(Q, K, V )= softmax(QKT√d) = attention score
    # softmax(QKT√d)V = attention vector
    # bmm(矩阵乘法)
    # div(除法)  
    # permute(将tensor的维度换位置)

    # return: attention vector
    # ==============================================================
    def attention(self, q, k, v):
        attention_score = torch.softmax(torch.div(
                torch.bmm(q.permute(0, 2, 1), k), np.sqrt(self.input_dim)),
                2)
        attention_vector = attention_score.bmm(v.permute(0, 2, 1)).permute(0, 2, 1)     
        return attention_vector


    # =====================conv multihead self attention=================
    # 加入 MultiHead(Q, K, V ) = [head1, ..., headh]where headi = Attention(Qi, Ki, Vi)
    # hiddens:attention vector
    # multihead_cnt 投影次數
    # chunk 分割
    # cat 串接在一起
    
    # return:hiddens(multihead後的值)
    # ==============================================================
    def multihead(self, hiddens):
        hiddens = [torch.chunk(hidden, self.multihead_cnt, 1)
                   for hidden in hiddens]
        hiddens = torch.cat([self.attention(hiddens[0][i], hiddens[1][i],
                                            hiddens[2][i])
                            for i in range(self.multihead_cnt)], 1)

        return hiddens


   
   
   
# ConvMultiheadSelfAttWord單詞階層
class ConvMultiheadSelfAttWord(ConvMultiheadAttention):
    def __init__(self, input_dim, kernel_dim, multihead_cnt=10, conv_cnt=6):
        super(ConvMultiheadSelfAttWord, self).\
              __init__(input_dim, kernel_dim, multihead_cnt, conv_cnt)

    # =====================Conv Multihead Self AttWord & Elementwise Multiply 單詞階層=================
    # 兩個做哈達瑪乘積 Parallel(E) = MultiHead(Qa, Ka, Va) MultiHead(Qb, Kb, Vb)
    # input:文本評論(Embedding後)

    # return:output(Elementwise Multiply後正規化結果)
    # =============================================================================================
    def forward(self, input):
        hiddens = [F.elu(conv(input)) for conv in self.convs[:-1]] 
        hiddens.append(torch.tanh(self.convs[-1](input)))
        
        elu_hid = self.multihead(hiddens[:3]) # [:3]  擷取前3的字串計算conv multihead self attention
        tanh_hid = self.multihead(hiddens[3:]) # [3:]  擷取後3的字串計算conv multihead self attention
        # F.layer_norm  layer正規化
        # mul   哈達瑪乘積
        output = F.layer_norm(torch.mul(elu_hid, tanh_hid), elu_hid.size()[1:])
        # shape为[batch *word_maxlen, embedding_size, sent_maxlen]
        return output


# ConvMultiheadTargetAttentionWord單詞階層
class ConvMultiheadTargetAttnWord(ConvMultiheadAttention):
    def __init__(self, input_dim, kernel_dim, multihead_cnt=10, conv_cnt=6):
        super(ConvMultiheadTargetAttnWord, self).\
              __init__(input_dim, kernel_dim, multihead_cnt, conv_cnt)
        self.target = nn.Parameter(torch.randn(input_dim, 1)) # 初始化句子target
        stdv = 1. / math.sqrt(self.target.size(1))
        self.target.data.uniform_(-stdv, stdv)


    # =====================Conv Multihead Target Attention 單詞階層=================
    # 加入target計算multihead
    # input:Elementwise Multiply後正規化結果

    # return:sent_vec 單詞階層輸出為句子向量
    # ============================================================================
    def forward(self, input):
        batch_size = input.size(0)
        hiddens = [F.elu(conv(input)) for conv in self.convs]
        sent_vec = self.multihead([self.target.expand
                                (batch_size, self.input_dim, 1)]+hiddens)
        # shape为[batch_size, embedding_size, 1]
        return sent_vec




# ConvMultiheadSelfAttSent句子階層
class ConvMultiheadSelfAttSent(ConvMultiheadAttention):
    def __init__(self, input_dim, kernel_dim, multihead_cnt=10, conv_cnt=6):
        super(ConvMultiheadSelfAttSent, self).\
              __init__(input_dim, kernel_dim, multihead_cnt, conv_cnt)


    # =====================Conv Multihead Self AttSent & Elementwise Multiply 句子階層================
    # 兩個做哈達瑪乘積 Parallel(E) = MultiHead(Qa, Ka, Va) MultiHead(Qb, Kb, Vb)
    # input:文本評論(Embedding後)

    # return:output (Elementwise Multiply後正規化結果)
    # =============================================================================================
    def forward(self, input):
        hiddens = [F.elu(conv(input)) for conv in self.convs[:-1]]
        hiddens.append(torch.tanh(self.convs[-1](input)))

        elu_hid = self.multihead(hiddens[:3]) # [:3]  擷取前3的字串計算conv multihead self attention
        tanh_hid = self.multihead(hiddens[3:]) # [3:]  擷取後3的字串計算conv multihead self attention
        # F.layer_norm  layer正規化
        # mul   哈達瑪乘積
        output = F.layer_norm(torch.mul(elu_hid, tanh_hid), elu_hid.size()[1:])
         # shape为[batch, embedding_size, sent_maxlen]
        return output


# MultiheadTargetAttentionSent句子階層
class ConvMultiheadTargetAttSent(ConvMultiheadAttention):
    def __init__(self, input_dim, kernel_dim, multihead_cnt=10, conv_cnt=6):
        super(ConvMultiheadTargetAttSent, self).\
              __init__(input_dim, kernel_dim, multihead_cnt, conv_cnt)
        self.target = nn.Parameter(torch.randn(input_dim, 3))# 初始化文檔target
        stdv = 1. / math.sqrt(self.target.size(1))
        self.target.data.uniform_(-stdv, stdv)


    # =====================Conv Multihead Target AttentionSent 句子階層=================
    # 加入target計算multihead
    # input:Elementwise Multiply後正規化結果

    # return:
    # sentiment_vec:aspect情感向量
    # aspect_vec:aspect特徵向量
    # ==================================================================================
    def forward(self, input):
        batch_size = input.size(0)
        hiddens = [F.elu(conv(input)) for conv in self.convs]
        sentiment_vec = self.multihead([self.target.expand(batch_size, self.input_dim, 3)]+hiddens)
       
        aspect_vec = self.target.permute(1, 0)
    
        # shape为[batch_size, embedding_size, 1]
        return sentiment_vec, aspect_vec
