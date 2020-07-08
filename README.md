# hcan-pytorch
hierarchical convolutional attention networks for text classification

(paper https://www.aclweb.org/anthology/W18-3002/)

hcan程式參考:https://github.com/yongqyu/hcan-pytorch


# Requirements:
python==3.7

pandas==0.25.1

nltk==3.4.5

gensim==3.8.1

nltk

sklearn

tf-nightly==2.2.0.dev20200130

arrow==0.15.5

torch==1.3.1 torchvision==0.4.2 -f https://download.pytorch.org/whl/torch_stable.html

https://pypi.org/project/torchsummary/#files

# Datasets:
Amazon product data(http://jmcauley.ucsd.edu/data/amazon/)

train 80% / valid 10% / test  10% 
| Dataset                 |資料筆數|
|-------------------------|:----------:|
| Musical Instruments     |   10240   |
| Patio Lawn and Garden   |   13248   |
| Automotive              |   20416   |
| Amazon Instant Video    |   37120   |
| Office Products         |   53248   |
| Digital Music           |   64704   |



# How to use my code
### STEP1-資料前處理+訓練詞庫(Word2Vec)
訓練詞庫(刪除詞頻小於5)、輸出詞表、計算文件長度、分割文件，刪除停用詞

最終獲得
**商品評分檔**(用戶ID/商品ID/整體評分)

**文字評論檔**(用戶評論文本/整體評分)

```
python pretrain.py
``` 
### STEP2-Training

```
python hcan_train.py
``` 
### STEP2-Aspect提取
```
python lda.py
python similar_by_vector.py
``` 



