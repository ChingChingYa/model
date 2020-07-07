# hcan-pytorch
hierarchical convolutional attention networks for text classification
(paper https://www.aclweb.org/anthology/W18-3002/)
hcan程式參考:https://github.com/yongqyu/hcan-pytorch


# How to use my code
With my code, you can:

Train your model with any dataset
Given either my trained model or yours, you could evaluate any test dataset whose have the same set of classes

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

| Dataset                 |資料筆數|
|-------------------------|:----------:|
| Musical Instruments     |   10240   |
| Patio Lawn and Garden   |   13248   |
| Automotive              |   20416   |
| Amazon Instant Video    |   37120   |
| Office Products         |   53248   |
| Digital Music           |   64704   |
