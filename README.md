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
Dataset	Classes	Train samples	Test samples
AG’s News	4	120 000	7 600
Sogou News	5	450 000	60 000
DBPedia	14	560 000	70 000
Yelp Review Polarity	2	560 000	38 000
Yelp Review Full	5	650 000	50 000
Yahoo! Answers	10	1 400 000	60 000
Amazon Review Full	5	3 000 000	650 000
Amazon Review Polarity	2	3 600 000	400 000
