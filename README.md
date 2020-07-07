# hcan-pytorch
hierarchical convolutional attention networks for text classification


```
from models import Proto_CNN as HCAN

model = HCAN(input_dim, hidden_dim, kernel_dim,
             sent_maxlen, dropout_rate, num_emb, pretrained_weight)
```
if you don't have ```pretrained_weight```, you should modify the Class ```Proto``` to ```pretrained_weight``` optional.

```
logtis = model(x, None)
```
second variable is length of ```x```, ```l```. However, since this variable is no longer needed, insert ```None```.   
```x```is index of words.

參考:https://github.com/yongqyu/hcan-pytorch



官方下载地址，http://mattmahoney.NET/dc/text8.zip

torch==1.3.1 torchvision==0.4.2 -f https://download.pytorch.org/whl/torch_stable.html

https://pypi.org/project/torchsummary/#files

使用GPU安裝教學 https://blog.csdn.net/tahir_111/article/details/84767650

STEP1: CUDA Toolkit 9.0 Downloads https://developer.nvidia.com/cuda-90-download-archive

STEP2: cuDNN Archive https://developer.nvidia.com/rdp/cudnn-archive

STEP3: conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

STEP4: 測試 nvidia-smi import torch torch.cuda.is_available()