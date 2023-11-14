# -*- coding: utf-8 -*-
# @Author: xiegr
# @Date:   2020-08-30 15:58:51
# @Last Modified by:   xiegr
# @Last Modified time: 2020-09-18 14:22:48
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math


torch.manual_seed(2020)
torch.cuda.manual_seed_all(2020)
np.random.seed(2020)
random.seed(2020)
torch.backends.cudnn.deterministic = True

class SelfAttention(nn.Module):
	"""docstring for SelfAttention"""
	def __init__(self, d_dim=256, dropout=0.1):
		super(SelfAttention, self).__init__()
		# for query, key, value, output
		self.dim = d_dim
		self.linears = nn.ModuleList([nn.Linear(d_dim, d_dim) for _ in range(4)])
		self.dropout = nn.Dropout(p=dropout)

	def attention(self, query, key, value):
		scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim)
		scores = F.softmax(scores, dim=-1)
		return scores

	def forward(self, query, key, value):
		# 1) query, key, value
		query, key, value = \
		[l(x) for l, x in zip(self.linears, (query, key, value))]

		# 2) Apply attention
		scores = self.attention(query, key, value)
		x = torch.matmul(scores, value)

		# 3) apply the final linear
		x = self.linears[-1](x.contiguous())
		# sum keepdim=False
		return self.dropout(x), torch.mean(scores, dim=-2)

class OneDimCNN(nn.Module):
	"""docstring for OneDimCNN"""
	# https://blog.csdn.net/sunny_xsc1994/article/details/82969867
	def __init__(self, max_byte_len, d_dim=256, \
		kernel_size = [3, 4], filters=256, dropout=0.1):
		super(OneDimCNN, self).__init__()
		self.kernel_size = kernel_size
		self.convs = nn.ModuleList([
						nn.Sequential(nn.Conv1d(in_channels=d_dim, 
												out_channels=filters, 
												kernel_size=h),
						#nn.BatchNorm1d(num_features=config.feature_size), 
						nn.ReLU(),
						# MaxPool1d: 
						# stride – the stride of the window. Default value is kernel_size
						nn.MaxPool1d(kernel_size=max_byte_len-h+1))
						for h in self.kernel_size
						]
						)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x):
		out = [conv(x.transpose(-2,-1)) for conv in self.convs]
		out = torch.cat(out, dim=1)
		out = out.view(-1, out.size(1))
		return self.dropout(out)


class SAM(nn.Module):
	"""docstring for SAM"""
	# total header bytes 24
	def __init__(self, num_class, max_byte_len, kernel_size = [3, 4], \
		d_dim=256, dropout=0.1, filters=256):
		super(SAM, self).__init__()
		self.posembedding = nn.Embedding(num_embeddings=max_byte_len, 
								embedding_dim=d_dim)
		self.byteembedding = nn.Embedding(num_embeddings=300, 
								embedding_dim=d_dim)
		self.attention = SelfAttention(d_dim, dropout)
		self.cnn = OneDimCNN(max_byte_len, d_dim, kernel_size, filters, dropout)
		self.fc = nn.Linear(in_features=256*len(kernel_size),
                            out_features=num_class)

	def forward(self, x, y):
		out = self.byteembedding(x) + self.posembedding(y)
		out, score = self.attention(out, out, out)
		out = self.cnn(out)
		out = self.fc(out)
		if not self.training:
			return F.softmax(out, dim=-1).max(1)[1], score
		return out

class FSSAM(nn.Module):
	"""docstring for SAM"""
	# total header bytes 24
	def __init__(self, num_class, max_byte_len, kernel_size = [3, 4], \
		d_dim=256, dropout=0.1, filters=512):
		super(FSSAM, self).__init__()
		self.posembedding = nn.Embedding(num_embeddings=max_byte_len*8, 
								embedding_dim=d_dim)
		self.byteembedding = nn.Embedding(num_embeddings=300, 
								embedding_dim=d_dim)
		self.attention = SelfAttention(d_dim, dropout)
		self.cnn = OneDimCNN(max_byte_len*8, d_dim, kernel_size, filters, dropout)
		self.fc = nn.Linear(in_features=512*len(kernel_size),
                            out_features=num_class)

	def forward(self, x, y):
		out = self.byteembedding(x) + self.posembedding(y)
		out, score = self.attention(out, out, out)
		out = self.cnn(out)
		out = self.fc(out)
		if not self.training:
			return F.softmax(out, dim=-1).max(1)[1], score
		return out


class OneCNN(nn.Module):
    def __init__(self,label_num):
        super(OneCNN,self).__init__()
        self.layer_1 = nn.Sequential(
            # 输入784*1
            nn.Conv2d(1,32,(1,25),1,padding='same'),
            nn.ReLU(),
            # 输出262*32
            nn.MaxPool2d((1, 3), 3, padding=0),
        )
        self.layer_2 = nn.Sequential(
            # 输入261*32
            nn.Conv2d(32,64,(1,25),1,padding='same'),
            nn.ReLU(),
            # 输入261*64
            nn.MaxPool2d((1, 3), 3, padding=0)
        )
        self.fc1=nn.Sequential(
            # 输入88*64
            nn.Flatten(),
            nn.Linear(87*64,1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024,label_num),
            nn.Dropout(p=0.3)
        )
    def forward(self,x):
        #print("x.shape:",x.shape)
        x=self.layer_1(x)
        #print("x.shape:",x.shape)
        x=self.layer_2(x)
        #print("x.shape:",x.shape)
        x=self.fc1(x)
        #print("x.shape:",x.shape)
        if not self.training:
            return F.softmax(x, dim=-1).max(1)[1]
        return x

class DatanetMLP(nn.Module):
    def __init__(self,label_num):
        super(DatanetMLP, self).__init__()
    
        self.fc1 = nn.Linear(in_features=784, out_features=128) 
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(in_features=32, out_features= label_num)
        self.dropout = nn.Dropout(0.3)
        self.f1 = nn.Flatten()
    def forward(self,x):
        #print("x.shape:",x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        x = self.f1(x)
        #print("x.shape:",x.shape)
        if not self.training:
        	return F.softmax(x, dim=-1).max(1)[1]
        return x


class DatanetCNN(nn.Module):
    def __init__(self,label_num):
        super(DatanetCNN,self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,              # 灰度图片的高度为1，input height
                out_channels=8,            # 16个卷积，之后高度为从1变成16，长宽不变，n_filters
                kernel_size=5,              # 5*5宽度的卷积，filter size
                stride=1,                   # 步幅为1，filter movement/step
                padding=2,                  # 周围填充2圈0，if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # 激活时，图片长宽高不变，activation
            nn.MaxPool2d(kernel_size=2),    # 4合1的池化，之后图片的高度不变，长宽减半，choose max value in 2x2 
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(8, 16, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),    
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.fc1=nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*3*3,1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024,label_num),
            nn.Dropout(p=0.3)
        )
    def forward(self,x):
        #print("x.shape:",x.shape)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        #print("x.shape:",x.shape)
        x = self.fc1(x)
        if not self.training:
        	return F.softmax(x, dim=-1).max(1)[1]
        return x


class BiLSTM(nn.Module):
    def __init__(self,label_num):
        super(BiLSTM, self).__init__()
        self.hidden_size = 128
        self.lstm = nn.LSTM(784, self.hidden_size, 2, bidirectional=True, batch_first=True)
        self.w = nn.Parameter(torch.zeros(self.hidden_size * 2))
        self.fc1  = nn.Linear(self.hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, label_num)
    def forward(self,x):
        #print("x.shape:",x.shape)
        H, _ = self.lstm(x)  
        #print('H.size is : ',H.shape)
        alpha = F.softmax(torch.matmul(H, self.w), dim=1).unsqueeze(-1)  
        out = H * alpha  
        out = torch.sum(out, 1)
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc2(out)  
        if not self.training:
        	return F.softmax(out, dim=-1).max(1)[1]
        return out


class DeepPacket(nn.Module):
    def __init__(self,label_num):
        super(DeepPacket, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=200,kernel_size=4,stride=3)
        self.conv2 = nn.Conv1d(in_channels=200,out_channels=200,kernel_size=5,stride=1)
        
        self.fc1 = nn.Linear(in_features=200*128, out_features=200) # ((28-5+1)/2 -5 +1)/2 = 4
        self.dropout = nn.Dropout(0.05)
        self.fc2 = nn.Linear(in_features=200, out_features=100)
        self.dropout = nn.Dropout(0.05)
        self.fc3 = nn.Linear(in_features=100, out_features=50)
        self.dropout = nn.Dropout(0.05)
        self.out = nn.Linear(in_features=50, out_features= label_num)
    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = F.max_pool1d(out, kernel_size=2)
        #print('out shape is:',out.shape)
        out = out.reshape(-1, 200*128) 
        
        out = self.fc1(out)
        out = self.dropout(out)
        out = F.relu(out)
        
        out = self.fc2(out)
        out = self.dropout(out)
        out = F.relu(out)

        out = self.fc3(out)
        out = self.dropout(out)
        out = F.relu(out)

        out = self.out(out)
        if not self.training:
        	return F.softmax(out, dim=-1).max(1)[1]
        return out


class TSCRNN(nn.Module):
    def __init__(self,label_num):
        super(TSCRNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels= 1, out_channels= 64 , kernel_size=3, stride=1, padding=1)
        #in channels 输入矩阵的行数
        self.bn1 = nn.BatchNorm1d(64,affine = True)
        self.conv2 = nn.Conv1d(in_channels= 64, out_channels= 64 , kernel_size=3, stride=1,  padding=1)
        self.bn2 = nn.BatchNorm1d(64,affine = True)
        self.lstm = nn.LSTM(196, 256, 2, bidirectional=True, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(in_features=512, out_features=label_num)
    def forward(self,x):
        #print("x.shape:",x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)

        x,_ = self.lstm(x)
        x = self.dropout(x)
        
        x = self.out(x[:, -1, :]) 
        #print("x.shape:",x.shape)
        if not self.training:
        	return F.softmax(x, dim=-1).max(1)[1]
        return x

		
if __name__ == '__main__':
	x = np.random.randint(0, 255, (10, 20))
	y = np.random.randint(0, 20, (10, 20))
	sam = SAM(num_class=5, max_byte_len=20)
	out = sam(torch.from_numpy(x).long(), torch.from_numpy(y).long())
	print(out[0])

	sam.eval()
	out, score = sam(torch.from_numpy(x).long(), torch.from_numpy(y).long())
	print(out[0], score[0])
