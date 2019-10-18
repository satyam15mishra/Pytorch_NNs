import numpy as np 
import pandas as pd 
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

#DATA PREPROCESSING
train_data = pd.read_csv('mnist_train_array.csv')
test_data = pd.read_csv('mnist_test_array.csv')
train_labels_data = pd.read_csv('train/train.csv')

train_labels = train_labels_data['label']
train_data['label'] = train_labels
new_train_data = train_data.iloc[:, 1:]
new_test_data = test_data.iloc[:, 1:]

test_X = (new_test_data.iloc[:, :784])
train_X = np.array(new_train_data.iloc[:, :784]).reshape(-1, 1, 28, 28) #PIXEL
#train_X = torch.from_numpy(train_X)
train_y = np.array(new_train_data.iloc[:, -1]) #LABEL
#train_y = torch.from_numpy(train_y)

train_batch = np.array_split(train_X, 50)
label_batch = np.array_split(train_y, 50)

for i in range(len(train_batch)):
    train_batch[i] = torch.from_numpy(train_batch[i]).float() #pixels
for i in range(len(label_batch)):
    label_batch[i] = torch.from_numpy(label_batch[i]).view(-1, 1) #corresponding labels


#BUILDING THE NEURAL NETWORK
class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(28*28, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, 64)
		self.fc4 = nn.Linear(64, 10)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return  F.log_softmax(x, dim = 1)

net = Net()
#feeding a random input
X_sample = torch.rand((28, 28))
X_sample = X_sample.view(-1, 28*28)
print output = net(X_sample)

#TRAINING THE NETWORK
optimizer = optim.Adam(net.parameters(), lr = 0.001)
EPOCHS = 5

for epoch in range(EPOCHS):
	for data in train_batch:
		X, y = train_batch, label_batch
		net.zero_grad()
		output = net(X.view(-1, 28*28))
		#this loss because the label is just a scalar otherwise we would've used MSE
		loss = F.nll_loss(output, y)
		loss.backward() #backprop
		optimizer.step() #adjust the weights for us
	print loss
