import torchvision
from torchvision import transforms
import pandas as pd 
from torch import nn 
import torch
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt

data = pd.read_csv('housedata.csv', delimiter = '\t')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

class MyNet(nn.Module):
	
	def __init__(self):
		super(MyNet, self).__init__()
		self.fc1 = nn.Linear(8, 4)
		self.fc2 = nn.Linear(4, 2)
		self.fc3 = nn.Linear(2, 1)
	
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		return x

train_batch = np.array_split(X_train, 50)
label_batch = np.array_split(y_train, 50)

for i in range(len(train_batch)):
    train_batch[i] = torch.from_numpy(train_batch[i].values).float()
for i in range(len(label_batch)):
    label_batch[i] = torch.from_numpy(label_batch[i].values).float().view(-1, 1)

X_test = torch.from_numpy(X_test.values).float()
y_test = torch.from_numpy(y_test.values).float().view(-1, 1)

model = MyNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 300

train_losses, test_losses = list(), list()

for e in range(epochs):
	model.train()
	train_loss = 0

	for i in range(len(train_batch)):
		optimizer.zero_grad()
		output = model(train_batch[i])
		loss = torch.sqrt(criterion(torch.log(output), torch.log(label_batch[i])))
		loss.backward()
		optimizer.step()
		train_loss += loss.item()

	else:
		test_loss = 0
		accuracy = 0

		with torch.no_grad():

			model.eval()
			predictions = model(X_test)
			test_loss += torch.sqrt(criterion(torch.log(predictions), torch.log(y_test)))

		train_losses.append(train_loss/len(train_batch))
		test_losses.append(test_losses)

		print 'Epoch:', e, '\tTraining Loss:', (train_loss/len(train_batch)), '\tTest Loss:', test_loss
 

plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label ='Validation Loss')
plt.show()