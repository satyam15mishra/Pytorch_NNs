import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import torch
from torch import nn

data = pd.read_csv('studentdata.csv')
X = data.iloc[:,[1,2,3]]
y = data.iloc[:, 0]

def plot_data(data):
	X_plot = np.array(data[['gre', 'gpa']])
	y_plot = np.array(data['admit'])
	admitted = X_plot[np.argwhere(y == 1)]
	rejected = X_plot[np.argwhere(y == 0)]
	plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'red', edgecolor = 'k')
	plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'cyan', edgecolor = 'k')
	plt.xlabel("GRE")
	plt.ylabel("CGPA")
	plt.show()

#one hot encoding the data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories = 'auto'), [-1])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
X = np.array(ct.fit_transform(X), dtype=np.float)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42)

#scaling the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#converting into numpy array and then into pytorch tensors
X_train = torch.from_numpy(np.asarray(X_train)) 
X_test = torch.from_numpy(np.asarray(X_test))
y_train = torch.from_numpy(np.asarray(y_train))
y_test = torch.from_numpy(np.asarray(y_test))
X_train = X_train.float()
X_test = X_test.float()
y_train = y_train.float()
y_test = y_test.float()

# network architecture
class MyNet(nn.Module):

	def __init__(self):
		super(MyNet, self).__init__()
		self.fc1 = torch.nn.Linear(6, 32)
		self.fc2 =  torch.nn.Linear(32, 16)
		self.fc3 = torch.nn.Linear(16, 8)
		self.fc4 = torch.nn.Linear(8, 1)

	def forward(self, x):
		x = torch.sigmoid(self.fc1(x))
		x = torch.sigmoid(self.fc2(x))
		x = torch.sigmoid(self.fc3(x))
		x = torch.sigmoid(self.fc4(x))
		return x

network = MyNet()
out = network(X_train)

import torch.optim as optim
optimizer = optim.SGD(MyNet().parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
optimizer.zero_grad()   # zero the gradient buffers
loss = criterion(out, y_train)

#backpropogation to minimize loss
loss.backward()
optimizer.step()

print 'Mean squared error or loss = ', loss
