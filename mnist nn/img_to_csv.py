import pandas as pd 
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError

train_data_path = 'train/Images/train/'
test_data_path = 'train/Images/test/'

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def preprocess_dataset(i):
	img = mpimg.imread(test_data_path + str(i + 49000) + '.png')     
	gray = rgb2gray(img) 
	b = gray.reshape(1, 784)
	print "Iteration", i
	return b

pd.concat([pd.DataFrame(preprocess_dataset(i)) for i in range(21000)]).to_csv('mnist_test_array.csv')
