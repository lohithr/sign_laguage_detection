import numpy as np 
import scipy.signal		
#import pandas as pd 
import glob
mymap =  {}
train = []
test = []
res = []
test_res = []
data_folder = "tctodd"
data_sets = glob.glob(data_folder+"/*")
#print data_sets
#train_fcn =  #change this fraction to change the %test_data
traind_len = len(data_sets)
counter = 0
val = 0
for i in range(traind_len):
	obs_files = glob.glob(data_sets[i]+"/*")
	for j in range(len(obs_files)):
		file =  open(obs_files[j])
		#converting attribute data to a 2D array
		data = [[float(n) for n in line.split()] for line in file]
		#resampling the data to sample 57 frames which is an average no of frames for all words
		data = scipy.signal.resample(data,57)
		#converting the list to an arry ang flattering  to form a 1d arry of attribute values 
		data = np.array(data)
		data = np.ndarray.flatten(data)
		#extracting the word from the file name 
		obs_files[j] = obs_files[j].replace(data_sets[i]+"/","")
		for j1 in range(6):
			obs_files[j] = obs_files[j].replace("-"+str(j1+1)+".tsd","")
			obs_files[j] = obs_files[j].replace("_"+str(j1),"")
		#Each word is mapped to a distinct integer to represent a class
		if obs_files[j] not in mymap:
			mymap[obs_files[j]] = counter
			counter = counter +1
		# tcdodd folders has data  to train and test folder has test data so based on file location appending file data into 
		# corresponding arrays
		if "tctodd" in data_sets[i]:
			train.append(data)
			res.append(mymap[obs_files[j]])
		if "test" in data_sets[i]:
			test.append(data)
			test_res.append(mymap[obs_files[j]])

########################################################
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
#iris = datasets.load_iris()
X, y = train,res
clf = SGDClassifier("log","l2")
print clf.fit(X, y).score(test,test_res)

			


		

