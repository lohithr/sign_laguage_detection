import numpy as np 
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
		data =  open(obs_files[j]).read().split()
		data = [float(l) for l in data]
		#temp = obs_files[j]
		obs_files[j] = obs_files[j].replace(data_sets[i]+"/","")
		for j1 in range(6):
			obs_files[j] = obs_files[j].replace("-"+str(j1+1)+".tsd","")
			obs_files[j] = obs_files[j].replace("_"+str(j1),"")
		#print temp,obs_files[j]
		if obs_files[j] not in mymap:
			mymap[obs_files[j]] = counter
			counter = counter +1
		val = max(val,len(data))
		if "tctodd" in data_sets[i]:
			train.append(data)
			res.append(mymap[obs_files[j]])
		if "test" in data_sets[i]:
			test.append(data)
			test_res.append(mymap[obs_files[j]])

i = 0
for i in range(len(train)):
	j=0
	for j in range(val-len(train[i])):
		train[i].append(float(0))
i = 0
for i in range(len(test)):
	j=0
	for j in range(val-len(test[i])):
		test[i].append(float(0))
	#print train[i]


########################################################
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
#iris = datasets.load_iris()
X, y = train,res
clf = SGDClassifier("log","l2")
print clf.fit(X, y).score(test,test_res)

			


		

