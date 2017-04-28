import numpy as np
import scipy.signal
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import scipy.signal
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


def print_confusion(conf_arr):
	legends = [i for i in range(95)]
	df_cm = pd.DataFrame(conf_arr, index = legends, columns = legends)
	plt.figure(figsize = (10,7))
	sn.set(font_scale=1.4)
	sn.heatmap(df_cm, annot=True, annot_kws={"size":16})
	sn.plt.show()

########################################################
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import metrics

X, y = train,res
clf = SGDClassifier("log","l2")
model = clf.fit(X, y)
print(model.score(test,test_res)*100)
predicted_res = model.predict(test)
print_confusion(metrics.confusion_matrix(test_res,predicted_res))

#########################################################

model = SVC()
model.fit(train,res)

predicted_res = model.predict(test)
print(metrics.classification_report(test_res,predicted_res))
# print(metrics.confusion_matrix(test_res,predicted_res)[0])
print_confusion(metrics.confusion_matrix(test_res,predicted_res))



