from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import scipy.signal
import numpy as np
import glob
import pandas

mymap =  {}
train = []
test = []
train_res = []
test_res = []
data_folder = "tctodd"
data_sets = glob.glob(data_folder+"/*")
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
			train_res.append(mymap[obs_files[j]])
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

##############################################################################
#################code after data processing###################################

np.random.seed(3)

encoder = LabelEncoder()
encoder.fit(train_res)
encoded_train_res = encoder.transform(train_res)
dummy_y = np_utils.to_categorical(encoded_train_res)

encoder1 = LabelEncoder()
encoder1.fit(test_res)
encoded_test_res = encoder1.transform(test_res)
dummy_test_y = np_utils.to_categorical(encoded_test_res)

#set our model with one hidden layer and one output layer
def baseline_model():
	model = Sequential()
	model.add(Dense(int(len(train[0])/10), input_dim=len(train[0]), kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(len(list(mymap.keys())), kernel_initializer='normal', activation='sigmoid'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=5, verbose=1)
kfold = KFold(n_splits=5, shuffle=True, random_state=3)


results = cross_val_score(estimator, train, dummy_y,cv = kfold)
test_results = cross_val_score(estimator, test, dummy_test_y)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print("Baseline for test: %.2f%% (%.2f%%)" % (test_results.mean()*100, test_results.std()*100))
