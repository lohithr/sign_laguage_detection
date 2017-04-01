import numpy as np 
#import pandas as pd 
import glob
mymap =  {}
train = []
res = []
data_folder = "tctodd"
data_sets = glob.glob(data_folder+"/*")
#train_fcn =  #change this fraction to change the %test_data
traind_len = len(data_sets)
counter = 0
val = 0
for i in range(traind_len):
	obs_files = glob.glob(data_sets[i]+"/*")
	for j in range(len(obs_files)):
		data =  open(obs_files[j]).read().split()
		temp = obs_files[j]
		obs_files[j] = obs_files[j].replace(data_sets[i]+"/","")
		for j1 in range(3):
			obs_files[j] = obs_files[j].replace("-"+str(j1+1)+".tsd","")
		print temp,obs_files[j]
		if obs_files[j] not in mymap:
			mymap[obs_files[j]] = counter
			counter = counter +1
		val = max(val,len(data))
		train.append(data)
		res.append(mymap[obs_files[j]])

for i in range(len(train)):
	print (train[i],res[i])
print val


		
			


		

