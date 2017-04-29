import numpy as np
import os
from hmmlearn import hmm
import warnings
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn import metrics

# for printing confusion matrix
def print_confusion(conf_arr):
	ticks=np.linspace(0, 94,num=95)
	# conf_arr = conf_arr[:,:,0]
	plt.imshow(conf_arr, interpolation='none',cmap="jet")
	plt.colorbar()
	plt.xticks(ticks,fontsize=6)
	plt.yticks(ticks,fontsize=6)
	plt.grid(True)
	plt.show()


warnings.filterwarnings("ignore") # Some functions which are depricated as part of hmmlearn library. 
									# We can use them anyways. It is just a warning that they will be 
									# removed in the next version

numClasses = 95 # one class for one word
filenames = {} # maps every word in the data set to a uniwue integer
revfilenames = {} # for a given integer, it stores the word for which the word got maped into in 
					# the dictionary filenames

#tctodd directory has folders tctodd1, tctodd2, ..., tctodd8, test. Each folder contains 3 instances of each of every 
# 95 words. Hence every tctoddi folder has 285 files.
for dirname in os.listdir("tctodd"):
	if dirname == "test":
		continue
	for filename in os.listdir("tctodd/"+dirname):
		word = filename[:-6]
		filenames[word] = 1 # currently every word is mapped to one but this will be changed later

wholeDataForOneClass = [[] for i in range(95)]
numFrames = [[0 for j in range(0)] for i in range(95)]
correctClass = [0 for i in range(95)]

# the following two arrays are required for printing the confusion Matrix
predicted_res = []
test_res = []

it = 0
for key, val in filenames.items():
	# Here mapping occurs from file name to integers
	filenames[key] = it
	it += 1

for key, val in filenames.items():
	# Reverse Mapping
	revfilenames[val] = key

for key, val in filenames.items():

	for dirname in os.listdir("tctodd"):
		if dirname == "test":
			continue
		for filename in os.listdir("tctodd/"+dirname):
			word = filename[:-6]
			if key != word:
				continue
			else:
				with open("tctodd/"+dirname+"/"+filename) as f:
					lineno = 0
					for line in f:
						dims = line.split("\t")
						dims = list(map(float, dims))
						wholeDataForOneClass[val].append(dims)
						lineno += 1
					numFrames[val].append(lineno)

n_components = 5 # Denotes the number of states in each HMM
print("Number of states in each HMM = ", n_components)
hmms = [hmm.GaussianHMM(n_components) for i in range(95)] # 95 HMMs one for each class
for i in range(95):
	hmms[i].fit(wholeDataForOneClass[i], numFrames[i]) # training HMMs

numCorrect = 0
numTotal = 0

# Loop for testing
for key, val in filenames.items():
	for sample in range(3):
		with open("tctodd/test/"+key+"-"+str(sample+1)+".tsd") as testC:
		#with open("tctodd/test/responsible-1.tsd") as testC:
			frames = []
			for line in testC:
				dims = line.split("\t")
				dims = list(map(float, dims))
				frames.append(dims)

		scores = []
		for i in range(95):
			j = hmms[i].score(frames) # Returns the score of HMM for a given observation. Higher score means
										# higher likelihood. Hence higher score is desirable if the model 
										# accurately classifies an observation sequence
			scores.append(j)

		# finding to which class each test data point is classified into and hence calculationg occuracy
		maxScore = scores[94]
		testScore = 0
		maxClass = revfilenames[94]

		for i in range(94):
			if revfilenames[i] == key:
				testScore = scores[i]
			if scores[i] > maxScore:
				maxScore = scores[i]
				maxClass = revfilenames[i]

		test_res.append(val)
		predicted_res.append(filenames[maxClass])
		if key == maxClass: 
			numCorrect += 1
			correctClass[val] += 1

		numTotal += 1

print_confusion(metrics.confusion_matrix(test_res,predicted_res)) #  for printing confusion Matrix

print("Number of correctly classified instances = ", numCorrect)
print("Number of total instances = ", numTotal)
print("Accuracy = ", (numCorrect*100.0)/numTotal)
