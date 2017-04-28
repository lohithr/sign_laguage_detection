import numpy as np
import os
from hmmlearn import hmm
import warnings
from scipy.stats import multivariate_normal

warnings.filterwarnings("ignore") # Some functions which are depricated as part of hmmlearn library. 
									# We can use them anyways. It is just a warning that they will be 
									# removed in the next version

numClasses = 95
filenames = {}
revfilenames = {}
for dirname in os.listdir("tctodd"):
	if dirname == "test":
		continue
	for filename in os.listdir("tctodd/"+dirname):
		word = filename[:-6]
		filenames[word] = 1

wholeDataForOneClass = [[] for i in range(95)]
numFrames = [[0 for j in range(0)] for i in range(95)]
correctClass = [0 for i in range(95)]

it = 0
for key, val in filenames.items():
	filenames[key] = it
	it += 1

for key, val in filenames.items():
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

hmms = [hmm.GaussianHMM(n_components = 5) for i in range(95)]
for i in range(95):
	hmms[i].fit(wholeDataForOneClass[i], numFrames[i])

numCorrect = 0
numTotal = 0
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
			j = hmms[i].score(frames)
			scores.append(j)

		maxScore = scores[94]
		testScore = 0
		maxClass = revfilenames[94]

		for i in range(94):
			if revfilenames[i] == key:
				testScore = scores[i]
			if scores[i] > maxScore:
				maxScore = scores[i]
				maxClass = revfilenames[i]

		if key == maxClass: 
			numCorrect += 1
			correctClass[val] += 1
		else:

		numTotal += 1

print("Number of correctly classified instances = ", numCorrect)
print("Number of total instances = ", numTotal)
print("Accuracy = ", (numCorrect*100.0)/numTotal)