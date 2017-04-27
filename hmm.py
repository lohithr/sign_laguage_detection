import numpy as np
import os
from hmmlearn import hmm
import warnings
from scipy.stats import multivariate_normal

warnings.filterwarnings("ignore", category=DeprecationWarning)

class HMM:
	def __init__(self, num_states):
		self.num_states = num_states
		self.pi = np.array(num_states)
		self.pi.fill(1.0/num_states)

		self.a = np.zeros((num_states, num_states))
		self.intialise_bakins()



	def initialise_bakins(self):
		for i in range(num_states):
			for j in range(num_states):
				if i-j==0 or i-j==1 or i-j==2:
					self.a[i][j]=1
		self.a = self.a / self.a.sum(axis=1, keepdims=True)

	def initialise_emission(self, num_dim):
		self.covarianceMatrices = np.array(self.num_states)
		for i in range(self.num_states):
			self.covarianceMatrices[i] = np.diag(np.random.rand(num_dim))
		self.means = np.array(self.num_states)
		for i in range(self.num_states):
			self.means[i] = np.random.rand(num_dim)
		####### DO K MEANS #######

	def iteration(self, observation_sequence):
		num_steps = observation_sequence.shape(0)
		self.alpha = np.zeros((num_steps, num_states))
		self.b = np.zeros((num_states, num_steps))
		for i in range(num_states):
			self.b[i] = multivariate_normal.pdf(observation_sequence, mean = self.means[i],
												 cov = self.covarianceMatrices[i])

		self.alpha[0] = np.multiply(self.pi, self.b[:,0])
		self.alpha[0] = self.alpha[0] / self.alpha[0].sum(axis=1, keepdims=True)
		for t in range(1, num_steps):
			self.alpha[t] = np.dot(self.alpha[t-1], self.a)
			self.alpha[t] = np.multiply(self.b[:, t], self.alpha[t])
			self.alpha[t] = self.alpha[t] / self.alpha[t].sum(axis=1, keepdims=True)

		self.beta = np.zeros((num_steps, num_states))
		self.beta[num_steps-1] = np.ones(num_states)
		self.beta[num_steps-1] = self.beta[num_steps-1] / self.beta[num_steps-1].sum(axis=1, keepdims=True)
		for t in range(num_steps-2, 0, -1):
			self.beta[t] = np.dot(np.multiply(self.beta[t+1], self.b[:, t]), self.a)
			self.beta[t] = self.beta[t] / self.beta[t].sum(axis=1, keepdims=True)

		self.zeta = np.zeros((num_steps-1, num_states, num_states))
		for t in range(num_steps-1):
			for i in range(num_states):
				for j in range(num_states):
					self.zeta[t][i][j] = alpha[t]*a[i][j]*beta[t+1][j]*b[j][t+1]

		self.gamma = np.sum(zeta, axis = 2)

		self.pi = self.gamma[0]
		self.a = np.sum(zeta, axis = 0)
		normalizationArray = np.sum(zeta, axis = 2)
		normalizationArray = np.sum(normalizationArray, axis = 0)
		self.a = self.a.T / normalizationArray
		self.means = np.dot(self.gamma.T, observation_sequence)
		normalizationArray = np.sum(self.gamma, axis = 0)
		self.means = self.means / normalizationArray

		num_dim = self.covarianceMatrices.shape(0)
		for i in range(num_dim):
			num = np.zeros((num_dim, num_dim))
			den = 0
			for t in range(num_steps):
				num += self.gamma[t][i]*(observation_sequence[t] - self.means[i])*((observation_sequence[t] - self.means[i]).T)
				den += self.gamma[t][i]
			self.covarianceMatrices[i] = num / den


numClasses = 95
filenames = {}
revfilenames = {}
for dirname in os.listdir("tctodd"):
	# print dirname
	if dirname == "test":
		continue
	for filename in os.listdir("tctodd/"+dirname):
		# print "\t"+filename
		word = filename[:-6]
		filenames[word] = 1

wholeDataForOneClass = [[] for i in range(95)]
numFrames = [[0 for j in range(0)] for i in range(95)]

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
						dims = map(float, dims)
						wholeDataForOneClass[val].append(dims)
						lineno += 1
					numFrames[val].append(lineno)

print "Division by zero after this .."

hmms = [hmm.GaussianHMM(n_components = 4) for i in range(95)]
for i in range(95):
	hmms[i].fit(wholeDataForOneClass[i], numFrames[i])

print "Division by zero ends .."

numCorrect = 0
numTotal = 0
for key, val in filenames.items():
	for sample in range(3):
		with open("tctodd/test/"+key+"-"+str(sample+1)+".tsd") as testC:
		#with open("tctodd/test/responsible-1.tsd") as testC:
			frames = []
			for line in testC:
				dims = line.split("\t")
				dims = map(float, dims)
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

		# print maxClass+" : "+str(maxScore)
		# print key+" : "+str(testScore)

		if key == maxClass: numCorrect += 1
		numTotal += 1

print numCorrect
print numTotal
print (numCorrect*1.0)/numTotal

# testKey = "responsible"
# with open("tctodd/test/"+testKey+"-1.tsd") as testC:
# 	for line in testC:
# 		dims = line.split("\t")
# 		dims = map(float, dims)
# 		frames.append(dims)

# 	scores = []
# 	for i in range(95):
# 		j = hmms[i].score(frames)
# 		scores.append(j)

# 	maxScore = scores[94]
# 	testScore = 0
# 	maxClass = revfilenames[94]

# 	for i in range(94):
# 		if revfilenames[i] == testKey:
# 			testScore = scores[i]
# 		if scores[i] > maxScore:
# 			maxScore = scores[i]
# 			maxClass = revfilenames[i]

# 	print maxClass+" : "+str(maxScore)
# 	print testKey+" : "+str(testScore)

# 	if testKey == maxClass: numCorrect += 1
# 	numTotal += 1
