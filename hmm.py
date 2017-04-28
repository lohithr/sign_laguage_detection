import numpy as np
import os
from hmmlearn import hmm
import warnings
from scipy.stats import multivariate_normal
import glob

mymap =  {}
train_list = []
test_list = []
train = []
test = []
res = []
test_res = []
class_string_list = []
data_folder = "tctodd"
data_sets = glob.glob(data_folder+"/*")

traind_len = len(data_sets)
counter = 0
val = 0
for i in range(traind_len):
	
	obs_files = glob.glob(data_sets[i]+"/*")
	for j in range(len(obs_files)):
		data =  open(obs_files[j]).read().split('\n')
		for i1 in range(len(data)):
			l = data[i1].split()
			if len(l) != 0:
				data[i1] = list(map(float, l))

		# for o in data : print(o)
		obs_files[j] = obs_files[j].replace(data_sets[i]+"/","")
		for j1 in range(6):
			obs_files[j] = obs_files[j].replace("-"+str(j1+1)+".tsd","")
			obs_files[j] = obs_files[j].replace("_"+str(j1),"")
		if obs_files[j] not in mymap:
			mymap[obs_files[j]] = counter
			train_list.append([])
			test_list.append([])
			counter = counter +1
		val = max(val,len(data))
		if "tctodd" in data_sets[i]:
			train.append(data)
			train_list[mymap[obs_files[j]]].append(data)
			res.append(mymap[obs_files[j]])
		if "test" in data_sets[i]:
			test.append(data)
			test_list[mymap[obs_files[j]]].append(data)
			test_res.append(mymap[obs_files[j]])


# print(train[0])

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

	def train(self,observation_sequence, num_iter = 10):
		for i in range(num_iter): self.iteration(observation_sequence)


numClasses = 95
hmms = [HMM for i in range(numClasses)]

# hmms contain 95 HMMs one for each Class
# for each hmm:
# 	say we are currently dealing with hmm of "alive"
# 	there are 24 observation sequences for alive.
# 	for each observation sequence of alive 
# 		call hmm.train(observation sequence, number of iterations)

num_itr = 10
class_string_list = list(mymap.keys())
for index in range(len(hmms)):
	for observation in train_list[index]:
		hmms[index].train(observation,num_itr)

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
