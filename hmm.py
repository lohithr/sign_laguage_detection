import numpy as np
import os
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
			else:
				data.pop()

		# for o in data : #print(o)
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

class HMM:
	def __init__(self, num_states, trainingData):
		self.num_states = num_states
		self.trainingData = np.array(trainingData)
		self.counter1 = 0

		#Initialise pi
		self.pi = np.zeros(num_states)
		self.pi.fill(1.0/num_states)

		# Initialise transition probabilty
		# Here we used bakins transitions
		# The allowed transitions are self loop, transition to next state and next next state (Atmost two distance)
		self.a = np.zeros((num_states, num_states))
		self.initialise_bakins()

		# Initialise emission probability distribution of states
		# We are using multivariate guassian with sigle mixture. In future can extend to multiple mixtures
		# Each state has a mean vector and covariance matrix
		self.covarianceMatrices = [None]*num_states
		self.means = [None]*num_states
		self.initialise_emission()

	def gaussian_pdf(self, sequence, mean, cov):
		n = mean.shape[0]
		pdfs = []
		det = np.linalg.det(cov)
		cov_inv = np.linalg.inv(cov)
		# #print("len of seq : ", len(sequence))
		for x in sequence:
			x_mean = x - mean
			# #print(x_mean.shape)
			exponent = -0.5*np.dot(np.dot(x_mean, cov_inv), x_mean.T)
			# #print(cov_inv.shape)
			# #print(exponent.shape)
			self.counter1+=1
			# #print(self.counter1, end=" ")
			# #print(exponent)

			if exponent<-1e4:
				pdfs.append(1e-4)
			else:
				pdf = (1/(((2*np.pi)**(n/2.0))*(det**0.5)))*np.exp(exponent)
				pdfs.append(pdf)

		pdfs = np.array(pdfs)
		# #print(pdfs)
		return pdfs

	# Function to initialise transition probality of bakins model
	def initialise_bakins(self):
		for i in range(self.num_states):
			for j in range(self.num_states):
				if i-j==0 or i-j==1 or i-j==2:
					self.a[i][j]=1
		self.a = self.a / self.a.sum(axis=1, keepdims=True)

	# Function to initialise emission probability distribution
	def initialise_emission(self):
		dim = len(self.trainingData[0][0])
		for i in range(self.num_states):
			self.covarianceMatrices[i] = np.diag(np.ones(dim))
		for i in range(self.num_states):
			self.means[i] = np.zeros(dim)

		# Mean initialisation
		T = len(self.trainingData)
		for t in range(T):
			num_frames = len(self.trainingData[t])
			frames_per_state = int(num_frames/self.num_states)
			j=0
			k=0
			temp_mean = np.zeros(dim)
			for i in range(num_frames):
				if j==frames_per_state:
					self.means[k] += temp_mean/(frames_per_state*1.0)
					k+=1
					temp_mean = np.zeros(dim)
					j=0
				temp_mean +=self.trainingData[t][i]
				j+=1

		for i in range(self.num_states):
			self.means[i] = self.means[i] / (T*1.0)
			#print(self.means[i])
		#print("asdfg\n")


	def iteration(self, observation_sequence):
		# #print("Iterate ...")
		num_steps = len(observation_sequence)
		self.alpha = np.zeros((num_steps, self.num_states))
		self.b = np.zeros((self.num_states, num_steps))
		for i in range(self.num_states):
			# #print(self.covarianceMatrices[i])
			# #print(np.linalg.eigvals(self.covarianceMatrices[i]))
			# if (self.covarianceMatrices[i].transpose() == self.covarianceMatrices[i]).all :
			# 	#print("symmetric")
			# else:
			# 	#print("assymetric")
			self.b[i] = self.gaussian_pdf(observation_sequence, self.means[i], self.covarianceMatrices[i])
		# #print(self.b[:,0])
		# #print(self.pi)

		# #print(self.b[:,0])
		self.alpha[0] = np.multiply(self.pi, self.b[:,0])
		# #print(self.alpha[0].sum())
		self.alpha[0] = self.alpha[0] / self.alpha[0].sum()
		for t in range(1, num_steps):
			self.alpha[t] = np.dot(self.alpha[t-1], self.a)
			self.alpha[t] = np.multiply(self.b[:, t], self.alpha[t])
			self.alpha[t] = self.alpha[t] / self.alpha[t].sum()

		self.beta = np.zeros((num_steps, self.num_states))
		self.beta[num_steps-1] = np.ones(self.num_states)
		self.beta[num_steps-1] = self.beta[num_steps-1] / self.beta[num_steps-1].sum()
		for t in range(num_steps-2, 0, -1):
			self.beta[t] = np.dot(np.multiply(self.beta[t+1], self.b[:, t]), self.a)
			self.beta[t] = self.beta[t] / self.beta[t].sum()

		self.zeta = np.zeros((num_steps, self.num_states, self.num_states))
		for t in range(num_steps-1):
			for i in range(self.num_states):
				for j in range(self.num_states):
					self.zeta[t][i][j] = self.alpha[t][i]*self.a[i][j]*self.beta[t+1][j]*self.b[j][t+1]

		normalizationArray = np.sum(self.zeta, axis = 2)
		normalizationArray = np.sum(normalizationArray, axis = 1)
		for t in range(num_steps-1):
			for i in range(self.num_states):
				for j in range(self.num_states):
					self.zeta[t][i][j] /= normalizationArray[t]		

		self.gamma = np.sum(self.zeta, axis = 2)
		self.gamma[num_steps-1] = self.alpha[num_steps-1]
		# #print(self.gamma)

		self.pi = self.gamma[0]
		self.a = np.sum(self.zeta, axis = 0)
		normalizationArray = np.sum(self.zeta, axis = 2)
		normalizationArray = np.sum(normalizationArray, axis = 0)
		self.a = self.a.T / normalizationArray
		self.means = np.dot(self.gamma.T, observation_sequence)
		normalizationArray = np.sum(self.gamma, axis = 0)
		for i in range(self.num_states):
			self.means[i] = self.means[i] / normalizationArray[i]
			# #print(self.means[i])
		# #print("cfgyui,m bh\n")

		num_dim = self.covarianceMatrices[0].shape[0]
		for i in range(self.num_states):
			num = np.zeros((num_dim, num_dim))
			den = 0
			for t in range(num_steps):
				num += self.gamma[t][i]*(observation_sequence[t] - self.means[i])*((observation_sequence[t] - self.means[i]).T)
				den += self.gamma[t][i]
			identity_matrix = np.zeros(num_dim)
			identity_matrix.fill(1e-12)
			identity_matrix = np.diag(identity_matrix)

			self.covarianceMatrices[i] = num / den + identity_matrix


	def train(self, num_iter = 10):
		for i in range(num_iter):
			for j in range(len(self.trainingData)):	
				self.iteration(self.trainingData[j])

	def predict(self, observation_sequence):
		num_steps = observation_sequence.shape[0]
		self.alpha = np.zeros((num_steps, self.num_states))
		self.c = np.array(num_steps)
		self.alpha[0] = np.multiply(self.pi, self.b[:,0])
		self.c[0] = np.sum(self.alpha[0])
		self.alpha[0] = self.alpha[0] / self.alpha[0].sum()
		for t in range(1, num_steps):
			self.alpha[t] = np.dot(self.alpha[t-1], self.a)
			self.alpha[t] = np.multiply(self.b[:, t], self.alpha[t])
			self.c[t] = np.sum(self.alpha[t])
			self.alpha[t] = self.alpha[t] / self.alpha[t].sum()

		probability = 1
		for t in range(num_steps):
			probability *- self.c[t]

		return probability


# hmm = HMM(5, train_list[0])
# hmm.train()

numClasses = 95
hmms = [HMM(5, train_list[i]) for i in range(numClasses)]


# hmms contain 95 HMMs one for each Class
# for each hmm:
# 	say we are currently dealing with hmm of "alive"
# 	there are 24 observation sequences for alive.
# 	for each observation sequence of alive 
# 		call hmm.train(observation sequence, number of iterations)

for index in range(len(hmms)):
	hmms[index].train()

scores = [0.0 for i in range(len(hmms))]
prediction_scores = [[0 for j in len(test_list[0])] for i in len(test_list)]
prediction_labels = [[0 for j in len(test_list[0])] for i in len(test_list)]
for i in len(test_list):
	for j in len(test_list[i]):
		max_hmm_score = float('-inf')
		max_label = 0
		for i in len(hmms):
			score = hmm.predict(test_list[i][j])
			if max_hmm_score > score:
				max_hmm_score = score
				max_label = i
		prediction_scores[i][j] = max_hmm_score
		prediction_labels[i][j] = max_label

numTotal = 0
numCorrect = 0
for key, val in mymap.items():
	numTotal += 3
	if prediction_labels[key][0] == val:
		numCorrect += 1
	if prediction_labels[key][1] == val:
		numCorrect += 1
	if prediction_labels[key][2] == val:
		numCorrect += 1

# #print numCorrect
# #print numTotal
# #print numCorrect*1.0/numTotal
# # it = 0
# # for key, val in filenames.items():
# # 	filenames[key] = it
# # 	it += 1

# # for key, val in filenames.items():
# # 	revfilenames[val] = key

# # for key, val in filenames.items():

# # 	for dirname in os.listdir("tctodd"):
# # 		if dirname == "test":
# # 			continue
# # 		for filename in os.listdir("tctodd/"+dirname):
# # 			word = filename[:-6]
# # 			if key != word:
# # 				continue
# # 			else:
# # 				with open("tctodd/"+dirname+"/"+filename) as f:
# # 					lineno = 0
# # 					for line in f:
# # 						dims = line.split("\t")
# # 						dims = map(float, dims)
# # 						wholeDataForOneClass[val].append(dims)
# # 						lineno += 1
# # 					numFrames[val].append(lineno)
