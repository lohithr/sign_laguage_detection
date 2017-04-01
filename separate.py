import random
import subprocess
import glob

basedir = "tctodd"

words = []

allfiles = glob.glob(basedir+"1/*")
for file in allfiles:
	file = file.replace(basedir+"1/","")
	for j1 in range(3):
		file = file.replace("-"+str(j1+1)+".tsd","")
	words.append(file)
words = set(words)
words = list(words)
words = sorted(words,key=lambda s: s.lower())

# words contains all the words we have in the data
# print(words)

for x in range(95):
	rand_list = random.sample(range(1,10),6)
	for y in range(6):
		rand_dir = basedir + str(rand_list[y])
		# print(rand_dir)
		# get file name
		filename = words[x] + "-1.tsd"
		source_path = rand_dir + "/" + filename
		dest_path = "test/" + filename + "_" + str(y)
		# mvcommand = "/bin/mv "+source_path + " " + dest_path
		subprocess.Popen(["/bin/mv",source_path,dest_path])