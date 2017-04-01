import random
import subprocess

basedir = "tctodd"


for x in range(95):
	rand_list = random.sample(range(1,10),6)
	for y in range(6):
		rand_dir = basedir + str(rand_list[y])
		print(rand_dir)
		# get file name
		filename = "alive-1.tsd"
		source_path = rand_dir + "/" + filename
		dest_path = "test/" + filename + "_" + str(y)
		# mvcommand = "/bin/mv "+source_path + " " + dest_path
		subprocess.Popen(["/bin/mv",source_path,dest_path])