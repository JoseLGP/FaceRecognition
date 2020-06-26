import os
import math
import shutil
import random

def picture_extraction(pictures_to_extract, destination_path, source_path):
	old_dir = os.getcwd()
	os.chdir(source_path)
	people = os.listdir(os.getcwd())
	people.sort()
	#print(people)
	#print(pictures_to_extract)
	#print(os.getcwd())
	#print(len(pictures_to_extract))

	for picture in pictures_to_extract:
		person = picture[0]
		img_num = picture[1]
		if len(img_num) > 1:
			directory = person + "/"
			filename = str(os.getcwd()) + "/" + person + "/" + person + "_00" + img_num + ".jpg"
			filename_destiny = destination_path + "/" + person + "/" + person + "_000" + img_num + ".jpg"
		else:
			directory = person + "/"
			filename = str(os.getcwd()) + "/" + person + "/" + person + "_000" + img_num + ".jpg"
			filename_destiny = destination_path + "/" + person + "/" + person + "_000" + img_num + ".jpg"

		#print("DIRECTORY: " + directory)
		#print("destination_path: " + destination_path)
		#print("FILENAME: " + filename)


		if not os.path.exists(destination_path + "/" + directory):
			os.mkdir(destination_path + "/" + directory)
			shutil.copy2(filename, filename_destiny)
		else:
			shutil.rmtree(destination_path + "/" + directory)
			os.mkdir(destination_path + "/" + directory)
			shutil.copy2(filename, filename_destiny)

	os.chdir(old_dir)

	return


def who_will_be_in_the_dataset(total_samples, num_train, num_test, img_path, train_list, test_list):
	total_train = 4038
	total_test = 1711

	who_train = [random.randint(0, total_samples) for _ in range(num_train)]
	#who_test = [random.randint(0, total_samples) for _ in range(num_test)]
	who_test = list(who_train)
	who_train.sort()
	print(who_test)
	#who_test.sort()
	#print(who_train)
	#print(who_test)

	people = os.listdir(img_path)
	people.sort()
	#print(people)
	indexes = [i for i in range(total_samples)]
	#print(indexes)
	#print(people[indexes[0]])
	people_with_index = [[people[i], indexes[i]] for i in range(total_samples)]


	trainy = list()
	for index in who_train:
		trainy.append(people[index])
	# trainy = ["Ben_Afflek", "Ricardo_Milos", ... , "Kiko_Loureiro"]

	trainX = list()
	pictures_to_extract = list()
	line_counter = 0

	with open(train_list) as file:
		for line in file:
			if line_counter == 0:
				line_counter += 1
				continue
			else:
				line_counter += 1
				line = line.strip().split("\t") # line[0] = "Ricardo_Milos" - line[1] = "2" (the pic number to extract from it's folder)
				if line[0] in trainy:
					pictures_to_extract.append(line)

	#print(pictures_to_extract) [["Ricardo_Milos", "2"], ["Ozzy_Osbourne", "3"]]
	#print(len(pictures_to_extract)),

	destination_path = "/home/jose/Documents/IPD441/Proyecto/imgs/lfw_jostel/train"
	picture_extraction(pictures_to_extract, destination_path, img_path)

	testy = list()
	for index in who_test:
		testy.append(people[index])
	# trainy = ["Ben_Afflek", "Ricardo_Milos", ... , "Kiko_Loureiro"]

	testX = list()
	pictures_to_extract = list()
	line_counter = 0

	with open(test_list) as file:
		for line in file:
			if line_counter == 0:
				line_counter += 1
				continue
			else:
				line_counter += 1
				line = line.strip().split("\t") # line[0] = "Ricardo_Milos" - line[1] = "2" (the pic number to extract from it's folder)
				if line[0] in testy:
					pictures_to_extract.append(line)

	#print(pictures_to_extract) [["Ricardo_Milos", "2"], ["Ozzy_Osbourne", "3"]]
	#print(len(pictures_to_extract)),


	destination_path = "/home/jose/Documents/IPD441/Proyecto/imgs/lfw_jostel/test"
	picture_extraction(pictures_to_extract, destination_path, img_path)

	return


def main():
	path = "imgs/lfw/"
	train_data = "peopleDevTrain.txt"
	test_data = "peopleDevTest.txt"
	how_many_train = 500
	how_many_test = 100
	total_people = 5749

	who_will_be_in_the_dataset(total_people, how_many_train, how_many_test, path, train_data, test_data)

def main2():
	#print(os.getcwd())
	#with open("customLFWpeople.txt", "w") as file:
	imgs_path = "/home/jose/Documents/IPD441/Proyecto/imgs/custom_datasets/lfw_jostel_2/train"
	people = os.listdir(imgs_path)
	train = 0.8
	test = 0.2
	for person in people: # person = "Keanu_Reeves"
		imgs_path = "/home/jose/Documents/IPD441/Proyecto/imgs/custom_datasets/lfw_jostel_2/train"
		how_many_photos = len(os.listdir(imgs_path + "/" + person))
		how_many_train = math.floor(how_many_photos * train)
		how_many_test = how_many_photos - how_many_train

		train_index = [(i+1) for i in range(how_many_train)]
		test_index = [(train_index[-1] + i+1 ) for i in range(how_many_test)]
		print(train_index)
		print(test_index)
		train_files = list()

		for j in train_index:
			#print(j)
			if (len(str(j))) > 1:
				filename = imgs_path + "/" + person + "/" + person + "_00" + str(j) + ".jpg"
			else:
				filename = imgs_path + "/" + person + "/" + person + "_000" + str(j) + ".jpg"
			print(filename)
			train_files.append(filename)

		# train_files = ["/home/jose/.../Keanu_Reeves_0001.jpg"]
		print("TRAIN FILES")
		print(train_files)

		# Clean train folder
		dir_to_clean = imgs_path + "/" + person
		dir_to_clean_files = os.listdir(dir_to_clean)
		#print(dir_to_clean_files)		
		dir_to_clean_files = [(dir_to_clean + "/" + file) for file in dir_to_clean_files]
		print("DIR TO CLEAN FILES - TRAIN")
		print(dir_to_clean_files)
		#print(train_files)
		for file in dir_to_clean_files:
			#filename = dir_to_clean + "/" + file
			print("FILE - TRAIN")
			print(file)
			#print(train_files[0])
			if file not in train_files:
				try:
					os.remove(file)
				except:
					print("Already removed. Continue ...")
					continue


		imgs_path = "/home/jose/Documents/IPD441/Proyecto/imgs/custom_datasets/lfw_jostel_2/val"
		test_files = list()
		for j in test_index:
			if (len(str(j))) > 1:
				filename = imgs_path + "/" + person + "/" + person + "_00" + str(j) + ".jpg"
			else:
				filename = imgs_path + "/" + person + "/" + person + "_000" + str(j) + ".jpg"
			print(filename)
			test_files.append(filename)

		print("TEST FILES")
		print(test_files)

		# test_files = ["/home/jose/.../Keanu_Reeves_0010.jpg"]

		# Clean test folder
		dir_to_clean = imgs_path + "/" + person
		dir_to_clean_files = os.listdir(dir_to_clean)		
		dir_to_clean_files = [(dir_to_clean + "/" + file) for file in dir_to_clean_files]
		print("DIR_TO_CLEAN_FILES")
		print(dir_to_clean_files)
		#print(train_files)
		for file in dir_to_clean_files:
			#filename = dir_to_clean + "/" + file
			#print(filename)
			#print(train_files[0])
			if file not in test_files:
				print("TEST - file")
				print(file)
				try:
					print("Removing: " + str(file))
					os.remove(file)
				except:
					print("Already removed. Continue ...")
					continue


	print(people)
	return

if __name__ == "__main__":
	#main()
	main2()