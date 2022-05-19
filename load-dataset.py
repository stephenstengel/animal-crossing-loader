#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  load-dataset.py
#  
#  2022 Stephen Stengel <stephen.stengel@cwu.edu>
#  
#  Script to load the dataset into file form for easy importing.
#  

print("Loading imports...")

import os
import platform
import random
import skimage
from skimage.io import imsave
from skimage.util import img_as_uint
import shutil
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm #Pretty loading bars

import numpy as np
import tensorflow as tf

print("Done!")

CLASS_BOBCAT = 0
CLASS_COYOTE = 1
CLASS_DEER = 2
CLASS_ELK = 3
CLASS_HUMAN = 4
CLASS_NOT_INTERESTING = 5
CLASS_RACCOON = 6
CLASS_WEASEL = 7

CLASS_BOBCAT_STRING = "bobcat"
CLASS_COYOTE_STRING = "coyote"
CLASS_DEER_STRING = "deer"
CLASS_ELK_STRING = "elk"
CLASS_HUMAN_STRING = "human"
CLASS_RACCOON_STRING = "raccoon"
CLASS_WEASEL_STRING = "weasel"

# ~ CLASS_INTERESTING = 0
# ~ CLASS_NOT_INTERESTING = 1

CLASS_INTERESTING_STRING = "interesting"
CLASS_NOT_INTERESTING_STRING = "not"

DATASET_DIRECTORY = os.path.normpath("./ftp.wsdot.wa.gov/public/I90Snoq/Biology/thermal/614s/")
INTERESTING_DIRECTORY = os.path.join(DATASET_DIRECTORY, "interesting")
INTERESTING_COYOTE_DIRECTORY = os.path.join(INTERESTING_DIRECTORY, "CALA - coyote")
INTERESTING_ELK_DIRECTORY = os.path.join(INTERESTING_DIRECTORY, "CEEL - elk")
INTERESTING_HUMAN_DIRECTORY = os.path.join(INTERESTING_DIRECTORY, "HOSA - human")
INTERESTING_BOBCAT_DIRECTORY = os.path.join(INTERESTING_DIRECTORY, "LYRU - bobcat")
INTERESTING_DEER_DIRECTORY = os.path.join(INTERESTING_DIRECTORY, "ODHE - deer")
INTERESTING_RACCOON_DIRECTORY = os.path.join(INTERESTING_DIRECTORY, "PRLO - raccoon")
INTERESTING_WEASEL_DIRECTORY = os.path.join(INTERESTING_DIRECTORY, "WEAS - weasel")
NOT_INTERESTING_DIRECTORY = os.path.join(DATASET_DIRECTORY, "not interesting")

DATASET_COPY_FOLDER = os.path.normpath("./tmpdata/")
# DATASET_COPY_FOLDER_INT = os.path.join(DATASET_COPY_FOLDER, CLASS_INTERESTING_STRING)
DATASET_COPY_FOLDER_COYOTE = os.path.join(DATASET_COPY_FOLDER, CLASS_COYOTE_STRING)
DATASET_COPY_FOLDER_ELK = os.path.join(DATASET_COPY_FOLDER, CLASS_ELK_STRING)
DATASET_COPY_FOLDER_HUMAN = os.path.join(DATASET_COPY_FOLDER, CLASS_HUMAN_STRING)
DATASET_COPY_FOLDER_BOBCAT = os.path.join(DATASET_COPY_FOLDER, CLASS_BOBCAT_STRING)
DATASET_COPY_FOLDER_DEER = os.path.join(DATASET_COPY_FOLDER, CLASS_DEER_STRING)
DATASET_COPY_FOLDER_RACCOON = os.path.join(DATASET_COPY_FOLDER, CLASS_RACCOON_STRING)
DATASET_COPY_FOLDER_WEASEL = os.path.join(DATASET_COPY_FOLDER, CLASS_WEASEL_STRING)
DATASET_COPY_FOLDER_NOT = os.path.join(DATASET_COPY_FOLDER, CLASS_NOT_INTERESTING_STRING)

DATASET_PNG_FOLDER = os.path.normpath("./datasets-as-png/")
DATASET_PNG_FOLDER_TRAIN = os.path.join(DATASET_PNG_FOLDER, "train")
# DATASET_PNG_FOLDER_TRAIN_INT = os.path.join(DATASET_PNG_FOLDER_TRAIN, CLASS_INTERESTING_STRING)
DATASET_PNG_FOLDER_TRAIN_COYOTE = os.path.join(DATASET_PNG_FOLDER_TRAIN, CLASS_COYOTE_STRING)
DATASET_PNG_FOLDER_TRAIN_ELK = os.path.join(DATASET_PNG_FOLDER_TRAIN, CLASS_ELK_STRING)
DATASET_PNG_FOLDER_TRAIN_HUMAN = os.path.join(DATASET_PNG_FOLDER_TRAIN, CLASS_HUMAN_STRING)
DATASET_PNG_FOLDER_TRAIN_BOBCAT = os.path.join(DATASET_PNG_FOLDER_TRAIN, CLASS_BOBCAT_STRING)
DATASET_PNG_FOLDER_TRAIN_DEER = os.path.join(DATASET_PNG_FOLDER_TRAIN, CLASS_DEER_STRING)
DATASET_PNG_FOLDER_TRAIN_RACCOON = os.path.join(DATASET_PNG_FOLDER_TRAIN, CLASS_RACCOON_STRING)
DATASET_PNG_FOLDER_TRAIN_WEASEL = os.path.join(DATASET_PNG_FOLDER_TRAIN, CLASS_WEASEL_STRING)
DATASET_PNG_FOLDER_TRAIN_NOT = os.path.join(DATASET_PNG_FOLDER_TRAIN, CLASS_NOT_INTERESTING_STRING)

DATASET_PNG_FOLDER_VAL = os.path.join(DATASET_PNG_FOLDER, "val")
# DATASET_PNG_FOLDER_VAL_INT = os.path.join(DATASET_PNG_FOLDER_VAL, CLASS_INTERESTING_STRING)
DATASET_PNG_FOLDER_VAL_COYOTE = os.path.join(DATASET_PNG_FOLDER_VAL, CLASS_COYOTE_STRING)
DATASET_PNG_FOLDER_VAL_ELK = os.path.join(DATASET_PNG_FOLDER_VAL, CLASS_ELK_STRING)
DATASET_PNG_FOLDER_VAL_HUMAN = os.path.join(DATASET_PNG_FOLDER_VAL, CLASS_HUMAN_STRING)
DATASET_PNG_FOLDER_VAL_BOBCAT = os.path.join(DATASET_PNG_FOLDER_VAL, CLASS_BOBCAT_STRING)
DATASET_PNG_FOLDER_VAL_DEER = os.path.join(DATASET_PNG_FOLDER_VAL, CLASS_DEER_STRING)
DATASET_PNG_FOLDER_VAL_RACCOON = os.path.join(DATASET_PNG_FOLDER_VAL, CLASS_RACCOON_STRING)
DATASET_PNG_FOLDER_VAL_WEASEL = os.path.join(DATASET_PNG_FOLDER_VAL, CLASS_WEASEL_STRING)
DATASET_PNG_FOLDER_VAL_NOT = os.path.join(DATASET_PNG_FOLDER_VAL, CLASS_NOT_INTERESTING_STRING)

DATASET_PNG_FOLDER_TEST = os.path.join(DATASET_PNG_FOLDER, "test")
# DATASET_PNG_FOLDER_TEST_INT = os.path.join(DATASET_PNG_FOLDER_TEST, CLASS_INTERESTING_STRING)
DATASET_PNG_FOLDER_TEST_COYOTE = os.path.join(DATASET_PNG_FOLDER_TEST, CLASS_COYOTE_STRING)
DATASET_PNG_FOLDER_TEST_ELK = os.path.join(DATASET_PNG_FOLDER_TEST, CLASS_ELK_STRING)
DATASET_PNG_FOLDER_TEST_HUMAN = os.path.join(DATASET_PNG_FOLDER_TEST, CLASS_HUMAN_STRING)
DATASET_PNG_FOLDER_TEST_BOBCAT = os.path.join(DATASET_PNG_FOLDER_TEST, CLASS_BOBCAT_STRING)
DATASET_PNG_FOLDER_TEST_DEER = os.path.join(DATASET_PNG_FOLDER_TEST, CLASS_DEER_STRING)
DATASET_PNG_FOLDER_TEST_RACCOON = os.path.join(DATASET_PNG_FOLDER_TEST, CLASS_RACCOON_STRING)
DATASET_PNG_FOLDER_TEST_WEASEL = os.path.join(DATASET_PNG_FOLDER_TEST, CLASS_WEASEL_STRING)
DATASET_PNG_FOLDER_TEST_NOT = os.path.join(DATASET_PNG_FOLDER_TEST, CLASS_NOT_INTERESTING_STRING)

DATASET_SAVE_DIR = os.path.normpath("./dataset/")
TRAIN_SAVE_DIRECTORY = os.path.join(DATASET_SAVE_DIR, "train")
VAL_SAVE_DIRECTORY = os.path.join(DATASET_SAVE_DIR, "val")
TEST_SAVE_DIRECTORY = os.path.join(DATASET_SAVE_DIR, "test")

ALL_FOLDERS_LIST = [
		DATASET_DIRECTORY,
		INTERESTING_COYOTE_DIRECTORY,
		INTERESTING_ELK_DIRECTORY,
		INTERESTING_HUMAN_DIRECTORY,
		INTERESTING_BOBCAT_DIRECTORY,
		INTERESTING_DEER_DIRECTORY,
		INTERESTING_RACCOON_DIRECTORY,
		INTERESTING_WEASEL_DIRECTORY,
		NOT_INTERESTING_DIRECTORY,
		DATASET_COPY_FOLDER,
		DATASET_COPY_FOLDER_COYOTE,
		DATASET_COPY_FOLDER_ELK,
		DATASET_COPY_FOLDER_HUMAN,
		DATASET_COPY_FOLDER_BOBCAT,
		DATASET_COPY_FOLDER_DEER,
		DATASET_COPY_FOLDER_RACCOON,
		DATASET_COPY_FOLDER_WEASEL,
		DATASET_COPY_FOLDER_NOT,
		DATASET_PNG_FOLDER,
		DATASET_PNG_FOLDER_TRAIN,
		DATASET_PNG_FOLDER_TRAIN_COYOTE,
		DATASET_PNG_FOLDER_TRAIN_ELK,
		DATASET_PNG_FOLDER_TRAIN_HUMAN,
		DATASET_PNG_FOLDER_TRAIN_BOBCAT,
		DATASET_PNG_FOLDER_TRAIN_DEER,
		DATASET_PNG_FOLDER_TRAIN_RACCOON,
		DATASET_PNG_FOLDER_TRAIN_WEASEL,
		DATASET_PNG_FOLDER_TRAIN_NOT,
		DATASET_PNG_FOLDER_VAL,
		DATASET_PNG_FOLDER_VAL_COYOTE,
		DATASET_PNG_FOLDER_VAL_ELK,
		DATASET_PNG_FOLDER_VAL_HUMAN,
		DATASET_PNG_FOLDER_VAL_BOBCAT,
		DATASET_PNG_FOLDER_VAL_DEER,
		DATASET_PNG_FOLDER_VAL_RACCOON,
		DATASET_PNG_FOLDER_VAL_WEASEL,
		DATASET_PNG_FOLDER_VAL_NOT,
		DATASET_PNG_FOLDER_TEST,
		DATASET_PNG_FOLDER_TEST_COYOTE,
		DATASET_PNG_FOLDER_TEST_ELK,
		DATASET_PNG_FOLDER_TEST_HUMAN,
		DATASET_PNG_FOLDER_TEST_BOBCAT,
		DATASET_PNG_FOLDER_TEST_DEER,
		DATASET_PNG_FOLDER_TEST_RACCOON,
		DATASET_PNG_FOLDER_TEST_WEASEL,
		DATASET_PNG_FOLDER_TEST_NOT,
		DATASET_SAVE_DIR,
		TRAIN_SAVE_DIRECTORY,
		VAL_SAVE_DIRECTORY,
		TEST_SAVE_DIRECTORY
		]

HIDDEN_DOWNLOAD_FLAG_FILE = ".isnotfirstdownload"

CLASS_NAMES_LIST_INT = [CLASS_BOBCAT, CLASS_COYOTE, CLASS_DEER, CLASS_ELK, CLASS_HUMAN, CLASS_NOT_INTERESTING, CLASS_RACCOON, CLASS_WEASEL]
CLASS_NAMES_LIST_STR = [CLASS_BOBCAT_STRING, CLASS_COYOTE_STRING, CLASS_DEER_STRING, CLASS_ELK_STRING, CLASS_HUMAN_STRING, CLASS_NOT_INTERESTING_STRING, CLASS_RACCOON_STRING, CLASS_WEASEL_STRING]

TEST_PRINTING = False
IS_SAVE_THE_DATASETS = True
IS_SAVE_THE_PNGS = False
IS_DOWNLOAD_PICTURES = False


def main(args):
	random.seed()
	rInt = random.randint(0, 2**32 - 1) ##must be between 0 and 2**32 - 1
	print("Global level random seed: " + str(rInt))
	tf.keras.utils.set_random_seed(rInt) #sets random, numpy, and tensorflow seeds.
	#Tensorflow has a global random seed and a operation level seeds: https://www.tensorflow.org/api_docs/python/tf/random/set_seed
	
	print("Hello! This is the Animal Crossing Dataset Loader!")
	if not areAllProgramsInstalled():
		print("Not all programs installed.")
		exit(-2)
	makeDirectories(ALL_FOLDERS_LIST)
	checkArgs(args)
	
	print("DATASET_DIRECTORY: " + str(DATASET_DIRECTORY))

	print("Creating file structure...")
	# createFileStructure(INTERESTING_DIRECTORY, DATASET_COPY_FOLDER_INT)
	createFileStructure(INTERESTING_COYOTE_DIRECTORY, DATASET_COPY_FOLDER_COYOTE)
	createFileStructure(INTERESTING_ELK_DIRECTORY, DATASET_COPY_FOLDER_ELK)
	createFileStructure(INTERESTING_HUMAN_DIRECTORY, DATASET_COPY_FOLDER_HUMAN)
	createFileStructure(INTERESTING_BOBCAT_DIRECTORY, DATASET_COPY_FOLDER_BOBCAT)
	createFileStructure(INTERESTING_DEER_DIRECTORY, DATASET_COPY_FOLDER_DEER)
	createFileStructure(INTERESTING_RACCOON_DIRECTORY, DATASET_COPY_FOLDER_RACCOON)
	createFileStructure(INTERESTING_WEASEL_DIRECTORY, DATASET_COPY_FOLDER_WEASEL)
	createFileStructure(NOT_INTERESTING_DIRECTORY, DATASET_COPY_FOLDER_NOT)
	print("Done!")
	
	# interestingFNames = getListOfAnimalPicsInOneClass(DATASET_COPY_FOLDER_INT)	
	coyoteFNames = getListOfAnimalPicsInOneClass(DATASET_COPY_FOLDER_COYOTE)
	elkFNames = getListOfAnimalPicsInOneClass(DATASET_COPY_FOLDER_ELK)
	humanFNames = getListOfAnimalPicsInOneClass(DATASET_COPY_FOLDER_HUMAN)
	bobcatFNames = getListOfAnimalPicsInOneClass(DATASET_COPY_FOLDER_BOBCAT)
	deerFNames = getListOfAnimalPicsInOneClass(DATASET_COPY_FOLDER_DEER)
	raccoonFNames = getListOfAnimalPicsInOneClass(DATASET_COPY_FOLDER_RACCOON)
	weaselFNames = getListOfAnimalPicsInOneClass(DATASET_COPY_FOLDER_WEASEL)
	notInterestingFNames = getListOfAnimalPicsInOneClass(DATASET_COPY_FOLDER_NOT)
	
	#These WILL change later
	# ~ img_width = 400
	# ~ img_height = 300
	img_width = 100
	img_height = 100
	batch_size = 32
	percentageTrain = 0.6
	percentageTestToVal = 0.75

	print("creating the datasets...")
	train_ds, val_ds, test_ds = createAnimalsDataset(
			DATASET_COPY_FOLDER, img_height, img_width, batch_size, percentageTrain, percentageTestToVal)
	print("Done!")
	
	print("Saving datasets...")
	if IS_SAVE_THE_DATASETS:
		saveDatasets(
				train_ds, TRAIN_SAVE_DIRECTORY,
				val_ds, VAL_SAVE_DIRECTORY,
				test_ds, TEST_SAVE_DIRECTORY)
		print("Done!")
	else:
		print("Saving disabled for now!")
	
	if IS_SAVE_THE_PNGS:
		print("Saving the datasets as image files...")
		saveDatasetAsPNG(train_ds, DATASET_PNG_FOLDER_TRAIN)
		saveDatasetAsPNG(val_ds, DATASET_PNG_FOLDER_VAL)
		saveDatasetAsPNG(test_ds, DATASET_PNG_FOLDER_TEST)
	else:
		print("PNG saving disabled for now!")

	print("Deleting the temporary image folder...")
	shutil.rmtree(DATASET_COPY_FOLDER)
	
	if sys.platform.startswith("linux"):
		os.sync()
	
	print("Done!")

	return 0


# Creates the necessary directories.
def makeDirectories(listOfFoldersToCreate):
	#Clear out old files -- Justin Case
	if os.path.isdir(DATASET_SAVE_DIR):
		shutil.rmtree(DATASET_SAVE_DIR, ignore_errors = True)
	if os.path.isdir(DATASET_PNG_FOLDER):
		shutil.rmtree(DATASET_PNG_FOLDER, ignore_errors = True)
	if os.path.isdir(DATASET_COPY_FOLDER):
		shutil.rmtree(DATASET_COPY_FOLDER, ignore_errors = True)
	
	for folder in listOfFoldersToCreate:
		if not os.path.isdir(folder):
			os.makedirs(folder)


# Retrieves the images if they're not here
def retrieveImages():
	print("Retrieving images...")
	wgetString = "wget -e robots=off -r -np --mirror https://ftp.wsdot.wa.gov/public/I90Snoq/Biology/thermal/"
	runSystemCommand(wgetString)
	print("Done!")


#Runs a system command. Input is the string that would run on linux or inside wsl.
def runSystemCommand(inputString):
	if sys.platform.startswith("win"):
		print("winders")
		theStringLol = "wsl  " + inputString
		print(theStringLol)
		sysRet = os.system(theStringLol)
		print(sysRet)
		exit(2)
	elif sys.platform.startswith("linux"):
		print("Linux")
		sysRet = os.system(inputString)
		print(sysRet)
		# ~ os.system(inputString)
		exit(3)
	else:
		print("MASSIVE ERROR LOL!")
		exit(-4)


#Checks if a flag file is in place to determine if the dataset should download from the ftp server.
def isDownloadedFlagFileSet():
	if not os.path.isfile(HIDDEN_DOWNLOAD_FLAG_FILE):
		Path(HIDDEN_DOWNLOAD_FLAG_FILE).touch(exist_ok=True)

		return False
	
	return True
	

#Takes some images from the validation set and sets the aside for the test set.
def createTestSet(val_ds, percentageTestToVal):
	length = np.asarray(val_ds.cardinality())
	numForTest = int(length * percentageTestToVal)
	test_dataset = val_ds.take(numForTest)
	val_ds = val_ds.skip(numForTest)
	
	return val_ds, test_dataset


def saveDatasets(train_ds, trainDir, val_ds, valDir, test_ds, testDir):
	tf.data.experimental.save(train_ds, trainDir)
	tf.data.experimental.save(val_ds, valDir)
	tf.data.experimental.save(test_ds, testDir)


#The batching makes them get stuck together in batches. Right now that's 32 images.
#So whenever you take one from the set, you get a batch of 32 images.
# percentageTrain is a decimal from 0 to 1 of the percent data that should be for train
# percentageTestToVal is a number from 0 to 1 of the percentage of the non-train data for use as test
def createAnimalsDataset(baseDirectory, img_height, img_width, batch_size, percentageTrain, percentageTestToVal):
	valSplit = 1 - percentageTrain
	splitSeed = random.randint(0, 2**32 - 1)
	print("Operation level random seed for image_dataset_from_directory(): " + str(splitSeed))
	
	train_ds = tf.keras.preprocessing.image_dataset_from_directory(
		baseDirectory,
		labels = "inferred",
		label_mode = "int",
		class_names = CLASS_NAMES_LIST_STR, #must match directory names
		color_mode = "rgb",
		validation_split = valSplit,
		subset="training",
		seed = splitSeed,
		image_size=(img_height, img_width),
		batch_size=batch_size)

	val_ds = tf.keras.preprocessing.image_dataset_from_directory(
		baseDirectory,
		labels = "inferred",
		label_mode = "int",
		class_names = CLASS_NAMES_LIST_STR, #must match directory names
		color_mode = "rgb",
		validation_split = valSplit,
		subset="validation",
		seed = splitSeed,
		image_size=(img_height, img_width),
		batch_size=batch_size)

	val_ds, test_ds = createTestSet(val_ds, percentageTestToVal)

	AUTOTUNE = tf.data.AUTOTUNE

	normalization_layer = tf.keras.layers.Rescaling(1./255) #for newer versions of tensorflow
	# ~ normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255) #for old versions
	train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y),  num_parallel_calls=AUTOTUNE)
	val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y),  num_parallel_calls=AUTOTUNE)
	test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y),  num_parallel_calls=AUTOTUNE)

	flippyBoy = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal")
	train_ds = train_ds.map(lambda x, y: (flippyBoy(x), y),  num_parallel_calls=AUTOTUNE)
	val_ds = val_ds.map(lambda x, y: (flippyBoy(x), y),  num_parallel_calls=AUTOTUNE)

	myRotate = tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
	train_ds = train_ds.map(lambda x, y: (myRotate(x), y),  num_parallel_calls=AUTOTUNE)
	val_ds = val_ds.map(lambda x, y: (myRotate(x), y),  num_parallel_calls=AUTOTUNE)

	train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
	val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
	test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
	
	if TEST_PRINTING:
		print("Showing some unaltered images from the testing set...")
		printSample(test_ds)

	if TEST_PRINTING:
		print("Showing some augmented images from the training set...")
		printSample(train_ds)

	return train_ds, val_ds, test_ds


# Prints first nine images from the first batch of the dataset.
# It's random as long as you shuffle the dataset! ;)
def printSample(in_ds):
	plt.figure(figsize=(10, 10))
	for img, label in in_ds.take(1):
		for i in tqdm(range(9)):
			ax = plt.subplot(3, 3, i + 1)
			myImg = np.asarray(img)
			plt.imshow(np.asarray(myImg[i]), cmap="gray")
			plt.title( CLASS_NAMES_LIST_STR[ np.asarray(label[i]) ]  )
			plt.axis("off")
		plt.show()


#save all images from dataset to file as png
def saveDatasetAsPNG(in_ds, saveFolder):
	i = 0
	for batch in tqdm(in_ds):
		imgArr = np.asarray(batch[0])
		labelArr = np.asarray(batch[1])
		for j in range(len(imgArr)):
			thisImg = imgArr[j]
			thisImg = img_as_uint(thisImg)
			thisLabel = labelArr[j]
			filenamestring = os.path.join(saveFolder, CLASS_NAMES_LIST_STR[thisLabel], str(i) + ".png")
			imsave(filenamestring, thisImg)
			i = i + 1
			

def createFileStructure(baseDirSource, destination):
	recursivelyCopyAllFilesInFolderToOneDestinationFolder(baseDirSource, destination)


def recursivelyCopyAllFilesInFolderToOneDestinationFolder(baseDirSource, destination):
	print("Copying files to " + str(destination))
	cpyFiles = getListOfFilenames(baseDirSource)
	for thisName in tqdm(cpyFiles):
		try:
			shutil.copy(thisName, destination)
		except:
			print("copy skipping: " + str(thisName))


def getListOfAnimalPicsInOneClass(classDir):
	dirNames = getListOfDirNames(classDir)
	picNames = []
	for dName in dirNames:
		picNames.extend( getCuratedListOfFileNames(dName) )
	
	return picNames
	

def getCuratedListOfFileNames(directoryName):
	thisNames = getListOfFilenames(directoryName)
	thisNames = keepOnlyJPG(thisNames)
	
	return thisNames


def keepOnlyJPG(inList):
	for thingy in inList:
		pathParts = os.path.splitext(thingy)
		if pathParts[-1].lower() != ".jpg" and pathParts[-1].lower() != ".jpeg":
			print("excluding non-jpg!: " + str(thingy))
			inList.remove(thingy)
	
	return inList


#Returns a list of filenames from the input directory
def getListOfFilenames(baseDirectory):
	myNames = []
	for (root, dirNames, fileNames) in os.walk(baseDirectory):
		for aFile in  fileNames:
			myNames.append( os.path.join( root, aFile ) )
	
	return myNames


#Returns a list of dirnames from the base
def getListOfDirNames(baseDirectory):
	myNames = []
	for (root, dirNames, fileNames) in os.walk(baseDirectory):
		for aDir in  dirNames:
			myNames.append( os.path.join( root, aDir ) )
	
	return myNames


def checkArgs(args):
	shouldIRetrieveImages = False

	#for people not using a terminal; they can set the flag.
	if IS_DOWNLOAD_PICTURES:
		shouldIRetrieveImages = True
	if len(args) > 1:
		downloadArgs = ["--download", "-download", "download", "d", "-d", "--d"]
		if not set(downloadArgs).isdisjoint(args):
			shouldIRetrieveImages = True
	#for the first time user
	if not isDownloadedFlagFileSet():
		shouldIRetrieveImages = True

	if shouldIRetrieveImages:
		retrieveImages()


## Check if all the required programs are installed
def areAllProgramsInstalled():
	if sys.platform.startswith("win"):
		if shutil.which("wsl") is not None:
			## call a subprocess to run the "which wget" command inside wsl
			isWgetInstalled = subprocess.check_call(["wsl", "which", "wget"]) == 0
			if isWgetInstalled:
				return True
			else:
				print("Missing wget")
				return False
		else:
			print("Missing wsl")
	elif sys.platform.startswith("linux"):
		if (shutil.which("wget") is not None):
			return True
		else:
			print("Missing wget")
			return False
	else:
		print("Unsupportd operating system! Halting!")
		print("This os: " + str(sys.platform))
		return False
	
	return False


if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
