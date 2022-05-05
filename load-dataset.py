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
import skimage
from skimage.io import imsave
from skimage.util import img_as_uint
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm #Pretty loading bars

import numpy as np
import tensorflow as tf

print("Done!")

CLASS_INTERESTING = 0
CLASS_NOT_INTERESTING = 1

CLASS_INTERESTING_STRING = "interesting"
CLASS_NOT_INTERESTING_STRING = "not"

DATASET_DIRECTORY = "./ftp.wsdot.wa.gov/public/I90Snoq/Biology/thermal/614s/"
INTERESTING_DIRECTORY = "./ftp.wsdot.wa.gov/public/I90Snoq/Biology/thermal/614s/interesting/"
NOT_INTERESTING_DIRECTORY = "./ftp.wsdot.wa.gov/public/I90Snoq/Biology/thermal/614s/not interesting/"

DATASET_COPY_FOLDER = "./tmpdata/"
DATASET_COPY_FOLDER_INT = DATASET_COPY_FOLDER + CLASS_INTERESTING_STRING + "/"
DATASET_COPY_FOLDER_NOT = DATASET_COPY_FOLDER + CLASS_NOT_INTERESTING_STRING + "/"

DATASET_PNG_FOLDER = "./datasets-as-png/"
DATASET_PNG_FOLDER_TRAIN = DATASET_PNG_FOLDER + "train/"
DATASET_PNG_FOLDER_TRAIN_INT = DATASET_PNG_FOLDER_TRAIN + CLASS_INTERESTING_STRING + "/"
DATASET_PNG_FOLDER_TRAIN_NOT = DATASET_PNG_FOLDER_TRAIN + CLASS_NOT_INTERESTING_STRING + "/"
DATASET_PNG_FOLDER_VAL = DATASET_PNG_FOLDER + "val/"
DATASET_PNG_FOLDER_VAL_INT = DATASET_PNG_FOLDER_VAL + CLASS_INTERESTING_STRING + "/"
DATASET_PNG_FOLDER_VAL_NOT = DATASET_PNG_FOLDER_VAL + CLASS_NOT_INTERESTING_STRING + "/"
DATASET_PNG_FOLDER_TEST = DATASET_PNG_FOLDER + "test/"
DATASET_PNG_FOLDER_TEST_INT = DATASET_PNG_FOLDER_TEST + CLASS_INTERESTING_STRING + "/"
DATASET_PNG_FOLDER_TEST_NOT = DATASET_PNG_FOLDER_TEST + CLASS_NOT_INTERESTING_STRING + "/"

DATASET_SAVE_DIR = "./dataset/"
TRAIN_SAVE_DIRECTORY = "./dataset/train/"
VAL_SAVE_DIRECTORY = "./dataset/val/"
TEST_SAVE_DIRECTORY = "./dataset/test/"

ALL_FOLDERS_LIST = [
		DATASET_DIRECTORY,
		INTERESTING_DIRECTORY,
		NOT_INTERESTING_DIRECTORY,
		DATASET_COPY_FOLDER,
		DATASET_COPY_FOLDER_INT,
		DATASET_COPY_FOLDER_NOT,
		DATASET_PNG_FOLDER,
		DATASET_PNG_FOLDER_TRAIN,
		DATASET_PNG_FOLDER_TRAIN_INT,
		DATASET_PNG_FOLDER_TRAIN_NOT,
		DATASET_PNG_FOLDER_VAL,
		DATASET_PNG_FOLDER_VAL_INT,
		DATASET_PNG_FOLDER_VAL_NOT,
		DATASET_PNG_FOLDER_TEST,
		DATASET_PNG_FOLDER_TEST_INT,
		DATASET_PNG_FOLDER_TEST_NOT,
		DATASET_SAVE_DIR,
		TRAIN_SAVE_DIRECTORY,
		VAL_SAVE_DIRECTORY,
		TEST_SAVE_DIRECTORY
		]

HIDDEN_DOWNLOAD_FLAG_FILE = ".isnotfirstdownload"

CLASS_NAMES_LIST_INT = [CLASS_INTERESTING, CLASS_NOT_INTERESTING]
CLASS_NAMES_LIST_STR = [CLASS_INTERESTING_STRING, CLASS_NOT_INTERESTING_STRING]

TEST_PRINTING = False
IS_SAVE_THE_DATASETS = True
IS_SAVE_THE_PNGS = True
IS_DOWNLOAD_PICTURES = False


def main(args):
	print("Hello! This is the Animal Crossing Dataset Loader!")
	makeDirectories(ALL_FOLDERS_LIST)
	checkArgs(args)
	print("DATASET_DIRECTORY: " + str(DATASET_DIRECTORY))

	print("Creating file structure...")
	createFileStructure(INTERESTING_DIRECTORY, DATASET_COPY_FOLDER_INT)
	createFileStructure(NOT_INTERESTING_DIRECTORY, DATASET_COPY_FOLDER_NOT)
	print("Done!")
	
	interestingFNames = getListOfAnimalPicsInOneClass(DATASET_COPY_FOLDER_INT)
	notInterestingFNames = getListOfAnimalPicsInOneClass(DATASET_COPY_FOLDER_NOT)
	
	#These WILL change later
	img_height = 100
	img_width = 100
	# ~ img_height = 150
	# ~ img_width = 150
	# ~ img_height = 512
	# ~ img_width = 512
	# ~ img_height = 600
	# ~ img_width = 800
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
        os.system("wget -e robots=off -r -np --mirror https://ftp.wsdot.wa.gov/public/I90Snoq/Biology/thermal/")
        print("Done!")


#Checks if a flag file is in place to determine if the dataset should download from the ftp server.
def isDownloadedFlagFileSet():
	if not os.path.isfile(HIDDEN_DOWNLOAD_FLAG_FILE):
		Path(HIDDEN_DOWNLOAD_FLAG_FILE).touch(exist_ok=True)

		return False
	
	return True
	

#Takes some images from the validation set and sets the aside for the test set.
def createTestSet(val_ds, percentageTestToVal):
	length = np.asarray(tf.data.experimental.cardinality(val_ds))
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
	
	train_ds = tf.keras.preprocessing.image_dataset_from_directory(
		baseDirectory,
		labels = "inferred",
		label_mode = "int",
		class_names = CLASS_NAMES_LIST_STR, #must match directory names
		color_mode = "grayscale",
		validation_split = valSplit,
		subset="training",
		seed=123,
		image_size=(img_height, img_width),
		batch_size=batch_size)

	val_ds = tf.keras.preprocessing.image_dataset_from_directory(
		baseDirectory,
		labels = "inferred",
		label_mode = "int",
		class_names = CLASS_NAMES_LIST_STR, #must match directory names
		color_mode = "grayscale",
		validation_split = valSplit,
		subset="validation",
		seed=123,
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
			filenamestring = saveFolder + CLASS_NAMES_LIST_STR[thisLabel] + "/" + str(i) + ".png"
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


if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
