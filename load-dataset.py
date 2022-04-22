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
import shutil
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import tensorflow as tf

print("Done!")

DATASET_DIRECTORY = "./ftp.wsdot.wa.gov/public/I90Snoq/Biology/thermal/614s/"
INTERESTING_DIRECTORY = "./ftp.wsdot.wa.gov/public/I90Snoq/Biology/thermal/614s/interesting/"
NOT_INTERESTING_DIRECTORY = "./ftp.wsdot.wa.gov/public/I90Snoq/Biology/thermal/614s/not interesting/"

DATASET_COPY_FOLDER = "./tmpdata/"
DATASET_COPY_FOLDER_INT = "./tmpdata/int/"
DATASET_COPY_FOLDER_NOT = "./tmpdata/not/"

DATASET_SAVE_DIR = "./dataset/"
TRAIN_SAVE_DIRECTORY = "./dataset/train/"
VAL_SAVE_DIRECTORY = "./dataset/val/"
TEST_SAVE_DIRECTORY = "./dataset/test/"

HIDDEN_DOWNLOAD_FLAG_FILE = ".isnotfirstdownload"

#unused currently because image_dataset_from_directory creates labels automatically.
CLASS_INTERESTING = 0
CLASS_NOT_INTERESTING = 1

TEST_PRINTING = False
IS_SAVE_THE_DATASETS = False
IS_DOWNLOAD_PICTURES = False


def main(args):
	print("Hello! This is the Animal Crossing Dataset Loader!")
	makeDirectories()
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
	# ~ img_height = 600
	# ~ img_width = 800
	batch_size = 32

	print("creating the datasets...")
	train_ds, val_ds, test_ds = createAnimalsDataset(
			DATASET_COPY_FOLDER, img_height, img_width, batch_size)
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

	return 0


#There is an easier way.
def makeDirectories():
	if not os.path.isdir(DATASET_COPY_FOLDER):
		os.mkdir(DATASET_COPY_FOLDER)
	if not os.path.isdir(DATASET_COPY_FOLDER_INT):
		os.mkdir(DATASET_COPY_FOLDER_INT)
	if not os.path.isdir(DATASET_COPY_FOLDER_NOT):
		os.mkdir(DATASET_COPY_FOLDER_NOT)
	
	if not os.path.isdir(DATASET_SAVE_DIR):
		os.mkdir(DATASET_SAVE_DIR)
	if not os.path.isdir(TRAIN_SAVE_DIRECTORY):
		os.mkdir(TRAIN_SAVE_DIRECTORY)
	if not os.path.isdir(VAL_SAVE_DIRECTORY):
		os.mkdir(VAL_SAVE_DIRECTORY)
	if not os.path.isdir(TEST_SAVE_DIRECTORY):
		os.mkdir(TEST_SAVE_DIRECTORY)
	
	#makedirs is the easy way. It makes all the required parent folders if they don't exist.
	if not os.path.isdir(DATASET_DIRECTORY):
		os.makedirs(DATASET_DIRECTORY)


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
def createTestSet(val_ds):
	val_batches = tf.data.experimental.cardinality(val_ds)
	test_dataset = val_ds.take(val_batches // 5)
	val_ds = val_ds.skip(val_batches // 5)
	
	return val_ds, test_dataset


def saveDatasets(train_ds, trainDir, val_ds, valDir, test_ds, testDir):
	tf.data.experimental.save(train_ds, trainDir)
	tf.data.experimental.save(val_ds, valDir)
	tf.data.experimental.save(test_ds, testDir)


#Split into helper functions later.
def createAnimalsDataset(baseDirectory, img_height, img_width, batch_size):
	train_ds = tf.keras.preprocessing.image_dataset_from_directory(
		baseDirectory,
		color_mode = "rgb",
		validation_split=0.2,
		subset="training",
		seed=123,
		image_size=(img_height, img_width),
		batch_size=batch_size)

	val_ds = tf.keras.preprocessing.image_dataset_from_directory(
		baseDirectory,
		color_mode = "rgb",
		validation_split=0.2,
		subset="validation",
		seed=123,
		image_size=(img_height, img_width),
		batch_size=batch_size)

	if TEST_PRINTING:
		plt.figure(figsize=(10, 10))
		for images, labels in train_ds.take(1):
			for i in range(9):
				ax = plt.subplot(3, 3, i + 1)
				plt.imshow(images[i].numpy().astype("uint8"))
				plt.title(class_names[labels[i]])
				plt.axis("off")
			plt.show()

	class_names = train_ds.class_names
	print("class names: " + str(class_names))

	# ~ normalization_layer = tf.keras.layers.Rescaling(1./255) #for new versions
	normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255) #for old versions
	n_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
	n_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

	n_val_ds, n_test_ds = createTestSet(n_val_ds)

	flippyBoy = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal")
	n_train_ds = val_ds.map(lambda x, y: (flippyBoy(x), y))
	n_val_ds = val_ds.map(lambda x, y: (flippyBoy(x), y))

	myRotate = tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
	n_train_ds = val_ds.map(lambda x, y: (myRotate(x), y))
	n_val_ds = val_ds.map(lambda x, y: (myRotate(x), y))

	AUTOTUNE = tf.data.AUTOTUNE
	n_train_ds = n_train_ds.prefetch(buffer_size=AUTOTUNE)
	n_val_ds = n_val_ds.prefetch(buffer_size=AUTOTUNE)
	n_test_ds = n_test_ds.prefetch(buffer_size=AUTOTUNE)
	
	return n_train_ds, n_val_ds, n_test_ds


def createFileStructure(baseDirSource, destination):
	copyDatasetToTMP(baseDirSource, destination)
	
	dirNames = getListOfDirNames(destination)
	for dName in dirNames:
		copyDatasetToTMP(dName, destination)
	

def copyDatasetToTMP(baseDirSource, destination):
	cpyFiles = getListOfFilenames(baseDirSource)
	for thisName in cpyFiles:
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
	#for people not using a terminal; they can set the flag.
	if IS_DOWNLOAD_PICTURES:
		retrieveImages()
	if len(args) > 1:
		downloadArgs = ["--download", "-download", "download", "d", "-d", "--d"]
		if not set(downloadArgs).isdisjoint(args):
			retrieveImages()
	#for the first time user
	if not isDownloadedFlagFileSet():
		retrieveImages()


if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
