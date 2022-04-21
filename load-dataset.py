#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  load-dataset.py
#  
#  2022 Stephen Stengel <stephen.stengel@cwu.edu>
#  
#  Script to load the dataset into file form for easy importing.
#  

import os
import skimage
import shutil
import matplotlib.pyplot as plt

import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
# ~ import tensorflow_datasets as tfds

DATASET_DIRECTORY = "../aminals/ftp.wsdot.wa.gov/public/I90Snoq/Biology/thermal/614s/"
INTERESTING_DIRECTORY = "../aminals/ftp.wsdot.wa.gov/public/I90Snoq/Biology/thermal/614s/interesting/"
NOT_INTERESTING_DIRECTORY = "../aminals/ftp.wsdot.wa.gov/public/I90Snoq/Biology/thermal/614s/not interesting/"
COMPILED_FILE_DIRECTORY = "./dataset/"
DATASET_COPY_FOLDER = "./tmpdata/"
DATASET_COPY_FOLDER_INT = "./tmpdata/int/"
DATASET_COPY_FOLDER_NOT = "./tmpdata/not/"


CLASS_INTERESTING = 0
CLASS_NOT_INTERESTING = 1

TEST_PRINTING = False


def main(args):
	print("hey")
	checkArgs(args)
	print("DATASET_DIRECTORY: " + str(DATASET_DIRECTORY))
	
	if not os.path.isdir(DATASET_COPY_FOLDER):
		os.mkdir(DATASET_COPY_FOLDER)
	if not os.path.isdir(DATASET_COPY_FOLDER_INT):
		os.mkdir(DATASET_COPY_FOLDER_INT)
	if not os.path.isdir(DATASET_COPY_FOLDER_NOT):
		os.mkdir(DATASET_COPY_FOLDER_NOT)
	
	print("Creating file structure...")
	createFileStructure(INTERESTING_DIRECTORY, DATASET_COPY_FOLDER_INT)
	createFileStructure(NOT_INTERESTING_DIRECTORY, DATASET_COPY_FOLDER_NOT)
	print("Done!")
	
	interestingFNames = getListOfAnimalPicsInOneClass(DATASET_COPY_FOLDER_INT)
	notInterestingFNames = getListOfAnimalPicsInOneClass(DATASET_COPY_FOLDER_NOT)
	
	#use the tensorflow api to load the data from the base folder.
	#DATASET_COPY_FOLDER
	
	#These could change later
	# ~ img_height = 180
	# ~ img_width = 180
	img_height = 600
	img_width = 800
	batch_size = 32


	train_ds, val_ds = createAnimalsDataset(DATASET_COPY_FOLDER, img_height, img_width, batch_size)
	

	return 0



#Must use tf.keras.layers.Rescaling(1./255) as first layer in model !!!
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

	class_names = train_ds.class_names
	print("class names: " + str(class_names))
	

	if TEST_PRINTING:
		plt.figure(figsize=(10, 10))
		for images, labels in train_ds.take(1):
			for i in range(9):
				ax = plt.subplot(3, 3, i + 1)
				plt.imshow(images[i].numpy().astype("uint8"))
				plt.title(class_names[labels[i]])
				plt.axis("off")
			plt.show()


	# ~ normalization_layer = tf.keras.layers.Rescaling(1./255) #for new versions
	normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255) #for old versions
	
	normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
	normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))


	return normalized_train_ds, normalized_val_ds

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
	if len(args) > 1:
		global DATASET_DIRECTORY
		DATASET_DIRECTORY = args[1]


if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
