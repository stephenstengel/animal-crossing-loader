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

DATASET_DIRECTORY = "../aminals/ftp.wsdot.wa.gov/public/I90Snoq/Biology/thermal/614s/"
INTERESTING_DIRECTORY = "../aminals/ftp.wsdot.wa.gov/public/I90Snoq/Biology/thermal/614s/interesting/"
NOT_INTERESTING_DIRECTORY = "../aminals/ftp.wsdot.wa.gov/public/I90Snoq/Biology/thermal/614s/not interesting/"
COMPILED_FILE_DIRECTORY = "./dataset/"
DATASET_COPY_FOLDER = "./tmpdata/"
DATASET_COPY_FOLDER_INT = "./tmpdata/int/"
DATASET_COPY_FOLDER_NOT = "./tmpdata/not/"


CLASS_INTERESTING = 0
CLASS_NOT_INTERESTING = 1


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
	
	
	createFileStructure(INTERESTING_DIRECTORY, DATASET_COPY_FOLDER_INT)
	createFileStructure(NOT_INTERESTING_DIRECTORY, DATASET_COPY_FOLDER_NOT)
	
	
	##copy each subfolder into the base folder.
	
	
	interestingFNames = getListOfAnimalPicsInOneClass(DATASET_COPY_FOLDER_INT)
	notInterestingFNames = getListOfAnimalPicsInOneClass(DATASET_COPY_FOLDER_NOT)
	
	
	

	return 0


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
