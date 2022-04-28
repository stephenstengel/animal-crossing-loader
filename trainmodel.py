#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  train-model.py
#  
#  2022 Stephen Stengel <stephen.stengel@cwu.edu>
#  
#  Script to train a model with our dataset.
#  

print("Loading imports...")

import os
import skimage
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm #Pretty loading bars

import numpy as np
import tensorflow as tf


print("Done!")

## These global defines are copy pasted from the load-dataset.py
## Might be a more elegant way

DATASET_SAVE_DIR = "./dataset/"
TRAIN_SAVE_DIRECTORY = "./dataset/train/"
VAL_SAVE_DIRECTORY = "./dataset/val/"
TEST_SAVE_DIRECTORY = "./dataset/test/"

CLASS_INTERESTING = 0
CLASS_NOT_INTERESTING = 1

CLASS_INTERESTING_STRING = "interesting"
CLASS_NOT_INTERESTING_STRING = "not"

CLASS_NAMES_LIST_INT = [CLASS_INTERESTING, CLASS_NOT_INTERESTING]
CLASS_NAMES_LIST_STR = [CLASS_INTERESTING_STRING, CLASS_NOT_INTERESTING_STRING]

TEST_PRINTING = True


def main(args):
	print("hi")
	
	train, val, test = getDatasets(TRAIN_SAVE_DIRECTORY, VAL_SAVE_DIRECTORY, TEST_SAVE_DIRECTORY)
	
	if TEST_PRINTING:
		printRandomSample(train)
	
	
	
	print("DONE")




def getDatasets(trainDir, valDir, testDir):
	train = tf.data.experimental.load(trainDir)
	val = tf.data.experimental.load(valDir)
	test = tf.data.experimental.load(testDir)
	
	return train, val, test
	

#Prints nine random images from the dataset.
def printRandomSample(in_ds):
	plt.figure(figsize=(10, 10))
	for img, label in in_ds.take(1):
		for i in tqdm(range(9)):
			ax = plt.subplot(3, 3, i + 1)
			myImg = np.asarray(img)
			plt.imshow(np.asarray(myImg[i]), cmap="gray")
			plt.title( CLASS_NAMES_LIST_STR[ np.asarray(label[i]) ]  )
			plt.axis("off")
		plt.show()


#imageShape is a tiple containing h w channels  (or was it w h channels? idk)
def makeTheModel(imageShape):
	inputShape = imageShape
	base_model = keras.applications.Xception(
		weights="imagenet",  # Load weights pre-trained on ImageNet.
		input_shape=(150, 150, 3),
		include_top=False,
	)  # Do not include the ImageNet classifier at the top.
	
	# Freeze the base_model
	base_model.trainable = False
	
	



if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
