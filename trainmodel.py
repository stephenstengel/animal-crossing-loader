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
from tensorflow.keras import layers, models
from tensorflow import keras
from keras import callbacks


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
	
	train, val, test, unscaledTest = getDatasets(TRAIN_SAVE_DIRECTORY, VAL_SAVE_DIRECTORY, TEST_SAVE_DIRECTORY)
	
	if TEST_PRINTING:
		printRandomSample(unscaledTest)
	
	shape = getPictureShape(test)
	print("the shape of the things: " + str(shape))
	
	# HARD CODED TO 150x150 right now!
	myModel = makeXceptionBasedModel()
	myHistory, myModel = trainModel(myModel, train, val)
	
	evaluateTraining(myHistory)
	
	evaluateModel(myModel, test)
	
	predictTestSet(myModel, test)
	
	print("DONE")


def predictTestSet(myModel, test_ds):
	predictions = myModel.predict(test_ds)
	print("predictions object: " + str(predictions))


def evaluateModel(myModel, test):
	print("Calculating scores...")
	scores = myModel.evaluate(test)
	print("Done!")
	print("Scores object: " + str(scores))
	
	print("Accuracy on test set: " + str(scores[1]))
	
	



def evaluateTraining(history):
	accuracy = history.history["accuracy"]
	val_accuracy = history.history["val_accuracy"]
	
	loss = history.history["loss"]
	val_loss = history.history["val_loss"]
	epochs = range(1, len(accuracy) + 1)
	plt.plot(epochs, accuracy, "o", label="Training accuracy")
	plt.plot(epochs, val_accuracy, "^", label="Validation accuracy")
	plt.title("Training and validation accuracy")
	plt.legend()
	plt.savefig("trainvalacc.png")
	plt.clf()
	
	plt.plot(epochs, loss, "o", label="Training loss")
	plt.plot(epochs, val_loss, "^", label="Validation loss")
	plt.title("Training and validation loss")
	plt.legend()
	plt.savefig("trainvalloss.png")
	plt.clf()


def getDatasets(trainDir, valDir, testDir):
	train = tf.data.experimental.load(trainDir)
	val = tf.data.experimental.load(valDir)
	test = tf.data.experimental.load(testDir)
	
	unscaledTest = test
	
	AUTOTUNE = tf.data.AUTOTUNE
	
	#Input needs to be scaled and shifted for xception.
	# ~ normalization_layer = tf.keras.layers.Rescaling(1.0 / 127.5, offset = -1)
	normalization_layer = tf.keras.layers.Rescaling(2.0, offset = -1) #this is lossy? #already scaled by load-dataset.py at the moment
	
	train = train.map(lambda x, y: (normalization_layer(x), y),  num_parallel_calls=AUTOTUNE)
	val = val.map(lambda x, y: (normalization_layer(x), y),  num_parallel_calls=AUTOTUNE)
	test = test.map(lambda x, y: (normalization_layer(x), y),  num_parallel_calls=AUTOTUNE)
	
	return train, val, test, unscaledTest
	

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
	
	plt.clf()


#get shape of pictures in model
def getPictureShape(in_ds):
	for img, label in in_ds.take(1):
		myImg = np.asarray(img)
		return np.asarray(myImg[0]).shape


#HARD CODED 150 RIGHT NOW
#imageShape is a tiple containing h w channels  (or was it w h channels? idk)
def makeXceptionBasedModel():
	base_model = keras.applications.Xception(
		weights="imagenet",  
		input_shape=(150, 150, 3),
		include_top=False,
	)  
	
	# Freeze the base_model
	base_model.trainable = False
	
	print("Base model summary...")
	print(base_model.summary())
	
	# Rest of the model
	x = base_model.output
	x = layers.GlobalAveragePooling2D()(x)
	x = layers.Dense(4096, activation='relu')(x)
	# ~ x = layers.Dense(2, activation='sigmoid')(x)
	x = layers.Dense(2)(x)
	
	myModel = models.Model(inputs=base_model.input, outputs=x)
	
	myModel.compile(
			optimizer='adam',
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy'])
	
	myModel.summary()
	
	return myModel


def trainModel(model, train_ds, val_ds):
	checkpointFolder = "checkpoint/"
	
	checkpointer = callbacks.ModelCheckpoint(
			filepath = checkpointFolder,
			monitor = "accuracy",
			save_best_only = True,
			mode = "max",
			verbose = 1)
	callbacks_list = [checkpointer]
		
	epochs = 20
	stepsPerEpoch = 20
	myHistory = model.fit(
			train_ds,
			epochs=epochs,
			# ~ steps_per_epoch = stepsPerEpoch,
			validation_data=val_ds,
			callbacks = callbacks_list)
			
			## Can also use validation_split, to automatically do validation
	
	return myHistory, model
	



if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
