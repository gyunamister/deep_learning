import os
import pandas as pd
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import shutil

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint

#fixed size for InceptionV3
IMAGE_WIDTH, IMAGE_HEIGHT = 299, 299
NUM_EPOCHS = 3
BAT_SIZE = 32
FC_SIZE = 1024
NUM_IV3_LAYERS_TO_FREEZE = 172

def listify_category(_dir):
	"""list category of images"""
	all_files = os.listdir(_dir)
	image_sets = sorted(list(set([filename.replace('.txt', '').strip().split('_')[0]  for filename in all_files if len(filename.replace('.txt', '').strip().split('_')) > 1])))
	return image_sets


def imgs_from_category(cat_name, dataset,set_dir):
	"""return file names for corresponding category"""
	filename = os.path.join(set_dir, cat_name + "_" + dataset + ".txt")
	df = pd.read_csv(
	    filename,
	    delim_whitespace=True,
	    header=None,
	    names=['filename', 'true'])
	return df

def imgs_from_category_as_list(cat_name, dataset, set_dir):
	"""return file names as list"""
	df = imgs_from_category(cat_name, dataset,set_dir)
	df = df[df['true'] == 1]
	return df['filename'].values





def get_num_files(directory):
	"""Get number of files by searching directory recursively"""
	if not os.path.exists(directory):
		return 0
	cnt = 0
	for r, dirs, files in os.walk(directory):
		for dr in dirs:
			cnt += len(glob.glob(os.path.join(r, dr + "/*")))
	return cnt

def add_new_last_layer(base_model, num_classes):
	"""Add last layer to the convnet

	Args:
	base_model: keras model excluding top
	num_classes: # of classes

	Returns:
	new keras model with last layer
	"""
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
	predictions = Dense(num_classes, activation='sigmoid')(x) #new sigmoid layer
	model = Model(inputs=base_model.input, outputs=predictions)
	return model


def setup_to_finetune(model):
	"""Freeze the bottom num_IV3_LAYERS and retrain the remaining top layers.

	note: num_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch

	Args:
	model: keras model
	"""
	for layer in model.layers[:NUM_IV3_LAYERS_TO_FREEZE]:
		layer.trainable = False
	for layer in model.layers[NUM_IV3_LAYERS_TO_FREEZE:]:
		layer.trainable = True
	model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


def train(args):
	"""Use transfer learning and fine-tuning to train a network on a new dataset"""
	num_train_samples = get_num_files(args.image_dir)
	num_classes = len(glob.glob(args.image_dir + "/*"))
	num_val_samples = get_num_files(args.val_dir)
	num_epoch = int(args.num_epoch)
	batch_size = int(args.batch_size)

	# data prep + augmentation
	train_datagen =  ImageDataGenerator(
		preprocessing_function=preprocess_input,
		rotation_range=30,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True
	)
	test_datagen = ImageDataGenerator(
		preprocessing_function=preprocess_input,
		rotation_range=30,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True
	)


	train_generator = train_datagen.flow_from_directory(
		args.image_dir,
		target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
		batch_size=batch_size,
	)

	validation_generator = test_datagen.flow_from_directory(
		args.val_dir,
		target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
		batch_size=batch_size,
	)

  	#setup model, excludes final FC layer
	base_model = InceptionV3(weights='imagenet', include_top=False) #
	model = add_new_last_layer(base_model, num_classes)
	#fine-tuning
	setup_to_finetune(model)


	#save best models while iterating
	filepath="./models/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
	checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]

	#Fit the model
	history_ft = model.fit_generator(
		train_generator,
		steps_per_epoch=num_train_samples/batch_size,
		epochs=num_epoch,
		validation_data=validation_generator,
		validation_steps=num_val_samples/batch_size,
		callbacks=callbacks_list
	)
	#save the last model
	model.save(args.output_model_file)


def name_padding(_name, length):
	"""return adequate image filename"""
	if len(_name) < length:
		diff = length - len(_name)
		_name = ''.join(['0']*diff) + _name
		return _name
	else:
		return _name


def classify_files(_type, root_dir):
	"""
	classify image files according to their categories
	to be used in flow_from_directory() of generator()
	"""
	img_dir = os.path.join(root_dir, 'JPEGImages')
	set_dir = os.path.join(root_dir, 'ImageSets', 'Main')
	image_sets = listify_category(set_dir)

	folders_to_be_created = image_sets

	source = os.getcwd()
	source = os.path.join(source,'data', _type)
	for category in folders_to_be_created:
		new_path = os.path.join(source, category)
		if not os.path.exists(new_path):
			os.makedirs(new_path)

	folders = folders_to_be_created.copy()

	for category in folders_to_be_created:
		file_names = imgs_from_category_as_list(category, _type, set_dir)
		modified_file_names = []
		for _name in file_names:
			modified_file_names.append(name_padding(str(_name),6) + '.jpg')
		for f in modified_file_names:
			if not os.path.exists("{}/{}/{}".format(source,category,f)):
				shutil.copy("{}/{}".format(img_dir, f), "{}/{}".format(source,category))


if __name__ == '__main__':
	a = argparse.ArgumentParser()
	a.add_argument("--image_dir")
	a.add_argument("--num_epoch", default=NUM_EPOCHS)
	a.add_argument("--batch_size", default=BAT_SIZE)
	a.add_argument("--output_model_file", default="cust_inceptionv3.model")

	args = a.parse_args()
	if args.image_dir is None:
		a.print_help()
		sys.exit(1)

	if not os.path.exists(args.image_dir):
		print("directories do not exist")
		sys.exit(1)

	classify_files('train',args.image_dir)
	classify_files('val',args.image_dir)

	args.image_dir = "./data/train"
	args.val_dir = "./data/val"
	train(args)
