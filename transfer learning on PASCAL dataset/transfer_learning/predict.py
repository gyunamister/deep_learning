import shutil
import os
import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

#fixed size for InceptionV3 architecture
target_size = (229, 229)

def move_files(_dir):
  all_files = os.listdir(_dir)
  os.mkdir("{}/{}".format(_dir,'test/'));
  for f in all_files:
    shutil.copy(_dir+'/'+f, "{}/{}".format(_dir,'test/'))

def predict(model, _dir, target_size):
  """
  predict the test dataset and return result as text(test.txt)
  """
  test_datagen =  ImageDataGenerator(
    preprocessing_function=preprocess_input
  )

  generator = test_datagen.flow_from_directory(
        _dir,
        target_size=target_size,
        batch_size=32,
        class_mode=None,  # only data, no labels
        shuffle=False)  # keep data in same order as labels
  preds = model.predict_generator(generator)
  list_preds = []
  for index, pred in enumerate(preds):
    list_preds.append([str(x) for x in pred])
  with open('test.txt', 'w') as file:
    file.writelines('\t'.join(i) + '\n' for i in list_preds)



if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--test_dir", help="path to test_image")
  a.add_argument("--model")
  args = a.parse_args()

  if args.test_dir is None:
    a.print_help()
    sys.exit(1)

  move_files(args.test_dir)
  """
  model = load_model(args.model)
  if args.test_dir is not None:
    preds = predict(model, args.test_dir, target_size)
  """