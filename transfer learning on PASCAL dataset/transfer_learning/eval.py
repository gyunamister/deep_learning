import os
import pandas as pd
import sys
import numpy as np
import argparse
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

root_dir = './VOC2007/'
img_dir = os.path.join(root_dir, 'JPEGImages')
set_dir = os.path.join(root_dir, 'ImageSets', 'Main')

def listify_category(_dir):
	all_files = os.listdir(_dir)
	image_sets = sorted(list(set([filename.replace('.txt', '').strip().split('_')[0]  for filename in all_files if len(filename.replace('.txt', '').strip().split('_')) > 1])))
	return image_sets

def read_result():
	result_file = './test.txt'
	result = pd.read_csv(result_file, sep='\t', index_col=None,names = ['{}'.format(i) for i in range(20)])
	return result

def classify_sets(_dir):
	result = read_result()

	image_sets = listify_category(_dir)
	result_index = pd.read_csv(_dir+'/test.txt', sep=' ', index_col=None, names=['index']).values.flatten()
	result.index = result_index
	print(result)
	for i, _set in enumerate(image_sets):
		set_result = result.loc[:,str(i)].astype(float)
		set_result.to_csv('./result/{}.txt'.format(_set), sep=' ', header=False)

def calculate_ap(image_sets,_dir):
	plt.figure(figsize=(30,15))
	for i, category in enumerate(image_sets):
		pred_path = './result/{}.txt'.format(category)
		pred = pd.read_csv(pred_path, delim_whitespace=True, index_col=0)
		true_path = _dir + '/{}_test.txt'.format(category)
		true = pd.read_csv(true_path, delim_whitespace=True,index_col=0)
		pred = pred.values.flatten()
		true = true.values.flatten()
		true = [0 if x==-1 else 1 for x in true]
		precision, recall, _ = precision_recall_curve(true, pred)
		ap = average_precision_score(true,pred)
		print("ap of {} = {}".format(category,ap))
		ax = plt.subplot(5,4,i+1)
		plt.step(recall, precision, color='b', alpha=0.2,
	         where='post')
		plt.fill_between(recall, precision, step='post', alpha=0.2,
		                 color='b')

		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim([0.0, 1.05])
		plt.xlim([0.0, 1.0])
		ax.set_title(category,fontweight='bold', color='blue')
		#plt.title('{} Precision-Recall curve: AP={0:0.2f}'.format(category,ap))
	plt.show()


if __name__=="__main__":
	a = argparse.ArgumentParser()
	a.add_argument("--test_dir")
	args = a.parse_args()
	image_sets = listify_category(args.test_dir)
	classify_sets(args.test_dir)
	calculate_ap(image_sets, args.test_dir)
