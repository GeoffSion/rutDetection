###################### rut_detection ######################
# This script is the main function that creates the U-net #
# calls the trainer, and makes the predictions.           #
###########################################################

from __future__ import division, print_function
from tf_unet.layers import (pixel_wise_softmax)
from tqdm import tqdm
# matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from tf_unet import unet
from tf_unet import image_util
import time
import cv2
import tensorflow as tf
from math import log


def tensorflow_shutup():
	"""
    Make Tensorflow less verbose
    """
	try:
		# noinspection PyPackageRequirements
		import os
		from tensorflow import logging
		logging.set_verbosity(logging.ERROR)
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

		# Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
		# noinspection PyUnusedLocal
		def deprecated(date, instructions, warn_once=True):
			def deprecated_wrapper(func):
				return func

			return deprecated_wrapper

		from tensorflow.python.util import deprecation
		deprecation.deprecated = deprecated

	except ImportError:
		pass


def diceCoeff(yTrue, yPred):
	eps = 1e-5
	yTrueF = yTrue.flatten()
	yPredF = yPred.flatten()
	# print(yPred[..., 0].mean(), yPred[..., 1].mean())
	intersection = np.sum(yPredF[yTrueF == 1])
	# intersectionZero = np.sum(yPredF[..., 1][yTrueF[..., 1] == 0])
	# print('intersectionZero', intersectionZero)
	# intersectionOne = np.sum(yPredF[..., 1][yTrueF[..., 1] == 1])
	# print('intersectionOne', intersectionOne)
	score = (2. * intersection + eps) / (eps + np.sum(yTrueF) + np.sum(yPredF))
	# scoreZero = (2. * intersectionZero + eps) / (eps + np.sum(yTrueF[..., 1]) + np.sum(yPredF[..., 1]))
	# scoreOne = (2. * intersectionOne + eps) / (eps + np.sum(yTrueF[..., 1]) + np.sum(yPred[..., 1]))
	# print(scoreZero, scoreOne, score)
	return score


def predict(model_name, test_data_path, train_path, data_suffix):
	print('Starting prediction for model evaluation...')
	data_provider = image_util.ImageDataProvider(search_path=test_data_path + "*" + data_suffix,
	                                             data_suffix=data_suffix,
	                                             mask_suffix="_mask" + data_suffix, shuffle_data='False', n_class=2)

	# print('Creating Unet')
	net = unet.Unet(channels=data_provider.channels, n_class=data_provider.n_class, layers=3, features_root=16)

	### Prediction ###
	# generator = image_util.ImageDataProvider(search_path=test_data_path+"*.jpg", data_suffix=".jpg",
	#                                          mask_suffix="_mask.jpg", shuffle_data='False')

	modelPerf_df = pd.Series([])
	pred_time_df = pd.Series([])

	# Compute diceCoeff for each image
	for i, j in zip(range(0, len(data_provider.data_files)), tqdm(range(len(data_provider.data_files)))):
		# Get test images and start prediction timer
		start_pred = float(time.time())
		x_test, y_test = data_provider(1)
		y_test = np.array(y_test, dtype='int')
		# Predict on images and end prediction timer
		prediction = net.predict(os.path.join(train_path, "model.ckpt"), x_test)
		end_pred = float(time.time())
		# Change type and shape of prediction to compare with true ground
		# (predicted mask is smaller due to convolutions without padding)
		# Input images must be with dimensions
		pred = np.array(prediction, dtype="float64")
		# print(pred[0, ..., 0].shape[0], pred[0, ..., 0].shape[1])

		a = y_test.shape[1] - pred.shape[1]  # x size difference
		b = y_test.shape[2] - pred.shape[2]  # y size difference

		resized_y_test = y_test[0, int(a / 2):-int(a / 2), int(b / 2):-int(b / 2), :]
		resized_x_test = x_test[0, int(a / 2):-int(a / 2), int(b / 2):-int(b / 2), :]

		##### post processing remove noise #####
		# my_dpi = 96
		# fig = plt.figure(frameon=False)
		# fig.set_size_inches(pred[0, ..., 0].shape[1] / my_dpi, pred[0, ..., 0].shape[0] / my_dpi)
		#
		# img = plt.imshow(pred[0, ..., 0])
		# img.set_cmap('Greys')
		# plt.axis('off')
		# plt.margins(0, 0)
		# plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
		#                     hspace=0, wspace=0)
		# fig.savefig(train_path + "/Predictions/figmask.png", dpi=my_dpi)
		# plt.close('all')
		#
		# figmask = cv2.imread(train_path + "/Predictions/figmask.png")
		#
		# # gray = cv2.cvtColor(figmask, cv2.COLOR_BGR2GRAY)  # convert to grayscale
		# blur = cv2.blur(figmask, (20, 20))  # blur the image
		# ret, thresh = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY)
		#
		# fig, ax = plt.subplots(1, 3, figsize=(12, 5))
		# ax[0].imshow(figmask, cmap='Greys', aspect="auto")
		# ax[1].imshow(blur, cmap='Greys', aspect="auto")
		# # mask = prediction[0, ..., 0] > 0.6
		# ax[2].imshow(thresh, cmap='Greys', aspect="auto")
		# plt.show()
		# print(pred[0])


		# test_pred = np.zeros(pred.shape)
		# test_pred[:, :, 450:, 1] = 1
		# test_pred[:, :, 450:, 0] = 0

		modelPerf = diceCoeff(resized_y_test, pred[0])  # pred[0]

		modelPerf_df = modelPerf_df.append(pd.Series([modelPerf]), ignore_index=True)
		pred_time_df = pred_time_df.append(pd.Series([end_pred - start_pred]), ignore_index=True)
		# print(data_provider.data_files[0][len(test_data_path):])
		# print(modelPerf_df.idxmax())
		# print(data_provider.data_files[modelPerf_df.idxmax()][len(test_data_path):])

		fig, ax = plt.subplots(1, 3, figsize=(12, 5))
		ax[0].imshow(resized_x_test[...], aspect="auto")
		ax[1].imshow(1 - resized_y_test[..., 1], cmap='Greys', aspect="auto")
		# ax[0].imshow(resized_x_test[...], aspect="auto")
		# ax[1].imshow(resized_y_test[..., -1], aspect="auto")
		mask = pred[0, ..., 1] > 0.6
		ax[2].imshow(1-pred[0, ..., 1], cmap='Greys', aspect="auto", vmin=0.0, vmax=1.0)
		ax[0].set_title("Input", fontsize=20)
		ax[1].set_title("Ground truth", fontsize=20)
		ax[2].set_title("Prediction", fontsize=20)
		ax[0].set_axis_off()
		ax[1].set_axis_off()
		ax[2].set_axis_off()
		fig.tight_layout()
		fig.savefig(train_path + "/Predictions/" + data_provider.data_files[i][len(test_data_path):], bbox_inches=0)
		plt.close()
	# plt.show()

	print('\nPrediction done')

	# Write stats in txt file
	f_stats = open(train_path + "/Predictions/" + model_name + "_stats.txt", "w+")
	f_stats.write(model_name + " stats\n\n")
	f_stats.write("Average Prediction Time (s): %1.3f  +- %2.3f" % (pred_time_df.mean(), pred_time_df.std()))
	f_stats.write("\nMax Dice Score: %1.3f  " % (modelPerf_df.max()))
	f_stats.write(data_provider.data_files[modelPerf_df.idxmax()][len(test_data_path):])
	f_stats.write("\nMin Dice Score: %1.3f  " % (modelPerf_df.min()))
	f_stats.write(data_provider.data_files[modelPerf_df.idxmin()][len(test_data_path):])
	f_stats.write("\nAverage Dice Score: %1.3f  +- %2.3f" % (modelPerf_df.mean(), modelPerf_df.std()))
	f_stats.write("\n\nDetails:\n")
	f_stats.write("Name                  pred_time      dicePerf\n")
	for i in range(len(data_provider.data_files)):
		f_stats.write(data_provider.data_files[i][len(test_data_path):])
		f_stats.write("         %1.3f s        %2.3f\n" % (pred_time_df[i], modelPerf_df[i]))

	f_stats.close()

	modelPerf_df.to_csv('Training_Results/' + model_name + '_stats_df', index=False, header=False)


#####     Update name for each training     #####
MODEL_NAME = 'FZ_100E_cross_300img_w2_1'
TRAIN_DIR_NAME = 'E:\Project Altran\Results/'
ROOT_DIR = os.path.abspath(os.curdir)
TRAIN_DIR = os.path.join(ROOT_DIR, TRAIN_DIR_NAME)
TRAIN_PATH = os.path.join(TRAIN_DIR, MODEL_NAME)

TEST_DATA_PATH = 'orniere_data/AUG_TEST_540p/'
tensorflow_shutup()
predict(model_name=MODEL_NAME, test_data_path=TEST_DATA_PATH, train_path=TRAIN_PATH, data_suffix='.png')
