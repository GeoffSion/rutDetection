###################### rut_detection ######################
# This script is the main function that creates the U-net #
# calls the trainer, and makes the predictions.           #
###########################################################

from __future__ import division, print_function
from visualize_logs import sort_out_log_file, data_visualization
from model_test import predict
from datetime import timedelta
import os
import tensorflow as tf
from tf_unet import unet
from tf_unet import image_util
import time

# np.random.seed(98765)
tf.logging.set_verbosity(tf.logging.ERROR)

timedatetime
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


def create_tree(directory_name):
	if not os.path.exists(directory_name):
		print('Creating directory tree...')
		os.makedirs(directory_name)
		os.makedirs(os.path.join(directory_name, "Predictions"))


tensorflow_shutup()

#####     Names and parameters     #####
MODEL_NAME = ['contour_100E_w1_15',
              'FZ_300i_1080p',
              'FZ_300i_540p_drop50']

TRAIN_PATH = ['orniere_data/contour_data/train_540p/',
              'orniere_data/AUG_TRAIN/',
              'orniere_data/AUG_TRAIN_540p/']

TEST_PATH = ['orniere_data/contour_data/test_540p/',
             'orniere_data/AUG_TEST_540p/',
             'orniere_data/AUG_TEST_540p/']

BATCH_SIZE = [4, 4, 4]
VAL_BATCH_SIZE = [3, 3, 3]
NBR_EPOCH = [2, 2, 2]
NBR_ITER = [78, 78, 78]
CLASS_WEIGHTS = [[1, 15], [1, 1], [1, 1]]
DROPOUT = [0.75, 0.75, 0.50]
COST_FUNCTION = ["cross_entropy", "cross_entropy", "cross_entropy"]  # "BCEDICE" or "cross_entropy"
TRAIN_DIR_NAME = 'Training_Results'
ROOT_DIR = os.path.abspath(os.curdir)
TRAIN_DIR = os.path.join(ROOT_DIR, TRAIN_DIR_NAME)
DATA_TYPE = ".png"

for modelName, trainPath, testPath, batchSize, valBatchSize, nbrEpoch, nbrIter, costFunction, class_weights, dropout in zip(
		MODEL_NAME, TRAIN_PATH, TEST_PATH, BATCH_SIZE, VAL_BATCH_SIZE, NBR_EPOCH, NBR_ITER, COST_FUNCTION,
		CLASS_WEIGHTS, DROPOUT):
	print('\n', modelName, 'model')
	TRAIN_PATH = os.path.join(TRAIN_DIR, modelName)
	create_tree(TRAIN_PATH)

	log = open(TRAIN_PATH + "/logs_" + modelName + ".txt", "w+")
	log.write(str(modelName) + '\n')
	log.write("Training Images: ")
	log.write(str(len(os.listdir(trainPath))) + '\n')
	log.write('Cost function: ')
	log.write(str(costFunction) + '\n')
	log.write('Class weights: ')
	log.write(str(class_weights) + '\n')
	log.close()

	total_time_start = time.time()
	### GET TRAIN AND VAL IMAGES ###
	data_provider = image_util.ImageDataProvider(search_path=trainPath + "*" + DATA_TYPE, data_suffix=DATA_TYPE,
	                                             mask_suffix="_mask" + DATA_TYPE, shuffle_data='True', n_class=2)

	### START TRAINING ###
	start = time.time()
	with tf.device('/cpu:0'):
		# with tf.device('/device:GPU:0'):
		print("n_class:", data_provider.n_class)
		net = unet.Unet(channels=data_provider.channels, n_class=data_provider.n_class, layers=3, features_root=16,
		                cost=costFunction, cost_kwargs=dict(class_weights=class_weights))
		trainer = unet.Trainer(net, batch_size=batchSize, verification_batch_size=valBatchSize, optimizer="momentum",
		                       opt_kwargs=dict(momentum=0.2))
		print('Training...')
		path = trainer.train(data_provider, TRAIN_PATH, training_iters=nbrIter, epochs=nbrEpoch,
		                     display_step=1, prediction_path=os.path.join(TRAIN_PATH, "Predictions"),
		                     dropout=dropout, log_file="logs_" + modelName + ".txt")

	### END TRAINING ###
	end = time.time()
	print('Model done')
	training_time = timedelta(seconds=end - start)
	print("Processing time: ", training_time)
	log = open(TRAIN_PATH + "/logs_" + modelName + ".txt", "a+")
	log.write("Processing time: ")
	log.write(str(training_time))
	log.close()

	### PLOT TRAINING DATA ###
	data_to_plot = sort_out_log_file(modelName, TRAIN_PATH, nbrIter)
	data_visualization(data=data_to_plot, epochs=nbrEpoch, model_name=modelName, train_path=TRAIN_PATH)

	### TEST MODEL ###
	predict(model_name=modelName, test_data_path=testPath, train_path=TRAIN_PATH, data_suffix=DATA_TYPE)
	total_time_end = time.time()
	total_time = timedelta(seconds=total_time_end - total_time_start)
	print('Training and testing of the ' + modelName + ' model were successfully achieved')
	print('Total time: ', total_time)
