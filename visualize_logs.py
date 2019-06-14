###################### visualize_logs ######################
# This script aims to put the console output of a training #
# from txt to dataframe and then plot the metrics:         #
# Epoch average loss, learning rate, ver_error and loss    #
# Image loss, training accuracy, and Image error           #
############################################################

import csv
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def data_visualization(data, epochs, model_name, train_path):

	print('Creating training curves...')
	############ ACCURACY ############
	plt.figure(1)
	plt.plot(range(epochs), data['Train_acc'], '-o', color='darkorange', label='Training acc',
	         alpha=0.5)
	plt.plot(range(epochs), data['Val_acc'], '-o', color='darkblue', label='Validation acc', alpha=0.5)
	# plt.plot(epochs, test_acc, '-o', color ='m', label='Testing acc', alpha=0.5)
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.title('Training and validation accuracy')
	plt.grid(linestyle='-', linewidth=1, alpha=0.5)
	plt.legend()
	# plt.savefig(train_path + '/' + model_name + '_acc.jpg', dpi=300, bbox_inches='tight')
	plt.close()

	############ LOSS ############
	plt.figure(2)
	plt.plot(range(epochs), data['Train_loss'], '-o', color='darkorange', label='Training loss', alpha=0.5)
	plt.plot(range(epochs), data['Val_loss'], '-o', color='darkblue', label='Validation loss', alpha=0.5)
	# plt.plot(epochs, test_loss, '-o', color ='m', label='Testing loss', alpha=0.5)
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Training and validation loss')
	plt.grid(linestyle='-', linewidth=1, alpha=0.7)
	plt.legend()
	# plt.savefig(train_path + '/' + model_name + '_loss.jpg', dpi=300, bbox_inches='tight')
	plt.close()

	############ BOTH ############
	fig = plt.figure(figsize=(10, 7))
	ax = fig.add_subplot(211)
	ax.plot(range(epochs), data['Train_acc'], '-o', color='darkorange', label='Training acc',
	        alpha=0.5)  # Divide by the number of iterations (batches)
	ax.plot(range(epochs), data['Val_acc'], '-o', color='darkblue', label='Validation acc', alpha=0.5)
	# ax.plot(epochs, test_acc, '-o', color ='m', label='Testing acc', alpha=0.5)*
	plt.ylabel('Accuracy')
	plt.title('Training and validation accuracy')
	plt.grid(linestyle='-', linewidth=1, alpha=0.5)
	plt.legend()

	ax2 = fig.add_subplot(212)
	ax2.plot(range(epochs), data['Train_loss'], '-o', color='darkorange', label='Training loss', alpha=0.5)
	ax2.plot(range(epochs), data['Val_loss'], '-o', color='darkblue', label='Validation loss', alpha=0.5)
	# ax2.plot(epochs, test_loss, '-o', color ='m', label='Testing loss', alpha=0.5)
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Training and validation loss')
	plt.grid(linestyle='-', linewidth=1, alpha=0.7)
	plt.legend()
	plt.savefig(train_path + '/' + model_name + '_acc_loss.jpg', dpi=300, bbox_inches='tight')

	plt.close('all')
	plt.show()


def sort_out_log_file(model_name, train_path, nbr_iter):
	print('Sorting out training stats...')
	f = open(os.path.join(train_path, "logs_" + model_name + ".txt"), "r")
	f_epoch = open(os.path.join(train_path, "epoch.txt"), "w+")
	f_verif = open(os.path.join(train_path, "verif.txt"), "w+")
	f_iter = open(os.path.join(train_path, "iter.txt"), "w+")
	f_test = open(os.path.join(train_path, "test.txt"), "w+")

	#### In log txt file f, find Epoch line and Iteration line and put them in separate files
	for line in f:
		if (line.find('Epoch') != -1):
			f_epoch.write(line[line.find('Epoch'):])
		elif (line.find('Verification') != -1):
			f_verif.write(line[line.find('Verification'):])
		elif (line.find('Iter') != -1):
			f_iter.write(line[line.find('Iter'):])
		elif (line.find('Test') != -1):
			f_test.write(line[line.find('Test'):])

	f.close()
	f_epoch.close()
	f_verif.close()
	f_iter.close()
	f_test.close()

	# Open each txt file and store the data in pandas DataFrames
	f_epoch = open(os.path.join(train_path, "epoch.txt"), "r+")
	f_verif = open(os.path.join(train_path, "verif.txt"), "r+")
	f_iter = open(os.path.join(train_path, "iter.txt"), "r+")
	f_test = open(os.path.join(train_path, "test.txt"), "r+")

	# Epoch logs
	epoch_df = pd.DataFrame(columns=['Epoch', 'Average Loss', 'Learning Rate'])  # Create epoch dataframe
	epoch_df = epoch_df.append({"Epoch": -1,  # Create tmp dataframe (row with epoch data)
	                            "Average Loss": np.NaN,
	                            "Learning Rate": np.NaN},
	                           ignore_index=True, sort=False)
	for line_epoch in f_epoch:  # Read each line
		epoch_data = re.findall(r"[-+]?\d*\.\d+|\d+", line_epoch)  # Find all the floating numbers
		df_tmp = pd.DataFrame({"Epoch": [epoch_data[0]],  # Create tmp dataframe (row with epoch data)
		                       "Average Loss": [epoch_data[1]],
		                       "Learning Rate": [epoch_data[2]]})
		epoch_df = epoch_df.append(df_tmp, ignore_index=True, sort=False)  # Add this row to the global dataframe

	# Verif logs
	verif_df = pd.DataFrame(columns=['Verification Error', 'Verification Loss'])  # Create verif dataframe
	for line_verif in f_verif:  # Read each line
		verif_data = re.findall(r"[-+]?\d*\.\d+|\d+", line_verif)  # Find all the floating numbers
		df_tmp = pd.DataFrame({"Verification Error": [verif_data[0]],  # Create tmp dataframe (row with verif data)
		                       "Verification Loss": [verif_data[1]]})
		verif_df = verif_df.append(df_tmp, ignore_index=True, sort=False)  # Add this row to the global dataframe

	# Merge verif with epoch
	epoch_df = pd.concat([epoch_df, verif_df], axis=1)
	epoch_df = epoch_df.astype('float')
	epoch_df.set_index('Epoch', inplace=True)
	# print('New epoch_df: ', epoch_df)
	# epoch_df.to_csv(os.path.join(model_name,'epoch_logs.csv'))               # Save epoch logs in csv file

	# Iter logs
	iter_df = pd.DataFrame(columns=['Iter', 'Image Loss', 'Training Accuracy', 'Image Error'])  # Create iter dataframe
	for line_iter in f_iter:  # Read each line
		iter_data = re.findall(r"[-+]?\d*\.\d+|\d+", line_iter)  # Find all the floating numbers
		df_tmp = pd.DataFrame({"Iter": [iter_data[0]],  # Create tmp dataframe (row with iter data)
		                       "Image Loss": [iter_data[1]],
		                       "Training Accuracy": [iter_data[2]],
		                       "Image Error": [iter_data[3]]})
		iter_df = iter_df.append(df_tmp, ignore_index=True, sort=False)  # Add this row to the global dataframe

	iter_df = iter_df.astype('float')

	# Test logs
	test_df = pd.DataFrame(columns=['Testing Error', 'Testing Loss'])  # Create verif dataframe
	for line_test in f_test:  # Read each line
		test_data = re.findall(r"[-+]?\d*\.\d+|\d+", line_test)  # Find all the floating numbers
		df_tmp = pd.DataFrame({"Testing Error": [test_data[0]],  # Create tmp dataframe (row with verif data)
		                       "Testing Loss": [test_data[1]]})
		test_df = test_df.append(df_tmp, ignore_index=True, sort=False)  # Add this row to the global dataframe

	test_df = test_df.astype('float')

	# do mean of iterations for each epochs
	iter_df_mean = pd.DataFrame(columns=['Image Loss', 'Training Accuracy', 'Image Error'])
	for i in range(0, len(iter_df.index), nbr_iter):
		df_tmp = pd.DataFrame({"Image Loss": [iter_df["Image Loss"].iloc[[i, i + nbr_iter - 1]].mean(axis=0)],
		                       "Training Accuracy": [
			                       iter_df["Training Accuracy"].iloc[[i, i + nbr_iter - 1]].mean(axis=0)],
		                       "Image Error": [iter_df["Image Error"].iloc[[i, i + nbr_iter - 1]].mean(axis=0)]})
		iter_df_mean = iter_df_mean.append(df_tmp, ignore_index=True, sort=False)
	# print("i: ", i, "i+it: ", i+nbr_iter)

	# print(iter_df_mean)

	# print(iter_df)
	# iter_df.to_csv(os.path.join(model_name,'iter_logs.csv'))               # Save iter logs in csv file

	f_epoch.close()
	f_verif.close()
	f_iter.close()
	epoch_df = epoch_df.iloc[1:]


	train_acc = iter_df_mean['Training Accuracy']
	val_acc = (100 - epoch_df['Verification Error']) / 100
	test_acc = (100 - test_df['Testing Error']) / 100
	train_loss = iter_df_mean['Image Loss']
	val_loss = epoch_df['Verification Loss']
	test_loss = test_df['Testing Loss']
	epochs = range(0, len(epoch_df.index))
	all_data = pd.DataFrame({'Train_acc': train_acc, 'Val_acc': val_acc, 'Test_acc': test_acc, 'Train_loss': train_loss,
	                         'Val_loss': val_loss, 'Test_loss': test_loss})
	all_data.to_csv(train_path + '/' + model_name + '_df', index=False)  # Save in model folder

	return all_data


MODEL_NAME = 'contour_100E_w1_30'
TRAIN_DIR_NAME = 'Training_Results/'
NBR_EPOCHS = 120
NBR_ITER = 78
TRAIN_PATH = os.path.join(TRAIN_DIR_NAME, MODEL_NAME)

data = sort_out_log_file(MODEL_NAME, TRAIN_PATH, NBR_ITER)

data_visualization(data, epochs=NBR_EPOCHS, model_name=MODEL_NAME, train_path=TRAIN_PATH)

