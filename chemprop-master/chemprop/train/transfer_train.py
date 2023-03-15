from argparse import Namespace
import csv
from logging import Logger
import os
from pprint import pformat
from typing import List, Optional

import numpy as np
from tensorboardX import SummaryWriter
import torch
from tqdm import trange, tqdm
import pickle
from torch.optim.lr_scheduler import ExponentialLR

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from .run_tuning import run_tuning
from chemprop.data import StandardScaler
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data, get_data_from_smiles
from chemprop.models import build_model
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
	makedirs, save_checkpoint, load_args, load_checkpoint, load_scalers

from .predict import predict
from chemprop.data import MoleculeDataset

def run_transfer_training(checkpoint_dir, tune_data_path, test_data_path, lr_ratio = 0.1, layer_list = [2]*100, logger: Logger = None, use_formulations = False):
	"""
	Does transfer learning: trains on a new test set specifically by fine-tuning a model with a small learning rate (and freezing initial few rows?).

	:param args: Arguments.
	:param logger: Logger.
	:return: A list of ensemble scores for each task.
	"""

	# model = load_checkpoint(checkpoint_path, cuda=args.cuda)
	epochs = len(layer_list)
	print('Loading training args')
	checkpoint_path = checkpoint_dir + '/fold_0/model_0/model.pt'
	scaler, features_scaler = load_scalers(checkpoint_path)
	train_args = load_args(checkpoint_path)
	print(train_args)
	print('\n\n\n')

	train_args.save_dir = checkpoint_dir + '/Tuned_model'
	train_args.data_path = tune_data_path
	train_args.separate_val_path = None
	train_args.separate_test_path = test_data_path
	train_args.epochs = epochs

	if use_formulations:
		train_args.features_path = [tune_data_path[:-4] + '_formulations.csv']
		train_args.separate_val_path = None
		train_args.separate_test_features_path = [test_data_path[:-4] + '_formulations.csv']
	# train_args.features_scaling = False
	# train_args.use_input_features = ['morgan_count', 'rdkit_2d']
	# train_args.features_generator = ['morgan_count', 'rdkit_2d']
	# train_args.save_smiles_splits = True

	# Update args with training arguments
	# for key, value in vars(train_args).items():
	# 	if not hasattr(predict_args, key):
	# 		setattr(predict_args, key, value)

	
	model = load_checkpoint(checkpoint_path)
	# Set learning rate low
	# Successively train different layers
	print(model)
	print(model.classification)
	run_tuning(model, train_args, lr_ratio = lr_ratio, layer_list = layer_list)