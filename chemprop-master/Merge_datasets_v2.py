import numpy as np 
import os
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score
from train_multitask import train_multitask_model, get_base_args, optimize_hyperparameters, train_hyperparam_optimized_model
from predict_multitask_from_json import predict_multitask_from_json, get_base_predict_args, predict_multitask_from_json_cv
from rdkit import Chem
import matplotlib.pyplot as plt
import scipy.stats
import json
import sys
import random


# Run conda activate chemprop before running this
# To copy a file over to luria, use a command like:
 # scp Merge_datasets_v2.py jswitten@luria:/home/jswitten/data/chemprop-master/Merge_datasets_v2.py
# jswitten@luria:/home/jswitten/data/chemprop-master/Data/Multitask_data/All_datasets/crossval_splits/all_amine_cv/in_silico_screen_results/cooh_linker_4cr_k_KK_hela.csv
# /Users/jacobwitten/Documents/Next_steps/Anderson/"ML for NP design"/Chemprop/chemprop-master/Data/Multitask_data/All_datasets/crossval_splits/all_amine_cv/screen_results
# For interactive run:
# srun --pty bash
# module load python3/3.6.4


def merge_datasets(experiment_list, path_to_folders = 'Data/Multitask_data/All_datasets'):
	# Each folder contains the following files: 
	# main_data.csv: a csv file with columns: 'smiles', which should contain the SMILES of the ionizable lipid, and any number of activity measurements for that measurement
	# formulations.csv: a csv file with columns:
		# Cationic_Lipid_Mol_Ratio
		# Phospholipid_Mol_Ratio
		# Cholesterol_Mol_Ratio
		# PEG_Lipid_mol_ratio
		# Cationic_Lipid_to_mRNA_weight_ratio
		# Helper_lipid_ID
		# If the dataset contains only 1 formulation in it: still provide the formulations data thing but with only one row; the model will copy it
		# Otherwise match the row to the data in formulations.csv
	# individual_metadata.csv: metadata that contains as many rows as main_data, each row is certain metadata for each lipid
		# For example, could contain the identity (SMILES) of the amine to be used in training/test splits, or contain a dosage if the dataset includes varying dosage
		# Either includes a column called "Sample_weight" with weight for each sample (each ROW, that is; weight for a kind of experiment will be determined separately)
			# alternatively, default sample weight of 1
	# experiment_metadata.csv: contains metadata about particular dataset. This includes:
		# Experiment_ID: each experiment will be given a unique ID.
		# There will be two ROWS and any number of columns

	# Based on these files, Merge_datasets will merge all the datasets into one dataset. In particular, it will output 2 files:
		# all_merged.csv: each row  will contain all the data for a measurement (SMILES, info on dose/formulation/etc, metadata, sample weights, activity value)
		# col_type.csv: two columns, column name and type. Four types: Y_val, X_val, X_val_cat (categorical X value), Metadata, Sample_weight

	# Some metadata columns that should be held consistent, in terms of names:
		# Purity ("Pure" or "Crude")
		# ng_dose (for the dose, duh)
		# Sample_weight
		# Amine_SMILES
		# Tail_SMILES
		# Library_ID
		# Experimenter_ID
		# Experiment_ID
		# Cargo (siRNA, DNA, mRNA, RNP are probably the relevant 4 options)
		# Model_type (either the cell type or the name of the animal (probably "mouse"))


	all_df = pd.DataFrame({})
	col_type = {'Column_name':[],'Type':[]}
	experiment_df = pd.read_csv(path_to_folders + '/experiment_metadata.csv')
	if experiment_list == None:
		experiment_list = list(experiment_df.Experiment_ID)
		print(experiment_list)
	y_val_cols = []
	helper_mol_weights = pd.read_csv(path_to_folders + '/Component_molecular_weights.csv')

	for folder in experiment_list:
		print(folder)
		contin = False
		try:
			main_temp = pd.read_csv(path_to_folders + '/' + folder + '/main_data.csv')
			contin = True
		except:
			pass
		if contin:
			y_val_cols = y_val_cols + list(main_temp.columns)
			data_n = len(main_temp)
			formulation_temp = pd.read_csv(path_to_folders + '/' + folder + '/formulations.csv')

			try:
				individual_temp = pd.read_csv(path_to_folders + '/' + folder + '/individual_metadata.csv')
			except:
				individual_temp = pd.DataFrame({})
			if len(formulation_temp) == 1:
				formulation_temp = pd.concat([formulation_temp]*data_n,ignore_index = True)
			elif len(formulation_temp) != data_n:
				print(len(formulation_temp))
				to_raise = 'For experiment ID: ',folder,': Length of formulation file (', str(len(formulation_temp))#, ') doesn\'t match length of main datafile (',str(data_n),')'
				raise ValueError(to_raise)

			# Change formulations from mass to molar ratio
			form_cols = formulation_temp.columns
			mass_ratio_variables = ['Cationic_Lipid_Mass_Ratio','Phospholipid_Mass_Ratio','Cholesterol_Mass_Ratio','PEG_Lipid_Mass_Ratio']
			molar_ratio_variables = ['Cationic_Lipid_Mol_Ratio','Phospholipid_Mol_Ratio','Cholesterol_Mol_Ratio','PEG_Lipid_Mol_Ratio']
			mass_count = 0
			molar_count = 0
			for col in form_cols:
				if col in mass_ratio_variables:
					mass_count += 1
				elif col in molar_ratio_variables:
					molar_count += 1
			if mass_count>0 and molar_count>0:
				raise ValueError('For experiment ID: ',folder,': Formulation information includes both mass and molar ratios.')
			elif mass_count<4 and molar_count<4:
				raise ValueError('For experiment ID: ',folder,': Formulation not completely specified, mass count: ',mass_count,', molar count: ',molar_count)
			elif mass_count == 4:
				cat_lip_mol_fracs = []
				phos_mol_fracs = []
				chol_mol_fracs = []
				peg_lip_mol_fracs = []
				# Change mass ratios to weight ratios
				for i in range(len(formulation_temp)):
					phos_id = formulation_temp['Helper_lipid_ID'][i]
					ion_lipid_mol = Chem.MolFromSmiles(main_temp['smiles'][i])
					ion_lipid_mol_weight = Chem.Descriptors.MolWt(ion_lipid_mol)
					phospholipid_mol_weight = helper_mol_weights[phos_id][0]
					cholesterol_mol_weight = helper_mol_weights['Cholesterol']
					PEG_lipid_mol_weight = helper_mol_weights['C14-PEG2000']
					ion_lipid_moles = formulation_temp['Cationic_Lipid_Mass_Ratio'][i]/ion_lipid_mol_weight
					phospholipid_moles = formulation_temp['Phospholipid_Mass_Ratio'][i]/phospholipid_mol_weight
					cholesterol_moles = formulation_temp['Cholesterol_Mass_Ratio'][i]/cholesterol_mol_weight
					PEG_lipid_moles = formulation_temp['PEG_Lipid_Mass_Ratio'][i]/PEG_lipid_mol_weight
					mol_sum = ion_lipid_moles+phospholipid_moles+cholesterol_moles+PEG_lipid_moles
					cat_lip_mol_fracs.append(float(ion_lipid_moles/mol_sum*100))
					phos_mol_fracs.append(float(phospholipid_moles/mol_sum*100))
					chol_mol_fracs.append(float(cholesterol_moles/mol_sum*100))
					peg_lip_mol_fracs.append(float(PEG_lipid_moles/mol_sum*100))
				formulation_temp['Cationic_Lipid_Mol_Ratio'] = cat_lip_mol_fracs
				formulation_temp['Phospholipid_Mol_Ratio'] = phos_mol_fracs
				formulation_temp['Cholesterol_Mol_Ratio'] = chol_mol_fracs
				formulation_temp['PEG_Lipid_Mol_Ratio'] = peg_lip_mol_fracs

		
			if len(individual_temp) != data_n:
				print(len(individual_temp))
				raise ValueError('For experiment ID: ',folder,': Length of individual metadata file  (',len(individual_temp), ') doesn\'t match length of main datafile (',data_n,')')
			experiment_temp = experiment_df[experiment_df.Experiment_ID == folder]
			# print(folder)
			# print(experiment_temp)
			experiment_temp = pd.concat([experiment_temp]*data_n, ignore_index = True).reset_index(drop = True)
			to_drop = []
			for col in experiment_temp.columns:
				if col in individual_temp.columns:
					print('Column ',col,' in experiment ID ',folder,'is being provided for each individual lipid.')
					to_drop.append(col)
			experiment_temp = experiment_temp.drop(columns = to_drop)
			# print(folder)
			# print(experiment_temp.columns)
			# print(main_temp.columns)
			# print(formulation_temp.columns)
			# print(individual_temp.columns)
			folder_df = pd.concat([main_temp, formulation_temp, individual_temp], axis = 1).reset_index(drop = True)
			folder_df = pd.concat([folder_df, experiment_temp], axis = 1)
			# print(folder_df.columns)
			if 'Sample_weight' not in folder_df.columns:
				# print(folder)
				# folder_df['Sample_weight'] = [float(folder_df.Experiment_weight[i])/list(folder_df.smiles).count(smile) for i,smile in enumerate(folder_df.smiles)]
				folder_df['Sample_weight'] = [float(folder_df.Experiment_weight[i]) for i,smile in enumerate(folder_df.smiles)]

				# print(folder_df.Experiment_weight)
				# print(folder_df['Sample_weight'])
			# print (folder_df.columns)
			all_df = pd.concat([all_df,folder_df], ignore_index = True)

	# Make the column type dict
	extra_x_variables = ['Cationic_Lipid_Mol_Ratio','Phospholipid_Mol_Ratio','Cholesterol_Mol_Ratio','PEG_Lipid_Mol_Ratio','Cationic_Lipid_to_mRNA_weight_ratio']
	# ADD HELPER LIPID ID
	# extra_x_categorical = ['Delivery_target','Helper_lipid_ID','Route_of_administration','Batch_or_individual_or_barcoded','screen_id']
	extra_x_categorical = ['Delivery_target','Helper_lipid_ID','Route_of_administration','Batch_or_individual_or_barcoded','Cargo_type','Model_type']

	# Make changes:
	all_df = all_df.replace('im','intramuscular')
	all_df = all_df.replace('iv','intravenous')
	all_df = all_df.replace('a549','lung')
	all_df = all_df.replace('bdmc','macrophage')
	all_df = all_df.replace('bmdm','dendritic_cell')
	all_df = all_df.replace('hela','generic_cell')
	all_df = all_df.replace('hek','generic_cell')
	all_df = all_df.replace('igrov1','generic_cell')
	all_df = all_df.replace({'Model_type':'muscle'},'Mouse')

	# other_x_vals = ['Target_organ']
	# form_variables.append('Helper_lipid_ID')

	for x_cat in extra_x_categorical:
		dummies = pd.get_dummies(all_df[x_cat], prefix = x_cat)
		print(dummies.columns)
		all_df = pd.concat([all_df, dummies], axis = 1)
		extra_x_variables = extra_x_variables + list(dummies.columns)

	for column in all_df.columns:
		col_type['Column_name'].append(column)
		if column in y_val_cols:
			col_type['Type'].append('Y_val')
		elif column in extra_x_variables:
			col_type['Type'].append('X_val')
		elif column in extra_x_categorical:
			col_type['Type'].append('Metadata')
		elif column == 'Sample_weight':
			col_type['Type'].append('Sample_weight')
		else:
			col_type['Type'].append('Metadata')

	col_type_df = pd.DataFrame(col_type)
	# print(col_type_df)
	norm_split_names, norm_del = generate_normalized_data(all_df)
	all_df['split_name_for_normalization'] = norm_split_names
	all_df.rename(columns = {'quantified_delivery':'unnormalized_delivery'}, inplace = True)
	all_df['quantified_delivery'] = norm_del
	all_df.to_csv(path_to_folders + '/all_data.csv', index = False)
	col_type_df.to_csv(path_to_folders + '/col_type.csv', index = False)
	# TODO: allow for no "individual_metadata" file
	# Add the variable classification

def split_df_by_col_type(df,col_types):
	# Splits into 4 dataframes: y_vals, x_vals, sample_weights, metadata
	y_vals_cols = col_types.Column_name[col_types.Type == 'Y_val']
	x_vals_cols = col_types.Column_name[col_types.Type == 'X_val']
	# print(x_vals_cols)
	# print(df.columns)
	xvals_df = df[x_vals_cols]
	weight_cols = col_types.Column_name[col_types.Type == 'Sample_weight']
	metadata_cols = col_types.Column_name[col_types.Type.isin(['Metadata','X_val_categorical'])]
	# x_val_cat_cols = list(col_types.Column_name[col_types.Type == 'X_val_categorical'])
	# xvals_df = pd.concat(([xvals_df] + [pd.get_dummies(df[col]) for col in x_val_cat_cols]),axis = 1)

	return df[y_vals_cols],xvals_df,df[weight_cols],df[metadata_cols]

def do_all_splits(path_to_splits = 'Data/Multitask_data/All_datasets/Split_specs'):
	all_csvs = os.listdir(path_to_splits)
	for csv in all_csvs:
		if csv.endswith('.csv'):
			specified_dataset_split(csv)

def train_valid_test_split(vals, train_frac, valid_frac, test_frac, random_state = 42):
	# only works for list inputs
	if train_frac + valid_frac + test_frac > 99:
		train_frac = float(train_frac)/100
		valid_frac = float(valid_frac)/100
		test_frac = float(test_frac)/100
	if abs(train_frac + valid_frac + test_frac-1)>0.01:
		raise ValueError('Sum of train, valid, test fractions is not 1! It\'s: ',train_frac + valid_frac + test_frac)
	if test_frac>0 and test_frac < 1:
		train, test = train_test_split(vals, test_size = test_frac, random_state = random_state)
	elif test_frac == 1:
		test = vals
		train = []
	else:
		train = vals
		test = []
	if valid_frac > 0 and valid_frac < 1:
		train, valid = train_test_split(train, test_size = valid_frac/(train_frac+valid_frac), random_state = random_state*2)
	elif valid_frac == 0:
		valid = []
	else:
		valid = train
		train = []
	return train, valid, test


def split_for_cv(vals,cv_fold, held_out_fraction):
	random.shuffle(vals)
	held_out_vals = vals[:int(held_out_fraction*len(vals))]
	cv_vals = vals[int(held_out_fraction*len(vals)):]
	return [cv_vals[i::cv_fold] for i in range(cv_fold)],held_out_vals

def specified_cv_split(split_spec_fname, path_to_folders = 'Data/Multitask_data/All_datasets', is_morgan = False, cv_fold = 5, ultra_held_out_fraction = -1.0, min_unique_vals = 2.0):
	# cv_fold: self-explanatory
	# ultra_held_out_fraction: if you want to hold a dataset out from even the cross-validation datasets this is theway to do it
	all_df = pd.read_csv(path_to_folders + '/all_data.csv')
	split_df = pd.read_csv(path_to_folders+'/crossval_split_specs/'+split_spec_fname)
	split_path = path_to_folders + '/crossval_splits/' + split_spec_fname[:-4]
	if ultra_held_out_fraction>-0.5:
		split_path = split_path + '_with_ultra_held_out'
	if is_morgan:
		split_path = split_path + '_morgan'
	if ultra_held_out_fraction>-0.5:
		path_if_none(split_path + '/ultra_held_out')
	for i in range(cv_fold):
		path_if_none(split_path+'/cv_'+str(i))

	perma_train = pd.DataFrame({})
	ultra_held_out = pd.DataFrame({})
	cv_splits = [pd.DataFrame({}) for _ in range(cv_fold)]

	for index, row in split_df.iterrows():
		dtypes = row['Data_types_for_component'].split(',')
		vals = row['Values'].split(',')
		df_to_concat = all_df
		for i, dtype in enumerate(dtypes):
			df_to_concat = df_to_concat[df_to_concat[dtype.strip()]==vals[i].strip()].reset_index(drop = True)
		values_to_split = df_to_concat[row['Data_type_for_split']]
		unique_values_to_split = list(set(values_to_split))
		# print(row)
		if row['Train_or_split'].lower() == 'train' or len(unique_values_to_split)<min_unique_vals*cv_fold:
			perma_train = pd.concat([perma_train, df_to_concat])
		elif row['Train_or_split'].lower() == 'split':
			cv_split_values, ultra_held_out_values = split_for_cv(unique_values_to_split, cv_fold, ultra_held_out_fraction)
			to_concat = df_to_concat[df_to_concat[row['Data_type_for_split']].isin(ultra_held_out_values)]
			# print('Type: ',type(to_concat))
			# print('Ultra held out type: ',type(ultra_held_out))
			ultra_held_out = pd.concat([ultra_held_out, to_concat])
			for i, val in enumerate(cv_split_values):
				cv_splits[i] = pd.concat([cv_splits[i], df_to_concat[df_to_concat[row['Data_type_for_split']].isin(val)]])

	col_types = pd.read_csv(path_to_folders + '/col_type.csv')

	# Now move the dfs to datafiles
	if ultra_held_out_fraction>-0.5:
		y,x,w,m = split_df_by_col_type(ultra_held_out,col_types)
		yxwm_to_csvs(y,x,w,m,split_path+'/ultra_held_out','test')

	for i in range(cv_fold):
		test_df = cv_splits[i]
		valid_df = cv_splits[(i+1)%cv_fold]
		train_inds = list(range(cv_fold))
		train_inds.remove(i)
		train_inds.remove((i+1)%cv_fold)
		train_df = pd.concat([perma_train]+[cv_splits[k] for k in train_inds])

		y,x,w,m = split_df_by_col_type(test_df,col_types)
		yxwm_to_csvs(y,x,w,m,split_path+'/cv_'+str(i),'test')
		y,x,w,m = split_df_by_col_type(valid_df,col_types)
		yxwm_to_csvs(y,x,w,m,split_path+'/cv_'+str(i),'valid')
		y,x,w,m = split_df_by_col_type(train_df,col_types)
		yxwm_to_csvs(y,x,w,m,split_path+'/cv_'+str(i),'train')

	# settype = 'train'
	# y_vals.to_csv(path_to_splits + '/' + settype + '.csv', index = False)
	# x_vals.to_csv(path_to_splits + '/' + settype + '_extra_x.csv', index = False)
	# weights.to_csv(path_to_splits + '/' + settype + '_weights.csv', index = False)
	# metadata_cols.to_csv(path_to_splits + '/' + settype + '_metadata.csv', index = False)

	# settype = 'valid'
	# y_vals_v.to_csv(path_to_splits + '/' + settype + '.csv', index = False)
	# x_vals_v.to_csv(path_to_splits + '/' + settype + '_extra_x.csv', index = False)
	# weights_v.to_csv(path_to_splits + '/' + settype + '_weights.csv', index = False)
	# metadata_cols_v.to_csv(path_to_splits + '/' + settype + '_metadata.csv', index = False)

	# y_vals,x_vals,weights,metadata_cols = split_df_by_col_type(test_df,col_types)
	# settype = 'test'
	# y_vals.to_csv(path_to_splits + '/' + settype + '.csv', index = False)
	# x_vals.to_csv(path_to_splits + '/' + settype + '_extra_x.csv', index = False)
	# weights.to_csv(path_to_splits + '/' + settype + '_weights.csv', index = False)
	# metadata_cols.to_csv(path_to_splits + '/' + settype + '_metadata.csv', index = False)

def yxwm_to_csvs(y, x, w, m, path,settype):
	# y is y values
	# x is x values
	# w is weights
	# m is metadata
	# set_type is either train, valid, or test
	y.to_csv(path+'/'+settype+'.csv', index = False)
	x.to_csv(path + '/' + settype + '_extra_x.csv', index = False)
	w.to_csv(path + '/' + settype + '_weights.csv', index = False)
	m.to_csv(path + '/' + settype + '_metadata.csv', index = False)


def specified_dataset_split(split_spec_fname, path_to_folders = 'Data/Multitask_data/All_datasets', is_morgan = False):
	# 3 columns: Data_type, Value, Split_type
	all_df = pd.read_csv(path_to_folders + '/all_data.csv')
	split_df = pd.read_csv(path_to_folders + '/Split_specs/' + split_spec_fname)
	split_path = path_to_folders + '/Splits/' + split_spec_fname[:-4]
	if is_morgan:
		split_path = split_path + '_morgan'
	path_if_none(split_path)
	train_df = pd.DataFrame({})
	valid_df = pd.DataFrame({})
	test_df = pd.DataFrame({})
	# print(all_df.columns)
	for index,row in split_df.iterrows():
		# print(all_df.columns)
		print(row)
		# print(row['Data_type_for_component'])
		dtypes = row['Data_types_for_component'].split(',')
		vals = row['Values'].split(',')
		df_to_concat = all_df
		for i, dtype in enumerate(dtypes):
			print(len(df_to_concat))
			df_to_concat = df_to_concat[df_to_concat[dtype.strip()]==vals[i].strip()].reset_index(drop = True)
		print(len(df_to_concat))
		# if row['Data_type_for_split'] == 'random':

		values_to_split = df_to_concat[row['Data_type_for_split']]
		unique_values_to_split = list(set(values_to_split))
		train_frac = float(row['Percent_train'])/100
		valid_frac = float(row['Percent_valid'])/100
		test_frac = float(row['Percent_test'])/100
		train_unique, valid_unique, test_unique = train_valid_test_split(unique_values_to_split,train_frac, valid_frac, test_frac)
		
		train_df = pd.concat([train_df,df_to_concat[df_to_concat[row['Data_type_for_split']].isin(train_unique)]])
		valid_df = pd.concat([valid_df,df_to_concat[df_to_concat[row['Data_type_for_split']].isin(valid_unique)]])
		test_df = pd.concat([test_df,df_to_concat[df_to_concat[row['Data_type_for_split']].isin(test_unique)]])
		# if row.Split_type == 'test':
		# 	test_df = pd.concat([test_df,df_to_concat], ignore_index = True)
		# elif row.Split_type == 'train':
		# 	train_df = pd.concat([train_df,df_to_concat], ignore_index = True)
		# else:
		# 	print('Split_type ',row['Split_type'], ' doesn\'t exist, it\'s only \'test\' or \'train\'')
		# print(test_df.head())
		# print(train_df.head())
	# train_df, valid_df = train_test_split(train_df, test_size = 0.2, random_state = 51)
	train_test_valid_dfs_to_csv(split_path, train_df, valid_df, test_df, path_to_folders)

def all_randomly_split_dataset(path_to_folders = 'Data/Multitask_data/All_datasets'):
	all_df = pd.read_csv(path_to_folders + '/all_data.csv')
	train_df, test_df = train_test_split(all_df, test_size = 0.2, random_state = 42)
	train_df, valid_df = train_test_split(train_df, test_size = 0.25, random_state = 27)
	newpath = path_to_folders + '/Splits/Fully_random_splits'
	if not os.path.exists(newpath):
		os.makedirs(newpath)
	train_test_valid_dfs_to_csv(newpath, train_df, valid_df, test_df, path_to_folders)


def train_test_valid_dfs_to_csv(path_to_splits, train_df, valid_df, test_df, path_to_col_types):
	col_types = pd.read_csv(path_to_col_types + '/col_type.csv')

	y_vals,x_vals,weights,metadata_cols = split_df_by_col_type(train_df,col_types)
	y_vals_v,x_vals_v,weights_v,metadata_cols_v = split_df_by_col_type(valid_df,col_types)
	for col in y_vals.columns:
		if col != 'smiles':
			if np.isnan(np.nanmax(y_vals[col])):
				print('Deleting column ',col,' from training and validation sets due to lack of values in the training set')
				y_vals = y_vals.drop(columns = [col])
				y_vals_v = y_vals_v.drop(columns = [col])
			elif np.isnan(np.nanmax(y_vals_v[col])):
				print('Deleting column ',col,' from training and validation sets due to lack of values in the validation set')
				y_vals = y_vals.drop(columns = [col])
				y_vals_v = y_vals_v.drop(columns = [col])

	settype = 'train'
	y_vals.to_csv(path_to_splits + '/' + settype + '.csv', index = False)
	x_vals.to_csv(path_to_splits + '/' + settype + '_extra_x.csv', index = False)
	weights.to_csv(path_to_splits + '/' + settype + '_weights.csv', index = False)
	metadata_cols.to_csv(path_to_splits + '/' + settype + '_metadata.csv', index = False)

	settype = 'valid'
	y_vals_v.to_csv(path_to_splits + '/' + settype + '.csv', index = False)
	x_vals_v.to_csv(path_to_splits + '/' + settype + '_extra_x.csv', index = False)
	weights_v.to_csv(path_to_splits + '/' + settype + '_weights.csv', index = False)
	metadata_cols_v.to_csv(path_to_splits + '/' + settype + '_metadata.csv', index = False)

	y_vals,x_vals,weights,metadata_cols = split_df_by_col_type(test_df,col_types)
	settype = 'test'
	y_vals.to_csv(path_to_splits + '/' + settype + '.csv', index = False)
	x_vals.to_csv(path_to_splits + '/' + settype + '_extra_x.csv', index = False)
	weights.to_csv(path_to_splits + '/' + settype + '_weights.csv', index = False)
	metadata_cols.to_csv(path_to_splits + '/' + settype + '_metadata.csv', index = False)


def path_if_none(newpath):
	if not os.path.exists(newpath):
		os.makedirs(newpath)

def run_training(path_to_folders = 'Data/Multitask_data/All_datasets/Splits', epochs = 40):
	train_multitask_model(get_base_args(),path_to_folders, epochs = epochs)

def run_ensemble_training(path_to_folders, ensemble_size = 5, epochs = 40, generator = None):
	for i in range(ensemble_size):
		train_multitask_model(get_base_args(), path_to_folders, epochs = epochs, generator = generator)
		os.rename(path_to_folders+'/trained_model',path_to_folders + '/trained_model_'+str(i))

def run_optimized_ensemble_training(path_to_folders, ensemble_size = 5, epochs = 40, generator = None, path_to_hyperparameters = None):
	if path_to_hyperparameters == None:
		opt_hyper = json.load(open(path_to_folders + '/hyperopt/optimized_configs.json','r'))
	else:
		opt_hyper = json.load(open(path_to_hyperparameters + '/hyperopt/optimized_configs.json','r'))
	print(opt_hyper)
	for i in range(ensemble_size):
		train_hyperparam_optimized_model(get_base_args(), path_to_folders, opt_hyper['depth'], opt_hyper['dropout'], opt_hyper['ffn_num_layers'], opt_hyper['hidden_size'], epochs = epochs, generator = generator)
		os.rename(path_to_folders+'/trained_model',path_to_folders + '/trained_model_'+str(i))

def run_all_trainings(path_to_splits = 'Data/Multitask_data/All_datasets'):
	all_csvs = os.listdir(path_to_splits+'/Split_specs')
	for csv in all_csvs:
		if csv.endswith('.csv'):
			path_to_folders = path_to_splits + '/Splits/'+csv[:-4]
			if not os.path.isdir(path_to_folders+'/trained_model'):
				# print('haven\'t yet trained: ',csv)
				run_training(path_to_folders = path_to_folders)
			else:
				print('already trained ',csv)

def combine_predictions(splits,combo_name, path_to_folders = 'Data/Multitask_data/All_datasets/Splits'):
	savepath = path_to_folders + '/Prediction_combos/'+combo_name
	path_if_none(savepath)
	all_df = {}
	for i,split in enumerate(splits):
		pred_df = pd.read_csv(path_to_folders +'/' + split + '/Predicted_vs_actual_in_silico.csv')
		# print(pred_df.smiles[:10])
		if i == 0:
			all_df['smiles'] = [smiles for smiles in pred_df['smiles']]
		# print(all_df['smiles'][:10])
		preds = pred_df['Avg_pred_quantified_delivery']
		mean = np.mean(preds)
		std = np.std(preds)
		all_df[split] = [(v - mean)/std for v in preds]
	all_avgs = []
	all_stds = []
	all_df = pd.DataFrame(all_df)
	print(all_df.head(10))
	print('now about to do a thing')
	for i, row in all_df.iterrows():
		all_avgs.append(np.mean([row[split] for split in splits]))
		all_stds.append(np.std([row[split] for split in splits]))
	all_df['Avg_pred'] = all_avgs
	all_df['Std_pred'] = all_stds
	all_df['Confidence'] = [1/val for val in all_df['Std_pred']]
	print(all_df.head(10))
	all_df.to_csv(savepath + '/predictions.csv', index = False)
	top_100 = np.argpartition(np.array(all_df.Avg_pred),-100)[-100:]
	top_100_df = all_df.loc[list(top_100),:]
	print('head of top 100: ')
	print(top_100_df.head(10))
	top_100_df.to_csv(savepath + '/top_100.csv',index = False)

	preds_for_pareto = all_df[['Avg_pred','Std_pred']].to_numpy()
	is_efficient = is_pareto_efficient(preds_for_pareto,return_mask = True)
	efficient_subset = all_df[is_efficient]

	plt.figure()
	plt.scatter(all_df.Avg_pred, all_df.Std_pred, color = 'gray')
	plt.scatter(efficient_subset.Avg_pred, efficient_subset.Std_pred, color = 'black')
	plt.xlabel('Average prediction')
	plt.ylabel('Standard deviation of predictions')
	# plt.legend(loc = 'lower right')
	plt.savefig(savepath + '/stdev_Pareto_frontier.png')
	plt.close()
	efficient_subset.to_csv(savepath + '/stdev_Pareto_frontier.csv', index = False)

	preds_for_pareto = all_df[['Avg_pred','Confidence']].to_numpy()
	is_efficient = is_pareto_efficient(preds_for_pareto,return_mask = True)
	efficient_subset = all_df[is_efficient]

	plt.figure()
	plt.scatter(all_df.Avg_pred, all_df.Std_pred, color = 'gray')
	plt.scatter(efficient_subset.Avg_pred, efficient_subset.Std_pred, color = 'black')
	plt.xlabel('Average prediction')
	plt.ylabel('Confidence of predictions')
	# plt.legend(loc = 'lower right')
	plt.savefig(savepath + '/confidence_Pareto_frontier.png')
	plt.close()
	efficient_subset.to_csv(savepath + '/confidence_Pareto_frontier.csv', index = False)

	for i in range(len(splits)):
		for j in range(i+1,len(splits)):
			plt.figure()
			plt.scatter(all_df[splits[i]], all_df[splits[j]],color = 'black')
			plt.xlabel(splits[i]+' prediction')
			plt.ylabel(splits[j]+' prediction')
			plt.savefig(savepath+'/'+splits[i]+'_vs_'+splits[j]+'.png')
			plt.close()


def ensemble_predict(path_to_folders = 'Data/Multitask_data/All_datasets/Splits', ensemble_size = 5, predictions_done = [], path_to_new_test = '',standardize_predictions = True):
	if path_to_new_test == '':
		path_to_data_folders = path_to_folders
		addition = ''
	else:
		addition = '_'+path_to_new_test
		path_to_data_folders = path_to_folders + '/in_silico_screens/'+path_to_new_test
	all_predictions = pd.read_csv(path_to_data_folders + '/test.csv')
	pred_names = list(all_predictions.columns)
	pred_names.remove('smiles')
	metadata = pd.read_csv(path_to_data_folders +'/test_metadata.csv')
	all_predictions = pd.concat([metadata, all_predictions], axis = 1)
	for i in range(ensemble_size):
		# try:
		# 	current_predictions = pd.read_csv(path_to_folders+'/trained_model_'+str(i)+'/Predictions/test_predictions'+addition+'.csv')
		# except:
		# if not i in predictions_done:
		# os.rename(path_to_folders + '/trained_model_'+str(i),path_to_folders+'/trained_model')
		make_predictions(path_to_folders, path_to_new_test = path_to_new_test, ensemble_number = i)
		# os.rename(path_to_folders+'/trained_model',path_to_folders + '/trained_model_'+str(i))
		current_predictions = pd.read_csv(path_to_folders+'/trained_model_'+str(i)+'/Predictions/test_predictions'+addition+'.csv')
		
		current_predictions.drop(columns = ['smiles'], inplace = True)
		for col in current_predictions.columns:
			if standardize_predictions:
				preds_to_standardize = current_predictions[col]
				std = np.std(preds_to_standardize)
				mean = np.mean(preds_to_standardize)
				current_predictions[col] = [(val-mean)/std for val in current_predictions[col]]
			current_predictions.rename(columns = {col:('m'+str(i)+'_pred_'+col)}, inplace = True)
		all_predictions = pd.concat([all_predictions, current_predictions], axis = 1)
	avg_pred = [[] for _ in pred_names]
	stdev_pred = [[] for _ in pred_names]
	# (root squared error)
	rse = [[] for _ in pred_names]
	# all_predictions.to_csv(path_to_folders+'/Predicted_vs_actual'+path_to_new_test+'.csv', index = False)
	for index, row in all_predictions.iterrows():
		for i,pname in enumerate(pred_names):
			all_preds = [row['m'+str(k)+'_pred_'+pname] for k in range(ensemble_size)]
			avg_pred[i].append(np.mean(all_preds))
			stdev_pred[i].append(np.std(all_preds, ddof = 1))
			if path_to_new_test=='':
				rse[i].append(np.sqrt((row[pname]-np.mean(all_preds))**2))
	for i, pname in enumerate(pred_names):
		all_predictions['Avg_pred_'+pname] = avg_pred[i]
		all_predictions['Std_pred_'+pname] = stdev_pred[i]
		if path_to_new_test == '':
			all_predictions['RSE_'+pname] = rse[i]
	all_predictions.to_csv(path_to_folders+'/Predicted_vs_actual'+path_to_new_test+'.csv', index = False)

def predict_each_test_set_cv(path_to_folders = 'Data/Multitask_data/All_datasets/crossval_splits', ensemble_size = 5, predictions_done = [], path_to_new_test = '',standardize_predictions = True):
	# if path_to_new_test == '':
	# 	path_to_data_folders = path_to_folders
	# 	addition = ''
	# 	all_predictions_fname = path_to_folders+'/Predicted_vs_actual'+path_to_new_test+'.csv'
	# path_to_data_folders = path_to_folders + '/in_silico_screens/'+path_to_new_test
	# path_if_none(path_to_folders+'/in_silico_screen_results')
	# all_predictions_fname = path_to_folders+'/in_silico_screen_results/'+path_to_new_test+'.csv'
	# all_predictions = pd.read_csv(path_to_data_folders + '/test.csv')
	# pred_names = list(all_predictions.columns)
	# pred_names.remove('smiles')
	# metadata = pd.read_csv(path_to_data_folders +'/test_metadata.csv')
	# all_predictions = pd.concat([metadata, all_predictions], axis = 1)
	for i in range(ensemble_size):
		# try:
		# 	current_predictions = pd.read_csv(path_to_folders+'/trained_model_'+str(i)+'/Predictions/test_predictions'+addition+'.csv')
		# except:
		# if not i in predictions_done:
		# os.rename(path_to_folders + '/trained_model_'+str(i),path_to_folders+'/trained_model')
		output = pd.read_csv(path_to_folders+'/cv_'+str(i)+'/test.csv')
		metadata = pd.read_csv(path_to_folders+'/cv_'+str(i)+'/test_metadata.csv')
		output = pd.concat([metadata, output], axis = 1)
		try:
			output = pd.read_csv(path_to_folders+'/cv_'+str(i)+'/Predicted_vs_actual.csv')
		except:
			try:
				current_predictions = pd.read_csv(path_to_folders+'/cv_'+str(i)+'/trained_model/Predictions/test_predictions.csv')
			except:
				make_predictions_cv(path_to_folders, path_to_new_test = '', ensemble_number = i)
			# os.rename(path_to_folders+'/trained_model',path_to_folders + '/trained_model_'+str(i))
				current_predictions = pd.read_csv(path_to_folders+'/cv_'+str(i)+'/trained_model/Predictions/test_predictions.csv')
			
			current_predictions.drop(columns = ['smiles'], inplace = True)
			for col in current_predictions.columns:
				if standardize_predictions:
					preds_to_standardize = current_predictions[col]
					std = np.std(preds_to_standardize)
					mean = np.mean(preds_to_standardize)
					current_predictions[col] = [(val-mean)/std for val in current_predictions[col]]
				current_predictions.rename(columns = {col:('cv_'+str(i)+'_pred_'+col)}, inplace = True)
			output = pd.concat([output, current_predictions], axis = 1)
			output.to_csv(path_to_folders+'/cv_'+str(i)+'/Predicted_vs_actual.csv', index = False)
	# avg_pred = [[] for _ in pred_names]
	# stdev_pred = [[] for _ in pred_names]
	# # (root squared error)
	# rse = [[] for _ in pred_names]
	# all_predictions.to_csv(path_to_folders+'/Predicted_vs_actual'+path_to_new_test+'.csv', index = False)
	# for index, row in all_predictions.iterrows():
	# 	for i,pname in enumerate(pred_names):
	# 		all_preds = [row['m'+str(k)+'_pred_'+pname] for k in range(ensemble_size)]
	# 		avg_pred[i].append(np.mean(all_preds))
	# 		stdev_pred[i].append(np.std(all_preds, ddof = 1))
	# 		if path_to_new_test=='':
	# 			rse[i].append(np.sqrt((row[pname]-np.mean(all_preds))**2))
	# for i, pname in enumerate(pred_names):
	# 	all_predictions['Avg_pred_'+pname] = avg_pred[i]
	# 	all_predictions['Std_pred_'+pname] = stdev_pred[i]
	# 	if path_to_new_test == '':
	# 		all_predictions['RSE_'+pname] = rse[i]
	# all_predictions.to_csv(all_predictions_fname, index = False)

def ensemble_predict_cv(path_to_folders = 'Data/Multitask_data/All_datasets/crossval_splits', ensemble_size = 5, predictions_done = [], path_to_new_test = '',standardize_predictions = True):
	# if path_to_new_test == '':
	# 	path_to_data_folders = path_to_folders
	# 	addition = ''
	# 	all_predictions_fname = path_to_folders+'/Predicted_vs_actual'+path_to_new_test+'.csv'
	if not path_to_new_test == '':
		addition = '_'+path_to_new_test
		path_to_data_folders = path_to_folders + '/in_silico_screens/'+path_to_new_test
		path_if_none(path_to_folders+'/in_silico_screen_results')
		all_predictions_fname = path_to_folders+'/in_silico_screen_results/'+path_to_new_test+'.csv'
		all_predictions = pd.read_csv(path_to_data_folders + '/test.csv')
		pred_names = list(all_predictions.columns)
		pred_names.remove('smiles')
		metadata = pd.read_csv(path_to_data_folders +'/test_metadata.csv')
		all_predictions = pd.concat([metadata, all_predictions], axis = 1)
	for i in range(ensemble_size):
		# try:
		# 	current_predictions = pd.read_csv(path_to_folders+'/trained_model_'+str(i)+'/Predictions/test_predictions'+addition+'.csv')
		# except:
		# if not i in predictions_done:
		# os.rename(path_to_folders + '/trained_model_'+str(i),path_to_folders+'/trained_model')
		# print('HERE!!!!')
		try:
			current_predictions = pd.read_csv(path_to_folders+'/cv_'+str(i)+'/trained_model/Predictions/test_predictions'+addition+'.csv')
		except:
			make_predictions_cv(path_to_folders, path_to_new_test = path_to_new_test, ensemble_number = i)
		# os.rename(path_to_folders+'/trained_model',path_to_folders + '/trained_model_'+str(i))
		current_predictions = pd.read_csv(path_to_folders+'/cv_'+str(i)+'/trained_model/Predictions/test_predictions'+addition+'.csv')
		
		current_predictions.drop(columns = ['smiles'], inplace = True)
		for col in current_predictions.columns:
			if standardize_predictions:
				preds_to_standardize = current_predictions[col]
				std = np.std(preds_to_standardize)
				mean = np.mean(preds_to_standardize)
				current_predictions[col] = [(val-mean)/std for val in current_predictions[col]]
			current_predictions.rename(columns = {col:('m'+str(i)+'_pred_'+col)}, inplace = True)
		all_predictions = pd.concat([all_predictions, current_predictions], axis = 1)
	avg_pred = [[] for _ in pred_names]
	stdev_pred = [[] for _ in pred_names]
	# (root squared error)
	rse = [[] for _ in pred_names]
	# all_predictions.to_csv(path_to_folders+'/Predicted_vs_actual'+path_to_new_test+'.csv', index = False)
	for index, row in all_predictions.iterrows():
		for i,pname in enumerate(pred_names):
			all_preds = [row['m'+str(k)+'_pred_'+pname] for k in range(ensemble_size)]
			avg_pred[i].append(np.mean(all_preds))
			stdev_pred[i].append(np.std(all_preds, ddof = 1))
			if path_to_new_test=='':
				rse[i].append(np.sqrt((row[pname]-np.mean(all_preds))**2))
	for i, pname in enumerate(pred_names):
		all_predictions['Avg_pred_'+pname] = avg_pred[i]
		all_predictions['Std_pred_'+pname] = stdev_pred[i]
		if path_to_new_test == '':
			all_predictions['RSE_'+pname] = rse[i]
	all_predictions.to_csv(all_predictions_fname, index = False)

def make_predictions_cv(path_to_folders = 'Data/Multitask_data/All_datasets/Splits', path_to_new_test = '', ensemble_number = -1):
	print('PATH TO FOLDERS: ',path_to_folders)
	predict_folder = path_to_folders + '/trained_model/Predictions'
	if ensemble_number>-0.5:
		predict_folder = path_to_folders +'/cv_'+str(ensemble_number)+ '/trained_model/Predictions'
	path_if_none(predict_folder)
	print('PATH TO FOLDERS: ',path_to_folders)
	predict_multitask_from_json_cv(get_base_predict_args(),model_path = path_to_folders, path_to_new_test = path_to_new_test, ensemble_number = ensemble_number)


def reanalyze_classification_predictions(split_name, path_to_preds = 'Data/Multitask_data/All_datasets'):
	perf_summary = pd.read_csv(path_to_preds+'/Splits/'+split_name+'/Results/Performance_summary.csv')
	custom_aucs = []
	for index, row in perf_summary.iterrows():
		value_cutoff = row['Value_cutoff']
		if np.isnan(value_cutoff):
			custom_aucs.append('n/a')
		else:
			print(value_cutoff)
			pred_split_name = row['Analysis']
			dtype = row['Measurement_type']
			analyzed_path = path_to_preds+'/Splits/'+split_name+'/Results/'+pred_split_name+'/'+dtype
			data_df = pd.read_csv(analyzed_path+'/pred_vs_actual_data.csv')
			analyzed_vbl = data_df[dtype]
			if row['Goal'] == 'max':
				classes = [vbl>value_cutoff for vbl in analyzed_vbl]
				predictions = data_df['Avg_pred_'+dtype]
			elif row['Goal']=='min':
				classes = [vbl<value_cutoff for vbl in analyzed_vbl]
				predictions = [-v for v in data_df['Avg_pred_'+dtype]]
			try:
				auc_score = roc_auc_score(classes, predictions)
			except:
				auc_score = np.nan
			custom_aucs.append(auc_score)
			data_df[('Is_hit_by_cutoff_%0.2f_'%value_cutoff) + dtype] = classes
			data_df.to_csv(analyzed_path + '/pred_vs_actual_data.csv', index = False)
			fpr, tpr, thresholds = roc_curve(classes,predictions)
			plt.figure()
			plt.plot(fpr, tpr, color = 'black', label = 'ROC curve cutoff %0.2f'%value_cutoff +' (area = %0.2f)' % auc_score)
			plt.plot([0,1],[0,1],color = 'blue',linestyle = '--')
			plt.xlim([0.0,1.0])
			plt.ylim([0.0,1.05])
			plt.xlabel('False positive rate')
			plt.ylabel('True positive rate')
			plt.legend(loc = 'lower right')
			plt.savefig(analyzed_path + '/custom_roc_curve.png')
			plt.close()

def analyze_new_lipid_predictions(split_name, addition = '_in_silico',path_to_preds = 'Data/Multitask_data/All_datasets'):
	preds_vs_actual = pd.read_csv(path_to_preds + '/Splits/'+split_name+'/Predicted_vs_actual'+addition+'.csv')
	analyzed_path = path_to_preds+'/Splits/'+split_name+'/in_silico_screen_results'
	preds_vs_actual['Confidence'] = [1/val for val in preds_vs_actual['Std_pred_quantified_delivery']]
	path_if_none(analyzed_path)
	preds_for_pareto = preds_vs_actual[['Avg_pred_quantified_delivery','Std_pred_quantified_delivery']].to_numpy()
	print('Dimensions: ',preds_for_pareto.shape)
	is_efficient = is_pareto_efficient(preds_for_pareto,return_mask = True)
	efficient_subset = preds_vs_actual[is_efficient]
	plt.figure()
	plt.scatter(preds_vs_actual.Avg_pred_quantified_delivery, preds_vs_actual.Std_pred_quantified_delivery, color = 'gray')
	plt.scatter(efficient_subset.Avg_pred_quantified_delivery, efficient_subset.Std_pred_quantified_delivery, color = 'black')
	plt.xlabel('Average prediction')
	plt.ylabel('Standard deviation of predictions')
	# plt.legend(loc = 'lower right')
	plt.savefig(analyzed_path + '/stdev_Pareto_frontier.png')
	plt.close()
	efficient_subset.to_csv(analyzed_path + '/stdev_Pareto_frontier.csv', index = False)

	preds_for_pareto = preds_vs_actual[['Avg_pred_quantified_delivery','Confidence']].to_numpy()
	print('Dimensions: ',preds_for_pareto.shape)
	is_efficient = is_pareto_efficient(preds_for_pareto,return_mask = True)
	efficient_subset = preds_vs_actual[is_efficient]
	plt.figure()
	plt.scatter(preds_vs_actual.Avg_pred_quantified_delivery, preds_vs_actual.Std_pred_quantified_delivery, color = 'gray')
	plt.scatter(efficient_subset.Avg_pred_quantified_delivery, efficient_subset.Std_pred_quantified_delivery, color = 'black')
	plt.xlabel('Average prediction')
	plt.ylabel('Standard deviation of predictions')
	# plt.legend(loc = 'lower right')
	plt.savefig(analyzed_path + '/confidence_Pareto_frontier.png')
	plt.close()
	efficient_subset.to_csv(analyzed_path + '/confidence_Pareto_frontier.csv', index = False)

	top_50 = np.argpartition(np.array(preds_vs_actual.Avg_pred_quantified_delivery),-50)[-50:]
	print(list(top_50))
	top_50_df = preds_vs_actual.loc[list(top_50),:]
	top_50_df.to_csv(analyzed_path + '/top_50.csv',index = False)

def generate_normalized_data(all_df, split_variables = ['Experiment_ID','Library_ID','Delivery_target','Route_of_administration']):
	split_names = []
	norm_dict = {}
	for index, row in all_df.iterrows():
		split_name = ''
		for vbl in split_variables:
			# print(row[vbl])
			# print(vbl)
			split_name = split_name + row[vbl]+'_'
		split_names.append(split_name[:-1])
	unique_split_names = set(split_names)
	for split_name in unique_split_names:
		data_subset = all_df[[spl==split_name for spl in split_names]]
		norm_dict[split_name] = (np.mean(data_subset['quantified_delivery']), np.std(data_subset['quantified_delivery']))
	norm_delivery = []
	for i, row in all_df.iterrows():
		val = row['quantified_delivery']
		split = split_names[i]
		stdev = norm_dict[split][1]
		mean = norm_dict[split][0]
		norm_delivery.append((float(val)-mean)/stdev)
	return split_names, norm_delivery


def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs>costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

def analyze_predictions(split_name,pred_split_variables = ['Experiment_ID','Library_ID','Delivery_target','Route_of_administration'], path_to_preds = 'Data/Multitask_data/All_datasets'):
	preds_vs_actual = pd.read_csv(path_to_preds + '/Splits/'+split_name+'/Predicted_vs_actual.csv')
	pred_split_names = []
	for index, row in preds_vs_actual.iterrows():
		pred_split_name = ''
		for vbl in pred_split_variables:
			pred_split_name = pred_split_name + row[vbl] + '_'
		pred_split_names.append(pred_split_name[:-1])
	preds_vs_actual['Prediction_split_name'] = pred_split_names
	unique_pred_split_names = set(pred_split_names)
	cols = preds_vs_actual.columns
	data_types = []
	for col in cols:
		if col.startswith('Avg_pred'):
			data_types.append(col[9:])

	summary_table = pd.DataFrame({})
	all_names = []
	all_dtypes = []
	all_ns = []
	all_pearson = []
	all_pearson_p_val = []
	all_kendall = []
	all_spearman = []
	all_rmse = []
	all_error_pearson = []
	all_error_pearson_p_val = []
	all_aucs = []
	all_goals = []

	for pred_split_name in unique_pred_split_names:
		path_if_none(path_to_preds+'/Splits/'+split_name+'/Results/'+pred_split_name)
		data_subset = preds_vs_actual[preds_vs_actual['Prediction_split_name']==pred_split_name].reset_index(drop=True)
		value_names = set(list(data_subset.Value_name))
		if len(value_names)>1:
			raise Exception('Multiple types of measurement in the same prediction split: split ',pred_split_name,' has value names ',value_names,'. Try adding more pred split variables.')
		else:
			value_name = [val_name for val_name in value_names][0]
		kept_dtypes = []
		for dtype in data_types:
			keep = False
			for val in data_subset[dtype]:
				if not np.isnan(val):
					keep = True
			if keep:
				analyzed_path = path_to_preds+'/Splits/'+split_name+'/Results/'+pred_split_name+'/'+dtype
				path_if_none(analyzed_path)
				# print(data_subset['Goal'])
				goal = data_subset['Goal'][0]
				all_goals.append(goal)
				kept_dtypes.append(dtype)
				analyzed_data = pd.DataFrame({'smiles':data_subset.smiles})
				actual = data_subset[dtype]
				pred = data_subset['Avg_pred_'+dtype]
				std_pred = data_subset['Std_pred_'+dtype]
				rse = data_subset['RSE_'+dtype]
				analyzed_data[dtype] = actual
				analyzed_data['Avg_pred_'+dtype] = pred
				analyzed_data['Std_pred_'+dtype] = std_pred
				analyzed_data['RSE_pred_'+dtype] = rse
				residuals = [actual[blah]-pred[blah] for blah in range(len(pred))]
				analyzed_data['Residual'] = residuals
				pearson = scipy.stats.pearsonr(actual, pred)
				spearman, pval = scipy.stats.spearmanr(actual, pred)
				kendall, pval = scipy.stats.kendalltau(actual, pred)
				rmse = np.sqrt(mean_squared_error(actual, pred))
				error_pearson = scipy.stats.pearsonr(std_pred,rse)
				all_names.append(pred_split_name)
				all_dtypes.append(dtype)
				all_pearson.append(pearson[0])
				all_pearson_p_val.append(pearson[1])
				all_kendall.append(kendall)
				all_spearman.append(spearman)
				all_rmse.append(rmse)
				all_error_pearson.append(error_pearson[0])
				all_error_pearson_p_val.append(error_pearson[1])
				all_ns.append(len(pred))

				# measure ROCs
				sorted_actual = sorted(actual)
				ranks = [float(sorted_actual.index(v))/len(actual) for v in actual]
				if goal == 'min':
					classification = [1*(rank<0.1) for rank in ranks]
					pred_for_class = [-v for v in pred]
				elif goal == 'max':
					classification = [1*(rank>0.9) for rank in ranks]
					pred_for_class = [v for v in pred]
				fpr, tpr, thresholds = roc_curve(classification,pred_for_class)
				# print(classification)
				try:
					auc_score = roc_auc_score(classification, pred_for_class)
				except:
					auc_score = np.nan
				all_aucs.append(auc_score)
				analyzed_data['Is_10th_percentile_hit_'+dtype] = classification


				plt.figure()
				plt.plot(fpr, tpr, color = 'black', label = 'ROC curve 10th percentile (area = %0.2f)' % auc_score)
				plt.plot([0,1],[0,1],color = 'blue',linestyle = '--')
				plt.xlim([0.0,1.0])
				plt.ylim([0.0,1.05])
				plt.xlabel('False positive rate')
				plt.ylabel('True positive rate')
				plt.legend(loc = 'lower right')
				plt.savefig(analyzed_path + '/roc_curve.png')
				plt.close()
				plt.figure()
				plt.scatter(pred,actual,color = 'black')
				plt.plot(np.unique(pred),np.poly1d(np.polyfit(pred, actual, 1))(np.unique(pred)))
				plt.xlabel('Predicted '+value_name)
				plt.ylabel('Experimental '+value_name)
				plt.savefig(analyzed_path+'/pred_vs_actual.png')
				plt.close()
				plt.figure()
				plt.scatter(std_pred,residuals,color = 'black')
				plt.plot(np.unique(std_pred),np.poly1d(np.polyfit(std_pred, residuals, 1))(np.unique(std_pred)))
				plt.xlabel('Residual (Actual-Predicted) '+value_name)
				plt.ylabel('Ensemble model uncertainty '+value_name)
				plt.savefig(analyzed_path+'/residual_vs_stdev.png')
				plt.close()
				plt.figure()
				plt.scatter(std_pred,rse,color = 'black')
				plt.plot(np.unique(std_pred),np.poly1d(np.polyfit(std_pred, rse, 1))(np.unique(std_pred)))
				plt.xlabel('Ensemble model uncertainty')
				plt.ylabel('Root quared error')
				plt.savefig(analyzed_path+'/std_vs_rse.png')
				plt.close()
				analyzed_data.to_csv(analyzed_path+'/pred_vs_actual_data.csv', index = False)
	summary_table['Analysis'] = all_names
	summary_table['Measurement_type'] = all_dtypes
	summary_table['n'] = all_ns
	summary_table['Goal'] = all_goals
	summary_table['pearson_rho'] = all_pearson
	summary_table['pearson_rho_p_val'] = all_pearson_p_val
	summary_table['kendall_tau'] = all_kendall
	summary_table['spearman_r'] = all_spearman
	summary_table['rmse'] = all_rmse
	summary_table['error_pearson'] = all_error_pearson
	summary_table['error_pearson_p_val'] = all_error_pearson_p_val
	summary_table['AUC 10th percentile'] = all_aucs
	summary_table['Value_cutoff'] = ['n/a' for _ in all_aucs]
	summary_table.to_csv(path_to_preds+'/Splits/'+split_name+'/Results/Performance_summary.csv', index = False)
# 
			
def run_optimized_cv_training(path_to_folders, ensemble_size = 5, epochs = 40, generator = None, path_to_hyperparameters = None):
	if path_to_hyperparameters == None:
		opt_hyper = json.load(open(path_to_folders + '/hyperopt/optimized_configs.json','r'))
	else:
		opt_hyper = json.load(open(path_to_hyperparameters + '/hyperopt/optimized_configs.json','r'))
	print(opt_hyper)
	for i in range(ensemble_size):
		train_hyperparam_optimized_model(get_base_args(), path_to_folders+'/cv_'+str(i), opt_hyper['depth'], opt_hyper['dropout'], opt_hyper['ffn_num_layers'], opt_hyper['hidden_size'], epochs = epochs, generator = generator)
		# os.rename(path_to_folders+'/trained_model',path_to_folders + '/trained_model_'+str(i))

def analyze_predictions_cv(split_name,pred_split_variables = ['Experiment_ID','Library_ID','Delivery_target','Route_of_administration'], path_to_preds = 'Data/Multitask_data/All_datasets/crossval_splits/', ensemble_number = 5, min_values_for_analysis = 5):
	summary_table = pd.DataFrame({})
	all_names = {}
	# all_dtypes = {}
	all_ns = {}
	all_pearson = {}
	all_pearson_p_val = {}
	all_kendall = {}
	all_spearman = {}
	all_rmse = {}
	all_unique = []
	for i in range(ensemble_number):
		preds_vs_actual = pd.read_csv(path_to_preds+split_name+'/cv_'+str(i)+'/Predicted_vs_actual.csv')
		pred_split_names = []
		for index, row in preds_vs_actual.iterrows():
			pred_split_name = ''
			for vbl in pred_split_variables:
				pred_split_name = pred_split_name + row[vbl] + '_'
			pred_split_names.append(pred_split_name[:-1])
		all_unique = all_unique + list(set(pred_split_names))
	unique_pred_split_names = set(all_unique)
	for un in unique_pred_split_names:
		# all_names[un] = []
		# all_dtype,s[un] = []
		all_ns[un] = []
		all_pearson[un] = []
		all_pearson_p_val[un] = []
		all_kendall[un] = []
		all_spearman[un] = []
		all_rmse[un] = []
	for i in range(ensemble_number):
		preds_vs_actual = pd.read_csv(path_to_preds+split_name+'/cv_'+str(i)+'/Predicted_vs_actual.csv')
		pred_split_names = []
		for index, row in preds_vs_actual.iterrows():
			pred_split_name = ''
			for vbl in pred_split_variables:
				pred_split_name = pred_split_name + row[vbl] + '_'
			pred_split_names.append(pred_split_name[:-1])
		preds_vs_actual['Prediction_split_name'] = pred_split_names
		# unique_pred_split_names = set(pred_split_names)
		cols = preds_vs_actual.columns
		data_types = []
		for col in cols:
			if col[:3]=='cv_':
				data_types.append(col)
			
		# all_error_pearson = {}
		# all_error_pearson_p_val = {}
		# all_aucs = []
		# all_goals = []

		for pred_split_name in unique_pred_split_names:
			path_if_none(path_to_preds+split_name+'/cv_'+str(i)+'/Results')
			data_subset = preds_vs_actual[preds_vs_actual['Prediction_split_name']==pred_split_name].reset_index(drop=True)
			value_names = set(list(data_subset.Value_name))
			if len(value_names)>1:
				raise Exception('Multiple types of measurement in the same prediction split: split ',pred_split_name,' has value names ',value_names,'. Try adding more pred split variables.')
			elif len(value_names)==0:
				value_name = 'Empty, ignore!'
			else:
				value_name = [val_name for val_name in value_names][0]
			kept_dtypes = []
			for dtype in data_types:
				# keep = False
				# for val in data_subset[dtype]:
				# 	if not np.isnan(val):
				# 		keep = True
				# if keep:
				analyzed_path = path_to_preds+split_name+'/cv_'+str(i)+'/Results/'+pred_split_name+'/'+dtype
				path_if_none(analyzed_path)
				# print(data_subset['Goal'])
				# goal = data_subset['Goal'][0]
				# all_goals.append(goal)
				kept_dtypes.append(dtype)
				analyzed_data = pd.DataFrame({'smiles':data_subset.smiles})
				actual = data_subset['quantified_delivery']
				pred = data_subset['cv_'+str(i)+'_pred_quantified_delivery']
				# std_pred = data_subset['Std_pred_'+dtype]
				# rse = data_subset['RSE_'+dtype]
				# analyzed_data[dtype] = actual
				# analyzed_data['Prediction'] = pred
				# analyzed_data['Actual'] = actual
				# analyzed_data['RSE_pred_'+dtype] = rse
				# residuals = [actual[blah]-pred[blah] for blah in range(len(pred))]
				# analyzed_data['Residual'] = residuals
				if len(actual)>=min_values_for_analysis:
					pearson = scipy.stats.pearsonr(actual, pred)
					spearman, pval = scipy.stats.spearmanr(actual, pred)
					kendall, pval = scipy.stats.kendalltau(actual, pred)
					
					# error_pearson = scipy.stats.pearsonr(std_pred,rse)
					# all_names[pred_split_name].append(pred_split_name)
					# all_dtypes.append(dtype)
					rmse = np.sqrt(mean_squared_error(actual, pred))
					all_rmse[pred_split_name] = all_rmse[pred_split_name] + [rmse]
					
					all_pearson[pred_split_name] = all_pearson[pred_split_name] + [pearson[0]]
					all_pearson_p_val[pred_split_name] = all_pearson_p_val[pred_split_name] + [pearson[1]]
					all_kendall[pred_split_name] = all_kendall[pred_split_name] + [kendall]
					all_spearman[pred_split_name] = all_spearman[pred_split_name] + [spearman]
					# all_pearson_p_val[pred_split_name].append(pearson[1])
					# all_kendall[pred_split_name].append(kendall)
					# all_spearman[pred_split_name].append(spearman)
					plt.figure()
					plt.scatter(pred,actual,color = 'black')
					plt.plot(np.unique(pred),np.poly1d(np.polyfit(pred, actual, 1))(np.unique(pred)))
					plt.xlabel('Predicted '+value_name)
					plt.ylabel('Experimental '+value_name)
					plt.savefig(analyzed_path+'/pred_vs_actual.png')
					plt.close()
				else:
					all_rmse[pred_split_name] = all_rmse[pred_split_name] + [float('nan')]
					all_pearson[pred_split_name] = all_pearson[pred_split_name] + [float('nan')]
					all_pearson_p_val[pred_split_name] = all_pearson_p_val[pred_split_name] + [float('nan')]
					all_kendall[pred_split_name] = all_kendall[pred_split_name] + [float('nan')]
					all_spearman[pred_split_name] = all_spearman[pred_split_name] + [float('nan')]
					# all_pearson_p_val[pred_split_name].append(float('nan'))
					# all_kendall[pred_split_name].append(float('nan'))
					# all_spearman[pred_split_name].append(float('nan'))
				# all_error_pearson.append(error_pearson[0])
				# all_error_pearson_p_val.append(error_pearson[1])
				# print(all_pearson)
				
				all_ns[pred_split_name] = all_ns[pred_split_name] + [len(pred)]
				# all_rmse[pred_split_name].append(rmse)
				# all_ns[pred_split_name].append(len(pred))

				# measure ROCs
				# sorted_actual = sorted(actual)
				# ranks = [float(sorted_actual.index(v))/len(actual) for v in actual]
				# if goal == 'min':
				# 	classification = [1*(rank<0.1) for rank in ranks]
				# 	pred_for_class = [-v for v in pred]
				# elif goal == 'max':
				# 	classification = [1*(rank>0.9) for rank in ranks]
				# 	pred_for_class = [v for v in pred]
				# fpr, tpr, thresholds = roc_curve(classification,pred_for_class)
				# print(classification)
				# try:
				# 	auc_score = roc_auc_score(classification, pred_for_class)
				# except:
				# 	auc_score = np.nan
				# all_aucs.append(auc_score)
				# analyzed_data['Is_10th_percentile_hit_'+dtype] = classification


				# plt.figure()
				# plt.plot(fpr, tpr, color = 'black', label = 'ROC curve 10th percentile (area = %0.2f)' % auc_score)
				# plt.plot([0,1],[0,1],color = 'blue',linestyle = '--')
				# plt.xlim([0.0,1.0])
				# plt.ylim([0.0,1.05])
				# plt.xlabel('False positive rate')
				# plt.ylabel('True positive rate')
				# plt.legend(loc = 'lower right')
				# plt.savefig(analyzed_path + '/roc_curve.png')
				# plt.close()
				
				# plt.figure()
				# plt.scatter(std_pred,residuals,color = 'black')
				# plt.plot(np.unique(std_pred),np.poly1d(np.polyfit(std_pred, residuals, 1))(np.unique(std_pred)))
				# plt.xlabel('Residual (Actual-Predicted) '+value_name)
				# plt.ylabel('Ensemble model uncertainty '+value_name)
				# plt.savefig(analyzed_path+'/residual_vs_stdev.png')
				# plt.close()
				# plt.figure()
				# plt.scatter(std_pred,rse,color = 'black')
				# plt.plot(np.unique(std_pred),np.poly1d(np.polyfit(std_pred, rse, 1))(np.unique(std_pred)))
				# plt.xlabel('Ensemble model uncertainty')
				# plt.ylabel('Root quared error')
				# plt.savefig(analyzed_path+'/std_vs_rse.png')
				# plt.close()
				analyzed_data.to_csv(analyzed_path+'/pred_vs_actual_data.csv', index = False)
	crossval_results_path = path_to_preds+split_name+'/crossval_performance'
	path_if_none(crossval_results_path)

# all_ns[un] = {}
# 			all_pearson[un] = {}
# 			all_pearson_p_val[un] = {}
# 			all_kendall[un] = {}
# 			all_spearman[un] = {}
# 			all_rmse[un] = {}
	# print(all_ns)
	pd.DataFrame.from_dict(all_ns).to_csv(crossval_results_path+'/n_vals.csv', index = True)
	pd.DataFrame.from_dict(all_pearson).to_csv(crossval_results_path+'/pearson.csv', index = True)
	pd.DataFrame.from_dict(all_pearson_p_val).to_csv(crossval_results_path+'/pearson_p_val.csv', index = True)
	pd.DataFrame.from_dict(all_kendall).to_csv(crossval_results_path+'/kendall.csv', index = True)
	pd.DataFrame.from_dict(all_spearman).to_csv(crossval_results_path+'/spearman.csv', index = True)
	pd.DataFrame.from_dict(all_rmse).to_csv(crossval_results_path+'/rmse.csv', index = True)
	# summary_table['Analysis'] = all_names
	# summary_table['Measurement_type'] = all_dtypes
	# summary_table['n'] = all_ns
	# summary_table['Goal'] = all_goals
	# summary_table['pearson_rho'] = all_pearson
	# summary_table['pearson_rho_p_val'] = all_pearson_p_val
	# summary_table['kendall_tau'] = all_kendall
	# summary_table['spearman_r'] = all_spearman
	# summary_table['rmse'] = all_rmse
	# summary_table['error_pearson'] = all_error_pearson
	# summary_table['error_pearson_p_val'] = all_error_pearson_p_val
	# summary_table['AUC 10th percentile'] = all_aucs
	# summary_table['Value_cutoff'] = ['n/a' for _ in all_aucs]
	# summary_table.to_csv(path_to_preds+'/Splits/'+split_name+'/Results/Performance_summary.csv', index = False)

def make_predictions(path_to_folders = 'Data/Multitask_data/All_datasets/Splits', path_to_new_test = '', ensemble_number = -1):
	predict_folder = path_to_folders + '/trained_model/Predictions'
	if ensemble_number>-0.5:
		predict_folder = path_to_folders + '/trained_model_'+str(ensemble_number)+'/Predictions'
	path_if_none(predict_folder)
	predict_multitask_from_json(get_base_predict_args(),model_path = path_to_folders, path_to_new_test = path_to_new_test, ensemble_number = ensemble_number)

def make_all_predictions(path_to_splits = 'Data/Multitask_data/All_datasets'):
	all_csvs = os.listdir(path_to_splits+'/Split_specs')
	for csv in all_csvs:
		if csv.endswith('.csv'):
			path_to_folders = path_to_splits + '/Splits/'+csv[:-4]
			if not os.path.isdir(path_to_folders+'/trained_model'):
				print('haven\'t yet trained: ',csv[:-4])
				# run_training(path_to_folders = path_to_folders)
			else:
				print('Doing predictions for: ',csv[:-4])
				make_predictions(path_to_folders = path_to_folders)

def hyperparam_optimize_split(split, niters = 20):
	generator = None
	if split.endswith('_morgan'):
		generator = ['morgan_count']
		specified_dataset_split(split[:-7]+'.csv',is_morgan = True)
	else:
		specified_dataset_split(split+'.csv')
	optimize_hyperparameters(get_base_args(), path_to_splits = 'Data/Multitask_data/All_datasets/Splits/'+split,epochs = 50, num_iters = niters, generator = generator)
	run_optimized_ensemble_training('Data/Multitask_data/All_datasets/Splits/'+split, ensemble_size = 5, epochs = 50, generator = generator)

# def analyze_predictions(split_folder, base_path = 'Data/Multitask_data/All_datasets'):

# merge_datasets(None)

# merge_datasets(['A549_form_screen','Whitehead_siRNA','LM_3CR','RM_BL_AG_carbonate'])


def main(argv):
	# args = sys.argv[1:]
	task_type = argv[1]
	if task_type == 'hyperparam_optimize':
		split_list = argv[2]
		# arg is the name of a split list file
		split_df = pd.read_csv('Data/Multitask_data/All_datasets/Split_lists/'+split_list+'.csv')
		for split in split_df['split']:
			# print('starting split: ',split)
			hyperparam_optimize_split(split)
	elif task_type == 'train_optimized_from_to_cv_already_split':
		from_split = argv[2]
		to_split = argv[3]
		generator = None
		if to_split.endswith('_morgan'):
			generator = ['morgan']
		run_optimized_cv_training('Data/Multitask_data/All_datasets/crossval_splits/'+to_split, epochs = 100, path_to_hyperparameters = 'Data/Multitask_data/All_datasets/Splits/'+from_split, generator = generator)
	elif task_type == 'specified_cv_split':
		split = argv[2]
		ultra_held_out = float(argv[3])
		is_morgan = False
		if len(argv)>4:
			if argv[4]=='morgan':
				is_morgan = True
		specified_cv_split(split,ultra_held_out_fraction = ultra_held_out, is_morgan = is_morgan)
	elif task_type == 'analyze_cv':
		# output.to_csv(path_to_folders+'/cv_'+str(i)+'/Predicted_vs_actual.csv', index = False)
		split = argv[2]
		predict_each_test_set_cv(path_to_folders =  'Data/Multitask_data/All_datasets/crossval_splits/'+split, predictions_done = [], ensemble_size = 5)
		analyze_predictions_cv(split)


	elif task_type == 'ensemble_screen_cv':
		split = argv[2]
		in_silico_folders = argv[3:]
		for folder in in_silico_folders:
			ensemble_predict_cv(path_to_folders = 'Data/Multitask_data/All_datasets/crossval_splits/'+split, ensemble_size = 5, path_to_new_test = folder)
	elif task_type == 'analyze':
		split = argv[2]
		ensemble_predict(path_to_folders =  'Data/Multitask_data/All_datasets/Splits/'+split, predictions_done = [], ensemble_size = 5)
		analyze_predictions(split)
	elif task_type == 'train_optimized_from_to_already_split':
		from_split = argv[2]
		to_split = argv[3]
		run_optimized_ensemble_training('Data/Multitask_data/All_datasets/Splits/'+to_split,ensemble_size = 5, epochs = 50, path_to_hyperparameters = 'Data/Multitask_data/All_datasets/Splits/'+from_split)
	elif task_type == 'train_optimized_from_to':
		from_split = argv[2]
		to_split = argv[3]
		if to_split.endswith('_morgan'):
			specified_dataset_split(to_split[:-7] + '.csv', is_morgan = True)
		else:
			specified_dataset_split(to_split + '.csv')
		run_optimized_ensemble_training('Data/Multitask_data/All_datasets/Splits/'+to_split,ensemble_size = 5, epochs = 50, path_to_hyperparameters = 'Data/Multitask_data/All_datasets/Splits/'+from_split)
	elif task_type == 'analyze_new_library':
		split = argv[2]
		ensemble_predict(path_to_folders = 'Data/Multitask_data/All_datasets/Splits/'+split, predictions_done = [], ensemble_size = 5, addition = '_in_silico')
		analyze_new_lipid_predictions(split)
	elif task_type == 'ensemble_screen':
		split = argv[2]
		in_silico_folders = argv[3:]
		for folder in in_silico_folders:
			ensemble_predict(path_to_folders = 'Data/Multitask_data/All_datasets/Splits/'+split, ensemble_size = 5, path_to_new_test = folder)
	elif task_type == 'combine_library_analyses':
		combo_name = argv[2]
		splits = argv[3:]
		combine_predictions(splits, combo_name)
	elif task_type == 'merge_datasets':
		merge_datasets(None)

if __name__ == '__main__':
	main(sys.argv)

# all_splits = ['Predict_MCR_all_train_amine_split','Predict_MCR_all_train_random_split','Predict_MCR_MCR_train_amine_split','Predict_MCR_MCR_train_random_split','Predict_sting_all_train_amine_split','Predict_sting_all_train_random_split','Predict_sting_MCR_train_amine_split','Predict_sting_MCR_train_random_split','All_amine_split','All_random']
# for split in all_splits:
	# specified_dataset_split(split + '.csv')

# split = 'Ubermodel_split_hyperopt'
# specified_dataset_split(split+'.csv')

# optimize_hyperparameters(get_base_args(),path_to_splits = 'Data/Multitask_data/All_datasets/Splits/'+split, epochs = 50, num_iters = 5)
# analyze_predictions('Predict_MCR_MCR_train_amine_split')
# run_optimized_ensemble_training('Data/Multitask_data/All_datasets/Splits/'+split, ensemble_size = 5, epochs = 50)

# all_splits = ['Predict_MCR_MCR_train_random_split','Predict_sting_all_train_amine_split','Predict_sting_all_train_random_split','Predict_sting_MCR_train_amine_split','Predict_sting_MCR_train_random_split','All_amine_split','All_random']

# all_splits = ['Ubermodel_split']
# all_splits = ['Predict_Raj_ester_raj_ester_only','Predict_Raj_ester_only_other_with_A549','Predict_Raj_ester_only_other_no_A549','Predict_Raj_ester_all_data']
# all_splits = ['Predict_Raj_ester_raj_ester_only_tail_split']
# for split in all_splits:
# 	specified_dataset_split(split + '.csv')
# for split in all_splits:
# 	nep = 40
# 	if split == 'Predict_Raj_ester_raj_ester_only':
# 		nep = 80
# 	run_ensemble_training(path_to_folders = 'Data/Multitask_data/All_datasets/Splits/'+split,ensemble_size = 3, epochs = nep)
# 	ensemble_predict(path_to_folders =  'Data/Multitask_data/All_datasets/Splits/'+split, predictions_done = [], ensemble_size = 3)
# 	analyze_predictions(split)
	# analyze_new_lipid_predictions(split, addition = '_in_silico')

# ensemble_predict(path_to_folders =  'Data/Multitask_data/All_datasets/Splits/Predict_ester_with_non_ester', predictions_done = True)
# analyze_predictions('Predict_ester_with_non_ester')
# reanalyze_classification_predictions('Predict_ester_with_non_ester')


# specified_dataset_split('RM_carbonate_and_LM_predict_no_other_libraries.csv')

# run_all_trainings()
# do_all_splits()
# run_all_trainings()
# make_all_predictions()


# split_dataset()

# run_training(path_to_folders = 'Data/Multitask_data/All_datasets/Splits/Predict_carbonate_whitehead_3cr')
# run_training(path_to_folders = 'Data/Multitask_data/All_datasets/Splits/RM_carbonate_and_LM_predict_no_other_libraries')

# run_training(path_to_folders = 'Data/Multitask_data/All_datasets/Splits/RM_test', epochs = 50)
# run_training(path_to_folders = 'Data/Multitask_data/All_datasets/Splits/RM_and_IR_test', epochs = 50)

# make_predictions('Data/Multitask_data/All_datasets/Splits/Predict_carbonate_whitehead_3cr')
# make_predictions('Data/Multitask_data/All_datasets/Splits/RM_carbonate_and_LM_predict_no_other_libraries')

# make_predictions('Data/Multitask_data/All_datasets/Splits/Predict_carbonate_whitehead_3cr')





# 