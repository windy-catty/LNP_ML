import numpy as np 
import os
import pandas as pd 

# /Users/jacobwitten/Documents/Next_steps/Anderson/ML for NP design/Chemprop/chemprop-master/Data/Multitask_data/All_datasets/IR_BL_AG_4CR/Raw data/Transfection_data_files

def flatten_2d_data(tail_names = 'Tail_tail_linker', screen_data = 'Transfection_data_files/2d_amine_dope_screen_data.csv',save_loc = 'Transfection_data_files/flattened_2d_amine_dope_screen.csv'):
	df = pd.read_csv(screen_data)
	names = []
	activities = []
	for i, row in df.iterrows():
		tailname = row[tail_names]
		for col in df.columns:
			if not col == tail_names:
				names.append(str(int(tailname))+'-'+str(int(col)))
				activities.append(row[col])
	flattened = pd.DataFrame.from_dict({'4CR_Lipid_name':names,'log_luminescence':activities})
	flattened.to_csv(save_loc, index = False)

def add_transfection_data_to_structures(raw_data_loc, data_collection_loc = 'Lipids_with_names_and_measurements.csv'):
	collect_df = pd.read_csv(data_collection_loc)
	raw_data_df = pd.read_csv(raw_data_loc)
	lipid_names = list(collect_df['4CR_Lipid_name'])
	if not 'log_luminescence' in collect_df.columns:
		log_lums = [-1 for i in range(len(collect_df))]
	else:
		log_lums = collect_df.log_luminescence
	for i, row in raw_data_df.iterrows():
		name = row['4CR_Lipid_name']
		lum = row['log_luminescence']
		# print(collect_df['4CR_Lipid_name'].index(name))
		log_lums[lipid_names.index(name)] = lum
	collect_df['log_luminescence'] = log_lums
	collect_df.to_csv('Lipids_with_names_and_measurements.csv', index = False)

# add_transfection_data_to_structures('Transfection_data_files/flattened_2d_amine_dope_screen.csv')
# add_transfection_data_to_structures('Transfection_data_files/log_lum_values_384_screen.csv')
add_transfection_data_to_structures('Transfection_data_files/IM_screen_data.csv')


