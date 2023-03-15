import pandas as pd 
import numpy as np

# Can do pwd /Users/jacobwitten/Documents/Next_steps/Anderson/ML for NP design/Chemprop/chemprop-master/Data/Multitask_data/All_datasets/Lei_Miao_3_component

amines = pd.read_csv('Headgroups.csv')
tails = pd.read_csv('Tailgroups.csv')
library_header = 'RM_branched_ester_'
linker = 'CCC(=O)OCCCCCCCC\C=C/CC(CCCCCC)O'

amines = pd.read_csv('All_headgroups_for_raj_in_silico_screen.csv')
tails = pd.read_csv('All_tailgroups_for_raj_in_silico_screen.csv')

def generate_branched_ester(head, tail):
	actual_tail = linker + tail
	struct = head.replace('()','('+actual_tail+')')
	struct = head.replace('*','('+actual_tail+')')
	return struct

def generate_all_lipids():
	all_lipid_dict = {'Lipid_name':[],'Amine':[],'Tail':[],'smiles':[]}
	for a, ami in amines.iterrows():
		for t, tai in tails.iterrows():
			tail = tai['Tailgroup']
			amine = ami['Headgroup']
			amine_smiles = ami['SMILES']
			tail_smiles = tai['SMILES']
			all_lipid_dict['Tail'].append(library_header + tail)
			all_lipid_dict['Amine'].append(library_header + amine)
			all_lipid_dict['smiles'].append(generate_branched_ester(amine_smiles, tail_smiles))
			all_lipid_dict['Lipid_name'].append(library_header+amine+'_'+tail)
	to_return = pd.DataFrame(all_lipid_dict)
	return to_return


def generate_structure_plus_activity_file():
	structure_df = pd.read_csv('Raw_data/Lipid_structures.csv')
	# structure_df.set_index(['Lipid_name'])
	for cell_type in ['HeLa','BDMC','BMDM']:
		structure_df[cell_type] = [np.nan for _ in structure_df.smiles]
		activity_df = pd.read_csv('Raw_data/'+cell_type+'_screen.csv')
		for i, row in activity_df.iterrows():
			for colname in activity_df.columns:
				if colname[:1]=='A':
					# print(row)
					lipid_name = library_header+colname+row.Isocyanate+row.Ketone
					structure_df.loc[structure_df.Lipid_name== lipid_name,cell_type] = row[colname]
	structure_df.to_csv('Raw_data/Structure_with_activities.csv', index = False)
	# for screen_df in (hela_df, bdmc_df, bmcm_df):
	# 	for i, row in screen_df.iterrows():
	# 		for colname in screen_df.columns:
	# 			if colname[:1] == 'A':

def add_transfection_data_to_structures(raw_data_loc, data_collection_loc = '../Raw_data/Structures_and_data.csv'):
	collect_df = pd.read_csv(data_collection_loc)
	raw_data_df = pd.read_csv(raw_data_loc)
	lipid_names = list(collect_df['Lipid_name'])
	if not 'log_luminescence' in collect_df.columns:
		log_lums = [0 for i in range(len(collect_df))]
	else:
		log_lums = collect_df.log_luminescence
	for i, row in raw_data_df.iterrows():
		name = row['Branched_ester_lipid_name']
		lum = row['avg']
		# print(collect_df['4CR_Lipid_name'].index(name))
		log_lums[lipid_names.index(name)] = lum
	collect_df['log_luminescence'] = log_lums
	collect_df.to_csv(data_collection_loc, index = False)


def generate_data_files():
	all_df = pd.read_csv('../Raw_data/Structures_and_data.csv')
	metadata = all_df[['Lipid_name','Amine','Ketone','Isocyanate','Is_cyclic']]
	experiment_data = all_df[['smiles','HeLa','BDMC','BMDM']]
	metadata.to_csv('../individual_metadata.csv', index = False)
	experiment_data.to_csv('../main_data.csv', index = False)

generate_all_lipids().to_csv('in_silico_screen_structures.csv',index = False)


# generate_all_lipids().to_csv('Lipid_structures.csv',index = False)
# add_transfection_data_to_structures('../Raw_data/Branched_ester_lipid_screen.csv')
# generate_structure_plus_activity_file()
# generate_data_files()





