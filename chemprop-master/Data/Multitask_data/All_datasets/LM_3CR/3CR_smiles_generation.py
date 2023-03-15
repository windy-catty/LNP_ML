import pandas as pd 
import numpy as np

# Can do pwd /Users/jacobwitten/Documents/Next_steps/Anderson/ML for NP design/Chemprop/chemprop-master/Data/Multitask_data/All_datasets/Lei_Miao_3_component

isocyanates = pd.read_csv('Raw_data/Miao_isocyanate.csv')
ketones = pd.read_csv('Raw_data/Miao_ketone.csv')
amines = pd.read_csv('Raw_data/Miao_amine.csv')
library_header = 'LM_3CR_'

def generate_cyclic_smiles(amine, iso, ketone):
	to_return = 'C1' + iso + 'N=C'+ketone+'N1('+amine+')'
	return to_return

def generate_noncyclic_smiles(amine, iso, ketone):
	to_return = 'N(' + iso + ')C(=O)'+ketone+amine
	return to_return

def generate_all_lipids():
	all_lipid_dict = {'Lipid_name':[],'Amine':[],'Ketone':[],'Isocyanate':[],'smiles':[],'Is_cyclic':[]}
	for a, ami in amines.iterrows():
		for k, ket in ketones.iterrows():
			for i, iso in isocyanates.iterrows():
				ketone = ket['Component']
				isocyanate = iso['Component']
				amine = ami['Component']
				ketone_smiles = ket['SMILES']
				all_lipid_dict['Ketone'].append(library_header + ketone)
				all_lipid_dict['Isocyanate'].append(library_header + isocyanate)
				all_lipid_dict['Amine'].append(library_header + amine)
				if ami['Cyclic'] == 'None' or iso['Cyclic'] == 'None':
					amine_smiles = ami['Noncyclic']
					isocyanate_smiles = iso['Noncyclic']
					all_lipid_dict['Is_cyclic'].append('No')
					all_lipid_dict['smiles'].append(generate_noncyclic_smiles(amine_smiles, isocyanate_smiles, ketone_smiles))
				else:
					amine_smiles = ami['Cyclic']
					isocyanate_smiles = iso['Cyclic']
					all_lipid_dict['Is_cyclic'].append('Yes')
					all_lipid_dict['smiles'].append(generate_cyclic_smiles(amine_smiles, isocyanate_smiles, ketone_smiles))
				all_lipid_dict['Lipid_name'].append(library_header+amine+isocyanate+ketone)
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



def generate_data_files():
	all_df = pd.read_csv('Raw_data/Structure_with_activities.csv')
	metadata = all_df[['Lipid_name','Amine','Ketone','Isocyanate','Is_cyclic']]
	experiment_data = all_df[['smiles','HeLa','BDMC','BMDM']]
	metadata.to_csv('individual_metadata.csv', index = False)
	experiment_data.to_csv('main_data.csv', index = False)

generate_all_lipids().to_csv('Raw_data/Lipid_structures.csv',index = False)
generate_structure_plus_activity_file()
generate_data_files()





