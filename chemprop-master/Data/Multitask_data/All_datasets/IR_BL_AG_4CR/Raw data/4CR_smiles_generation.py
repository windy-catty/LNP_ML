import pandas as pd 
import numpy as np

# Can do pwd /Users/jacobwitten/Documents/Next_steps/Anderson/ML for NP design/Chemprop/chemprop-master/Data/Multitask_data/All_datasets/Lei_Miao_3_component

aldehydes = pd.read_csv('ir_bl_ag_4cr_aldehydes.csv')
carboxylic_acids = pd.read_csv('ir_bl_ag_4cr_carboxylic_acids.csv')
amines = pd.read_csv('ir_bl_ag_4cr_amines.csv')
isocyanides = pd.read_csv('ir_bl_ag_4cr_isocyanides.csv')
library_header = 'IR_4CR_aldehyde_'

# & should get the isocyanide and aldehyde attached, * should be swapped for the carboxylic acid
def generate_4cr_smiles(amine, iso, aldehyde, carboxy):
	smiles = amine
	smiles = smiles.replace('*','(C(=O)'+carboxy+')')
	other_addition = 'C('+aldehyde+')C(=O)N'+iso
	smiles = smiles.replace('&','('+other_addition+')')
	return smiles


def generate_all_lipids():
	# Naming: aldehydecarboxylicacidisocyanide-amine
	all_lipid_dict = {'Lipid_name':[],'4CR_Lipid_name':[],'Amine':[],'Isocyanide':[],'Aldehyde':[],'Carboxylic_acid':[],'smiles':[]}
	for a, ami in amines.iterrows():
		for c, car in carboxylic_acids.iterrows():
			for i, iso in isocyanides.iterrows():
				for al, ald in aldehydes.iterrows():
					carboxylic_acid = car['smiles']
					isocyanide = iso['smiles']
					amine = ami['smiles']
					aldehyde = ald['smiles']
					all_lipid_dict['Amine'].append(ami['Amine'])
					all_lipid_dict['Isocyanide'].append(iso['Isocyanide'])
					all_lipid_dict['Aldehyde'].append(ald['Aldehyde'])
					all_lipid_dict['Carboxylic_acid'].append(car['Carboxylic_acid'])
					all_lipid_dict['smiles'].append(generate_4cr_smiles(amine, isocyanide, aldehyde, carboxylic_acid))
					lipid_name = str(ald['Number'])+str(car['Number'])+str(iso['Number'])+'-'+str(ami['Number'])
					all_lipid_dict['Lipid_name'].append(library_header+lipid_name)
					all_lipid_dict['4CR_Lipid_name'].append(lipid_name)
	to_return = pd.DataFrame(all_lipid_dict)
	return to_return


def generate_structure_plus_activity_file():
	structure_df = pd.read_csv('Lipid_structures.csv')
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

generate_all_lipids().to_csv('Lipid_structures.csv',index = False)
# generate_structure_plus_activity_file()
# generate_data_files()





