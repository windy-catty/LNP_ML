import pandas as pd 
import numpy as np

# Can do pwd /Users/jacobwitten/Documents/Next_steps/Anderson/ML for NP design/Chemprop/chemprop-master/Data/Multitask_data/All_datasets/Lei_Miao_3_component

all_components = pd.read_csv('4CR_Ketone_components_adamantane.csv')
isos = [iso for iso in all_components.iso_smiles if not iso == 'None']
tail1s = [tail for tail in all_components.tail_1_1_smiles if not tail == 'None']
tail2s = [tail for tail in all_components.tail_1_2_smiles if not tail == 'None']
carboxys  = [carboxy for carboxy in all_components.carboxy_smiles if not carboxy == 'None']
amines  = [amine for amine in all_components.amine_smiles if not amine == 'None']


iso_names = [iso for iso in all_components.Isocyanide if not iso == 'None']
carboxy_names = [carboxy for carboxy in all_components.Carboxylic_acid if not carboxy == 'None']
ketone_names = [ket for ket in all_components.Ketone if not ket == 'None']
amine_names  = [amine for amine in all_components.Amine if not amine == 'None']
library_header = 'IR_4CR_Ketone_'


def generate_smiles(amine, iso, tail_1p1, tail_1p2, carboxy):
	smiles = amine + '('+carboxy + ')'
	smiles = smiles + 'C(CCCC(=O)'+tail_1p1+')(CCCC(=O)'+tail_1p2+')C(=O)'+iso
		# smiles = smiles.replace('Cyc'+str(i),'C(T1)(T2)C=NC'+str(i)+'(C(=O)OIso')
	return smiles


def generate_all_lipids():
	all_lipid_dict = {'Lipid_name':[],'Amine':[],'Ketone':[],'Isocyanide':[],'Carboxylic_acid':[],'smiles':[],'quantified_delivery':[]}
	for a, ami in enumerate(amine_names):
		for k, ket in enumerate(ketone_names):
			for i, iso in enumerate(iso_names):
				for c, car in enumerate(carboxy_names):
					amine_smiles = amines[a]
					iso_smiles = isos[i]
					car_smiles = carboxys[c]
					tail_1p1 = tail1s[k]
					tail_1p2 = tail2s[k]
					full_smiles = generate_smiles(amine_smiles, iso_smiles, tail_1p1, tail_1p2, car_smiles)
					all_lipid_dict['Ketone'].append(library_header + ket)
					all_lipid_dict['Isocyanide'].append(library_header + iso)
					all_lipid_dict['Amine'].append(library_header + ami)
					all_lipid_dict['Carboxylic_acid'].append(library_header+car)
					all_lipid_dict['smiles'].append(full_smiles)
					all_lipid_dict['Lipid_name'].append(library_header+ami+'_'+iso+'_'+ket+'_'+car)
					all_lipid_dict['quantified_delivery'].append(0)
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

generate_all_lipids().to_csv('Lipid_structures_with_fake_activities_adamantane.csv',index = False)
# generate_structure_plus_activity_file()
# generate_data_files()





