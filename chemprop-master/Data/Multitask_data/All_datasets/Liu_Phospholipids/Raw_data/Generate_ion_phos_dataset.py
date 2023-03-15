from PIL import Image
import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors, MolFromSmiles

library_header = 'Liu_phospholipids_'

def add_screen_data():
	structures = pd.read_csv('Structures.csv')
	screen_df = pd.read_csv('akinc_screen_silencing_values.csv')
	silencing_vals = [-1 for v in structures.smiles]
	col_names = screen_df.columns
	names = list(structures['Lipid_name'])
	for i, row in screen_df.iterrows():
		amine_number = str(row['Amine'])
		for col in col_names:
			if col[:1]== 'O' or col[:1] == 'N':
				if row[col] > 0:
					silencing_value = row[col]
					name = library_header + 'Amine_'+amine_number + '_' + col
					try:
						index = names.index(name)
						silencing_vals[index] = silencing_value
					except:
						print('Couldn\'t find lipid ',name)
	structures['quantified_delivery'] = silencing_vals
	return structures


def match_lum_values():
	struc_df = pd.read_csv('Ion_phos_structures.csv')
	data_df = pd.read_csv('Ion_phos_screen_lum_values.csv')
	tails = ['P'+str(i) for i in range(4,17)]
	lum_values = [-1 for _ in struc_df.Lipid_name]
	names = list(struc_df.Lipid_name)
	for i, row in data_df.iterrows():
		for tail in tails:
			lipid_name = library_header + row['Header'] + tail
			value = row[tail]
			# print(names.index(lipid_name))
			lum_values[names.index(lipid_name)] = value
	struc_df['quantified_delivery'] = lum_values
	struc_df.to_csv('iphos_strucs_with_activity.csv',index = False)

# For weight ratios: mRNA (FFL) is 1929bp or 655000 g/mol
def get_molecular_weights():
	df = pd.read_csv('iphos_strucs_with_activity.csv')
	mol_weights = []
	for smiles in df.smiles:
		m = MolFromSmiles(smiles)
		mol_weights.append(Descriptors.MolWt(m))
	df['MolWt'] = mol_weights
	df.to_csv('iphos_strucs_with_activity.csv', index = False)


def structure_from_components(amine, tail, tail_num):
	to_return = amine
	for i in range(1,tail_num+1):
		to_return = to_return.replace(str(i),'CCOP(=O)([O-])O'+tail)
	for i in range(tail_num+1,6):
		to_return = to_return.replace(str(i),'[H]')
	if 'Q' in amine and tail_num == 4:
		to_return = to_return.replace('NQ','[N+](CCOP(=O)([O-])O'+tail+')')
	else:
		to_return = to_return.replace('NQ','N')
	return to_return

def tail_range(amine_smiles):
	if 'Q' in amine_smiles:
		return range(1,5)
	for i in [5,4,3,2,1]:
		if str(i) in amine_smiles:
			return range(1,i+1)

def generate_structures():
	to_return = {}
	tail_name = []
	amine_name = []
	lipid_name = []
	tail_count = []
	all_smiles = []
	components = pd.read_csv('Ion_phosph_components.csv')
	tail_names = [v for v in components.Tail if not v == 'None']
	tail_smiles = [v for v in components.tail_smiles if not v == 'None']
	amine_names = [v for v in components.Amine if not v == 'None']
	amine_smiles = [v for v in components.amine_smiles if not v == 'None']
	# print(tail_smiles)
	for t, tail in 	enumerate(tail_names):
		for a, amine in enumerate(amine_names):
			tsmiles = tail_smiles[t]
			asmiles = amine_smiles[a]
			for tail_num in tail_range(asmiles):
			# print(amine)
				all_smiles.append(structure_from_components(asmiles, tsmiles, tail_num))
				tail_name.append(library_header + tail)
				amine_name.append(library_header + amine)
				lipid_name.append(library_header + amine + str(tail_num) + tail)
				tail_count.append(tail_num)
	to_return['Amine'] = amine_name
	to_return['Tail'] = tail_name
	to_return['Lipid_name'] = lipid_name
	to_return['Tail_count'] = tail_count
	to_return['smiles'] = all_smiles
	return to_return

# structures = pd.DataFrame(generate_structures())
# structures.to_csv('Ion_phos_structures.csv', index = False)

# match_lum_values()
get_molecular_weights()

# add_screen_data().to_csv('Structures_with_activities.csv',index = False)
