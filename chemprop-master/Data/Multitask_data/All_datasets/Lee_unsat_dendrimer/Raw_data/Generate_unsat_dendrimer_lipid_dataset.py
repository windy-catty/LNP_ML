from PIL import Image
import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors, MolFromSmiles

library_header = 'Lee_unsat_dendrimer_'

def add_screen_data():
	structures = pd.read_csv('dendrimer_structures.csv')
	screen_df = pd.read_csv('unsat_dendrimer_lum_results.csv')
	silencing_vals = [-1 for v in structures.smiles]
	col_names = screen_df.columns
	names = list(structures['Lipid_name'])
	for i, row in screen_df.iterrows():
		tail_type = str(row['Tail'])
		for col in col_names:
			if not col=='Tail':
				silencing_value = row[col]
				name = library_header + col + '-' + tail_type
				# try:
				index = names.index(name)
				silencing_vals[index] = silencing_value
				# except:
					# print('Couldn\'t find lipid ',name)
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


def structure_from_components(amine, tail):
	to_return = amine
	to_return = to_return.replace('*',"CCC(=O)OCCOC(=O)C(C)C"+tail)
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
	all_smiles = []
	all_tail_nums = []
	all_mol_wts = []
	components = pd.read_csv('Lee_unsat_dendrimer_components.csv')
	tail_names = [v for v in components.Tail if not v == 'None']
	tail_smiles = [v for v in components.tail_smiles if not v == 'None']
	amine_names = [v for v in components.Amine if not v == 'None']
	amine_smiles = [v for v in components.amine_smiles if not v == 'None']
	# print(tail_smiles)
	for t, tail in 	enumerate(tail_names):
		for a, amine in enumerate(amine_names):
			tsmiles = tail_smiles[t]
			asmiles = amine_smiles[a]
			dendrimer_smiles = structure_from_components(asmiles, tsmiles)
			all_smiles.append(dendrimer_smiles)
			tail_name.append(library_header + tail)
			amine_name.append(library_header + amine)
			lipid_name.append(library_header + amine + '-'+tail)
			# print(amine+'-'+tail)
			all_tail_nums.append(asmiles.count('*'))
			m = MolFromSmiles(dendrimer_smiles)
			all_mol_wts.append(Descriptors.MolWt(m))
			# print(dendrimer_smiles)
	to_return['Amine'] = amine_name
	to_return['Tail'] = tail_name
	to_return['Lipid_name'] = lipid_name
	to_return['Num tails'] = all_tail_nums
	to_return['MolWt'] = all_mol_wts
	to_return['smiles'] = all_smiles
	return to_return

# structures = pd.DataFrame(generate_structures())
# structures.to_csv('dendrimer_structures.csv', index = False)

# match_lum_values()
# get_molecular_weights()

add_screen_data().to_csv('unsat_endrimer_sructures_with_activities.csv',index = False)
