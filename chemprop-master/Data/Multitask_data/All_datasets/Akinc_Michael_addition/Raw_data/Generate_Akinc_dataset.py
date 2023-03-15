from PIL import Image
import numpy as np
import pandas as pd

library_header = 'Akinc_Michael_'

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
			# else:
				# print('Couldnt find column: ',col)
	structures['quantified_delivery'] = silencing_vals
	return structures


def structure_from_components(amine, tail):
	to_return = amine
	to_return = to_return.replace('*','('+tail+')')
	return to_return

def generate_structures():
	to_return = {}
	tail_name = []
	amine_name = []
	lipid_name = []
	all_smiles = []
	components = pd.read_csv('Akinc_components.csv')
	tail_names = [v for v in components.Tail_name if not v == 'None']
	tail_smiles = [v for v in components.tail_smiles if not v == 'None']
	amine_names = [v for v in components.Amine_name if not v == 'None']
	amine_smiles = [v for v in components.amine_smiles if not v == 'None']
	# print(tail_smiles)
	for t, tail in 	enumerate(tail_names):
		for a, amine in enumerate(amine_names):
			# print(amine)
			tsmiles = tail_smiles[t]
			asmiles = amine_smiles[a]
			all_smiles.append(structure_from_components(asmiles, tsmiles))
			tail_name.append(library_header + tail)
			amine_name.append(library_header + amine)
			lipid_name.append(library_header + amine + '_' + tail)
	to_return['Amine_name'] = amine_name
	to_return['Tail_name'] = tail_name
	to_return['Lipid_name'] = lipid_name
	to_return['smiles'] = all_smiles
	return to_return

structures = pd.DataFrame(generate_structures())
structures.to_csv('structuress.csv', index = False)

add_screen_data().to_csv('Structures_with_activities.csv',index = False)
