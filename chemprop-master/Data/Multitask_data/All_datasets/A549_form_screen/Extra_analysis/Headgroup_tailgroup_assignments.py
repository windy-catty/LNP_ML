from rdkit.Chem import Descriptors, MolFromSmiles, MolToSmiles
import numpy as np
import pandas as pd






def generate_branched_ester(head, tail):
	actual_tail = linker + tail
	struct = head.replace('()','('+actual_tail+')')
	return struct

def generate_reductive_amination(head, tail, red_amin_linker = 'CC(C(O)CCCCCO^)CCCCO^'):
	actual_tail = red_amin_linker.replace('^',tail)
	return head.replace('*','('+actual_tail+')')

def generate_all_lipids(library_header, amines, tails, is_branched_ester):
	all_lipid_dict = {'Lipid_name':[],'Amine':[],'Tail':[],'smiles':[]}
	for a, ami in amines.iterrows():
		for t, tai in tails.iterrows():
			tail = tai['Tailgroup']
			amine = ami['Headgroup']
			amine_smiles = ami['SMILES']
			tail_smiles = tai['SMILES']
			all_lipid_dict['Tail'].append(library_header + str(tail))
			all_lipid_dict['Amine'].append(library_header + str(amine))
			if is_branched_ester:
				all_lipid_dict['smiles'].append(generate_branched_ester(amine_smiles, tail_smiles))
			else:
				all_lipid_dict['smiles'].append(generate_reductive_amination(amine_smiles, tail_smiles))
			all_lipid_dict['Lipid_name'].append(library_header+str(amine)+'_'+str(tail))
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

def add_red_amin_metadata():
	amines = pd.read_csv('Idris_red_amin_amines.csv')
	tails = pd.read_csv('Idris_red_amin_tails.csv')
	current_data = pd.read_csv('head_tail_assigned.csv')
	library_header = 'IR_red_amin_'
	all_lipid_df = generate_all_lipids(library_header, amines, tails, False)
	all_lipid_df.to_csv('all_possible_red_amin_structures.csv', index = False)
	all_mols = []
	for i, row in all_lipid_df.iterrows():
		all_mols.append(MolToSmiles(MolFromSmiles(row['smiles'])))
	heads = current_data.Amine
	tails = current_data.Tail
	lipid_names = current_data.Lipid_name
	correct_smiles = current_data.smiles
	library = current_data.Library_ID
	correction_dict = {}

	for i, smiles in enumerate(correct_smiles):
		# smiles = row['smiles']
		if library[i] == 'IR_Reductive_amination':
			if smiles in correction_dict.keys():
				smiles = correction_dict[smiles]
				# print('correcting!')
			smiles = MolToSmiles(MolFromSmiles(smiles))
			if smiles in correction_dict.keys():
				smiles = correction_dict[smiles]
				# print('correcting!')
			smiles = MolToSmiles(MolFromSmiles(smiles))
			try:
				idx = all_mols.index(smiles)
				heads[i] = all_lipid_df['Amine'][idx]
				tails[i] = all_lipid_df['Tail'][idx]
			except Exception as e:
				heads[i] = 'Control_lipid???'
				tails[i] = 'Control_lipid'
			# heads.append(head_to_add)
			# tails.append(tail_to_add)
			ta = tails[i].split('_')[len(tails[i].split('_'))-1]
			lipid_names[i] = (str(heads[i]+'_'+ta))
			correct_smiles[i] = smiles
		if lipid_names[i]=='Control_lipid???_lipid':
			heads[i] = 'na'
			tails[i] = 'na'
			lipid_names[i] = 'other'
			library[i] = 'other'
	current_data['Amine'] = heads
	current_data['Tail'] = tails
	current_data['Lipid_name'] = lipid_names
	current_data['smiles'] = correct_smiles
	current_data.to_csv('head_tail_assigned.csv',index = False)

def add_extra_data():
	extra = pd.read_csv('Extra_data_to_tack_on.csv')
	current_data = pd.read_csv('head_tail_assigned.csv')
	raj_structures = pd.read_csv('all_possible_structures.csv')
	idris_structures = pd.read_csv('all_possible_red_amin_structures.csv')
	rm_header = 'RM_branched_ester_'
	ir_header = 'IR_red_amin_'
	heads = list(current_data.Amine)
	tails = list(current_data.Tail)
	lipid_names = list(current_data.Lipid_name)
	correct_smiles = list(current_data.smiles)
	library = list(current_data.Library_ID)
	for i, row in extra.iterrows():
		# print(row['Library'])
		library.append(row['Library'])
		if row['Library'] == 'RM_Michael_addition_branched':
			# print('here!!!\n\n')
			header = rm_header
			search_df = raj_structures
		else:
			header = ir_header
			search_df = idris_structures
		heads.append(header+row.Amine)
		tails.append(header+row.Tail)
		lipid_name = header+row.Amine+'_'+row.Tail
		lipid_names.append(lipid_name)
		idx = list(search_df.Lipid_name).index(lipid_name)
		correct_smiles.append(search_df.smiles[idx])
	new_data = pd.DataFrame()
	new_data['Amine'] = heads
	new_data['Tail'] = tails
	new_data['Lipid_name'] = lipid_names
	new_data['smiles'] = correct_smiles
	new_data['Library_ID'] = library
	new_data.to_csv('head_tail_assigned_with_extra_data.csv',index = False)

def add_Raj_ester_metadata():
	amines = pd.read_csv('Raj_ester_Headgroups.csv')
	tails = pd.read_csv('Raj_ester_Tailgroups.csv')
	smiles_list = pd.read_csv('../main_data.csv')
	metadata = pd.read_csv('../individual_metadata.csv')
	library_header = 'RM_branched_ester_'
	linker = 'CCC(=O)OCCCCCCCC\C=C/CC(CCCCCC)O'

	all_lipid_df = generate_all_lipids()
	all_lipid_df.to_csv('all_possible_structures.csv',index = False)

	all_mols = []
	for i, row in all_lipid_df.iterrows():
		all_mols.append(MolToSmiles(MolFromSmiles(row['smiles'])))
	heads = []
	tails = []
	lipid_names = []
	correct_smiles = []
	correction_dict = {}
	correction_dict['O=C(OCCCCCCCC/C=C\CC(CCCCCC)OC(C(CCCCCCCC)CCCCCC)=O)CCN(C)CCCCCCN(CCC(OCCCCCCCC/C=C\CC(CCCCC)OC(C(CCCCCCCC)CCCCCC)=O)=O)C'] = 'CN(CCC(=O)OCCCCCCCC\C=C/CC(CCCCCC)OC(=O)C(CCCCCC)CCCCCCCC)CCCCCCN(CCC(=O)OCCCCCCCC\C=C/CC(CCCCCC)OC(=O)C(CCCCCC)CCCCCCCC)C'
	correction_dict['CN(CCC(OCCCCCCCC/C=C\CC(CCCCCC)OC(CC/C=C\CCCCC)=O)=O)CCCN(C)CCCN(CCC(OCCCCCCCC/C=C\CC(CCCCC)OC(CC/C=C\CCCCC)=O)=O)C'] = 'CN(CCC(=O)OCCCCCCCC\C=C/CC(CCCCCC)OC(=O)CC/C=C\CCCCC)CCCN(C)CCCN(CCC(=O)OCCCCCCCC\C=C/CC(CCCCCC)OC(=O)CC/C=C\CCCCC)C'
	correction_dict['CN(CCC(OCCCCCCCC/C=C\CC(CCCCCC)OC(C(CCCCCCCC)CCCCCC)=O)=O)CCCN(C)CCCN(CCC(OCCCCCCCC/C=C\CC(CCCCC)OC(C(CCCCCCCC)CCCCCC)=O)=O)C'] = 'CN(CCC(OCCCCCCCC/C=C\CC(CCCCCC)OC(C(CCCCCCCC)CCCCCC)=O)=O)CCCN(C)CCCN(CCC(OCCCCCCCC/C=C\CC(CCCCCC)OC(C(CCCCCCCC)CCCCCC)=O)=O)C'
	correction_dict['CCCCCCCCC(CCCCCC)C(=O)OC(C/C=C\CCCCCCCCOC(=O)CCN(C)CCCN(C)CCCN(C)CCC(=O)OCCCCCCCC/C=C\CC(CCCCCC)OC(=O)C(CCCCCC)CCCCCCCC)CCCCC'] = 'CCCCCCCCC(CCCCCC)C(=O)OC(C/C=C\CCCCCCCCOC(=O)CCN(C)CCCN(C)CCCN(C)CCC(=O)OCCCCCCCC/C=C\CC(CCCCCC)OC(=O)C(CCCCCC)CCCCCCCC)CCCCCC'
	correction_dict['CCCCCCCCC(CCCCCC)C(=O)OC(C/C=C\CCCCCCCCOC(=O)CCN(C)CCCCCCN(C)CCC(=O)OCCCCCCCC/C=C\CC(CCCCCC)OC(=O)C(CCCCCC)CCCCCCCC)CCCCC'] = 'CCCCCCCCC(CCCCCC)C(=O)OC(C/C=C\CCCCCCCCOC(=O)CCN(C)CCCCCCN(C)CCC(=O)OCCCCCCCC/C=C\CC(CCCCCC)OC(=O)C(CCCCCC)CCCCCCCC)CCCCCC'
	correction_dict['CCCCC/C=C\CCC(=O)OC(C/C=C\CCCCCCCCOC(=O)CCN(C)CCCN(C)CCCN(C)CCC(=O)OCCCCCCCC/C=C\CC(CCCCCC)OC(=O)CC/C=C\CCCCC)CCCCC'] = 'CCCCC/C=C\CCC(=O)OC(C/C=C\CCCCCCCCOC(=O)CCN(C)CCCN(C)CCCN(C)CCC(=O)OCCCCCCCC/C=C\CC(CCCCCC)OC(=O)CC/C=C\CCCCC)CCCCCC'
	correction_dict['CCCCCCC/C=C\CCCCCCCCC(=O)OC(C/C=C\CCCCCCCCOC(=O)CCN(CCCN(CC)CC)CCC(=O)OCCCCCCCC/C=C\CC(CCCCCC)OC(=O)CCCCCCCC/C=C\CCCCCCC)CCCCCC'] = 'N(CCC(=O)OCCCCCCCC\C=C/CC(CCCCCC)OC(=O)CCCCCCC/C=C\CCCCCCCC)(CCC(=O)OCCCCCCCC\C=C/CC(CCCCCC)OC(=O)CCCCCCC/C=C\CCCCCCCC)CCCN(CC)CC'
	correction_dict['CCCCCCCCC/C=C\CCCCCCCC(=O)OC(C/C=C\CCCCCCCCOC(=O)CCN(CCCN(C)C)CCC(=O)OCCCCCCCC/C=C\CC(CCCCCC)OC(=O)CCCCCCC/C=C\CCCCCCCCC)CCCCCC'] = 'N(CCC(=O)OCCCCCCCC\C=C/CC(CCCCCC)OC(=O)CCCCCCC/C=C\CCCCCCCC)(CCC(=O)OCCCCCCCC\C=C/CC(CCCCCC)OC(=O)CCCCCCC/C=C\CCCCCCCC)CCCN(C)C'
	correction_dict['CCCCCC/C=C\CCCCCCCC(=O)OC(C/C=C\CCCCCCCCOC(=O)CCN(CCCN(CC)CC)CCC(=O)OCCCCCCCC/C=C\CC(CCCCCC)OC(=O)CCCCCCC/C=C\CCCCCCC)CCCCCC'] = 'CCCCCC/C=C\CCCCCCCC(=O)OC(C/C=C\CCCCCCCCOC(=O)CCN(CCCN(CC)CC)CCC(=O)OCCCCCCCC/C=C\CC(CCCCCC)OC(=O)CCCCCCC/C=C\CCCCCC)CCCCCC'
	correction_dict['CCCCCC/C=C\CCCCCCCC(=O)OC(C/C=C\CCCCCCCCOC(=O)CCN(CCCN(C)C)CCC(=O)OCCCCCCCC/C=C\CC(CCCCCC)OC(=O)CCCCCCC/C=C\CCCCCCC)CCCCCC'] = 'CCCCCC/C=C\CCCCCCCC(=O)OC(C/C=C\CCCCCCCCOC(=O)CCN(CCCN(C)C)CCC(=O)OCCCCCCCC/C=C\CC(CCCCCC)OC(=O)CCCCCCC/C=C\CCCCCC)CCCCCC'
	# assignment_dict = 
	for i, row in smiles_list.iterrows():
		smiles = row['smiles']
		if smiles in correction_dict.keys():
			smiles = correction_dict[smiles]
			# print('correcting!')
		smiles = MolToSmiles(MolFromSmiles(smiles))
		if smiles in correction_dict.keys():
			smiles = correction_dict[smiles]
			# print('correcting!')
		smiles = MolToSmiles(MolFromSmiles(smiles))
		while('C(=O)CC/C=C\CCCCCC' in smiles):
			smiles = smiles.replace('C(=O)CC/C=C\CCCCCC','C(=O)CC/C=C\CCCCC')
			smiles = MolToSmiles(MolFromSmiles(smiles))
		while('C(=O)CCCCCCCC/C=C\CCCC' in smiles):
			smiles = smiles.replace('C(=O)CCCCCCCC/C=C\CCCC','C(=O)CCCCCCC/C=C\CCCC')
			smiles = MolToSmiles(MolFromSmiles(smiles))
		while('CCCC/C=C\CCCCCCCCC(=O)' in smiles):
			smiles = smiles.replace('CCCC/C=C\CCCCCCCCC(=O)','CCCC/C=C\CCCCCCCC(=O)')
			smiles = MolToSmiles(MolFromSmiles(smiles))
		try:
			idx = all_mols.index(smiles)
			head_to_add = all_lipid_df['Amine'][idx]
			tail_to_add = all_lipid_df['Tail'][idx]
		except Exception as e:
			# print(e)
			if smiles == 'N(CCC(=O)OCCCCCCCC\C=C/CC(CCCCCC)OC(=O)CCCCCCC/C=C\CCCCCCCC)(CCC(=O)OCCCCCCCC\C=C/CC(CCCCCC)OC(=O)CCCCCCC/C=C\CCCCCCCC)CCCN(CC)CC':
				print('?????')
				print(smiles)
				print(all_mols[64])
			head_to_add = 'Control_lipid???'
			tail_to_add = 'Control_lipid'
		heads.append(head_to_add)
		tails.append(tail_to_add)
		ta = tail_to_add.split('_')[len(tail_to_add.split('_'))-1]
		lipid_names.append(str(head_to_add+'_'+ta))

		correct_smiles.append(smiles)
	metadata['Amine'] = heads
	metadata['Tail'] = tails
	metadata['Lipid_name'] = lipid_names
	metadata['smiles'] = correct_smiles
	metadata.to_csv('head_tail_assigned.csv',index = False)
# add_red_amin_metadata()
add_extra_data()
# generate_all_lipids().to_csv('Lipid_structures.csv',index = False)
# add_transfection_data_to_structures('../Raw_data/Branched_ester_lipid_screen.csv')
# # generate_structure_plus_activity_file()
# generate_data_files()

