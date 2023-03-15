import pandas as pd 
import numpy as np
from rdkit import Chem
import os
import sys
import random
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

# Can do pwd /Users/jacobwitten/Documents/Next_steps/Anderson/ML for NP design/Chemprop/chemprop-master/Data/Multitask_data/All_datasets/Lei_Miao_3_component

all_components = pd.read_csv('All_MCR_no_nh_achiral_with_oh_plus_cooh.csv')
isos = [iso for iso in all_components.iso_smiles if not iso == 'None' and str(iso)[:1] == 'N']
amines = [amine for amine in all_components.amine_smiles if not amine == 'None']
ohs = [oh for oh in all_components.term_oh_smiles if str(oh)[:1]=='O']
coohs = [cooh for cooh in all_components.term_cooh_smiles if str(cooh)[:4]=='C(=O']
# print(len(ohs))
# # print(ohs[-1])
# # print(ohs[-5])
# print(ohs[29])
# print(ohs[27])
# print(ohs[22])
# print(ohs[19])
print(len(coohs))
cooh_linkers  = [cooh for cooh in all_components.cooh_linker_smiles if str(cooh)[:4]=='C(=O']
oh_linkers  = [oh for oh in all_components.oh_linker_smiles if str(oh)[:1]=='O']

# isos = isos[:3]
# amines = amines[:3]
# ohs = ohs[:3]
# coohs = coohs[43:46]
# cooh_linkers = cooh_linkers[10:12]
# oh_linkers = oh_linkers[-2:]

# library_header = 'IR_4CR_Ketone_'

def ring_number_adjust_all(all_smiles_to_adjust):
    to_return = []
    current = ''
    for smiles in all_smiles_to_adjust:
        if not smiles is None:
            adjusted = ring_number_adjust(current, smiles)
            to_return.append(adjusted)
            current = current + adjusted
        else:
            to_return.append(None)
    return to_return


def ring_number_adjust(prev_smiles, new_smiles):
    # Makes sure that cycle numbers don't conflict
    ring_ids  = [str(i) for i in range(1,10)]+['%'+str(i) for i in range(10,100)]
	#     print(ring_ids)
    unused_ring_ids = [val for val in ring_ids if val not in prev_smiles]
    try:
    	new_ring_ids = [val for val in ring_ids if val in new_smiles]
    except:
    	print(prev_smiles)
    	print(new_smiles)
    	new_ring_ids = [val for val in ring_ids if val in new_smiles]
    # all_used_ids = [val for val in old_ring_ids]
    rename_map = {}
    for i,ring in enumerate(new_ring_ids):
        rename_map[ring] = unused_ring_ids[i]
    to_return = ''
    for i,char in enumerate(new_smiles):
        if char in rename_map.keys():
            to_return = to_return + rename_map[char]
        else:
            to_return = to_return + char
    return to_return

def generate_4cr_aldehyde_smiles_name_and_components(amine, iso, coohal, cooh, oh_after_coohal = None, oh_after_cooh = None):
	#     these are the indices
    aldehyde_name = 'c'+str(coohal)
    if not oh_after_coohal is None:
        aldehyde_name = aldehyde_name + '&h'+str(oh_after_coohal)
    cooh_name = 'c'+str(cooh)
    if not oh_after_cooh is None:
        cooh_name = cooh_name + '&h'+str(oh_after_cooh)
    amine_name = 'a'+str(amine)
    iso_name = 'i' + str(iso)
    name = '4K:'+amine_name+'|'+iso_name+'|'+aldehyde_name+'|'+cooh_name
    return name, amine_name, iso_name, aldehyde_name, cooh_name

def generate_4cr_aldehyde_smiles(amine, iso, coohal, cooh, oh_after_coohal = None, oh_after_cooh = None):
    amine, iso, coohal, cooh, oh_after_coohal, oh_after_cooh = ring_number_adjust_all([amine, iso, coohal, cooh, oh_after_coohal, oh_after_cooh])
    if '^' in coohal:
        coohal = coohal.replace('^',oh_after_coohal)
    if '^' in cooh:
        cooh = cooh.replace('^',oh_after_cooh)
    smiles = amine + '('+cooh + ')'
    smiles = smiles + 'C(CCCCCO'+coohal+')C(=O)'+iso
        # smiles = smiles.replace('Cyc'+str(i),'C(T1)(T2)C=NC'+str(i)+'(C(=O)OIso')
    return smiles

def generate_4cr_ketone_smiles_name_and_components(amine, iso, oh1, oh2, cooh, oh_after_cooh = None, cooh_after_oh_1 = None, cooh_after_oh_2 = None):
	#     these are the indices
    oh1_name = 'h'+str(oh1)
    if not cooh_after_oh_1 is None:
        oh1_name = oh1_name + '&c'+str(cooh_after_oh_1)
    oh2_name = 'h'+str(oh2)
    if not cooh_after_oh_2 is None:
        oh2_name = oh2_name + '&c'+str(cooh_after_oh_2)
    cooh_name = 'c'+str(cooh)
    if not oh_after_cooh is None:
        cooh_name = cooh_name + '&h'+str(oh_after_cooh)
    ketone_name = oh1_name+'+'+oh2_name
    amine_name = 'a'+str(amine)
    iso_name = 'i' + str(iso)
    name = '4K:'+amine_name+'|'+iso_name+'|'+ketone_name+'|'+cooh_name
    return name, amine_name, iso_name, ketone_name, cooh_name

def generate_4cr_ketone_smiles(amine, iso, oh1, oh2, cooh, oh_after_cooh = None, cooh_after_oh_1 = None, cooh_after_oh_2 = None, dechiralize = True):
    amine, iso, oh1, oh2, cooh, oh_after_cooh, cooh_after_oh_1, cooh_after_oh_2 = ring_number_adjust_all([amine, iso, oh1, oh2, cooh, oh_after_cooh, cooh_after_oh_1, cooh_after_oh_2])
    if '*' in oh1:
        oh1 = oh1.replace('*',cooh_after_oh_1)
    if '*' in oh2:
        oh2 = oh2.replace('*',cooh_after_oh_2)
    if '^' in cooh:
        cooh = cooh.replace('^',oh_after_cooh)
    smiles = amine + '('+cooh + ')'
    smiles = smiles + 'C(CCCC(=O)'+oh1+')(CCCC(=O)'+oh2+')C(=O)'+iso
    if dechiralize:
    	smiles = smiles.replace('@','')
        # smiles = smiles.replace('Cyc'+str(i),'C(T1)(T2)C=NC'+str(i)+'(C(=O)OIso')
    return smiles

def is_valid_structure(n_isomers, sp3_ns = 1):
	# return True
	return n_isomers<3
	# otherwise I leave out the pyrazoles and stuff!!!!!
	# return n_isomers < 3 and sp3_ns > 0

def generate_simple_4cr_ketones():
	all_lipid_dict = {'Lipid_name':[],'Amine':[],'Ketone':[],'Isocyanide':[],'Carboxylic_acid':[],'smiles':[],'n_stereoisomers':[],'sp3_nitrogens':[],'quantified_delivery':[]}
	for a, ami in enumerate(amines):
		print('on to amine #',a,': ',ami)
		for o1, oh1 in enumerate(ohs):
			# print('on to oh #',o1,': ',oh1)
			for o2,oh2 in enumerate(ohs):
				if oh2 >= oh1:
					for i, iso in enumerate(isos):
						for c, cooh in enumerate(coohs):
							print(ami,', ',str(a))
							print(oh1,', ',str(o1))
							print(oh2,', ',str(o2))
							print(iso,', ',str(i))
							print(cooh,', ',str(c))
							full_smiles = generate_4cr_ketone_smiles(ami, iso, oh1, oh2, cooh)
							mol = Chem.MolFromSmiles(full_smiles)
							n_isomers = len(tuple(EnumerateStereoisomers(mol)))
							sp3_ns = 0
							for x in mol.GetAtoms():
								if str(x.GetSymbol())=='N' and str(x.GetHybridization()) == 'SP3':
									sp3_ns += 1
							if is_valid_structure(n_isomers, sp3_ns):
								print(full_smiles)
								name, amine_name, iso_name, ketone_name, cooh_name = generate_4cr_ketone_smiles_name_and_components(a+1,i+1,o1+1,o2+1,c+1)
								all_lipid_dict['Ketone'].append(ketone_name)
								all_lipid_dict['Isocyanide'].append(iso_name)
								all_lipid_dict['Amine'].append(amine_name)
								all_lipid_dict['Carboxylic_acid'].append(cooh_name)
								all_lipid_dict['smiles'].append(full_smiles)
								all_lipid_dict['Lipid_name'].append(name)
								all_lipid_dict['quantified_delivery'].append(0)
								
								all_lipid_dict['n_stereoisomers'].append(n_isomers)
								all_lipid_dict['sp3_nitrogens'].append(sp3_ns)
	to_return = pd.DataFrame(all_lipid_dict)
	return to_return

def generate_symmetric_oh_linker_4cr_ketones(fraction = 2, fixed_number_to_screen = None):
	all_lipid_dict = {'Lipid_name':[],'Amine':[],'Ketone':[],'Isocyanide':[],'Carboxylic_acid':[],'smiles':[],'n_stereoisomers':[],'sp3_nitrogens':[],'quantified_delivery':[]}
	amis = {'Started_amines':[]}
	good_linkers = ['OCC#CCO*','OC/C=C\CO*','OCCCCO*','OCCCCCCO','OCC(CO*)(CO*)CO*','OCC(CC)(CO*)CO*','OCC(C)(CO*)CO*']
	if not fixed_number_to_screen is None:
		total_possible = len(amines)*len(good_linkers)*len(isos)*len(coohs)*len(coohs)
		fraction = float(fixed_number_to_screen)/float(total_possible)
		print(fraction)
	for a, ami in enumerate(amines):
		# print('on to amine #',a,': ',ami)
		amis['Started_amines'].append(ami)
		pd.DataFrame(amis).to_csv('Files_for_screen/Completed_simple_symmetric_amines.csv',index = False)
		for ol, ohlinker in enumerate(oh_linkers):
			if ohlinker in good_linkers:
				# print('on to oh linker #',ol,': ',ohlinker)
				for i, iso in enumerate(isos):
					# print('on to iso #',i,': ',iso)
					for c1, cooh1 in enumerate(coohs):
						for c2, cooh2 in enumerate(coohs):
							if random.random()<fraction:
								# print(ami,', ',str(a))
								# print(oh1,', ',str(o1))
								# print(oh2,', ',str(o2))
								# print(iso,', ',str(i))
								# print(cooh,', ',str(c))
								# print(cooh1)
								# print(cooh2)
								full_smiles = generate_4cr_ketone_smiles(ami, iso, ohlinker, ohlinker, cooh1, cooh_after_oh_1 = cooh2, cooh_after_oh_2 = cooh2)
								mol = Chem.MolFromSmiles(full_smiles)
								n_isomers = len(tuple(EnumerateStereoisomers(mol)))
								sp3_ns = 0
								for x in mol.GetAtoms():
									if str(x.GetSymbol())=='N' and str(x.GetHybridization()) == 'SP3':
										sp3_ns += 1
								if is_valid_structure(n_isomers, sp3_ns):
								# print(full_smiles)
									name, amine_name, iso_name, ketone_name, cooh_name = generate_4cr_ketone_smiles_name_and_components(a+1,i+1,ol+1,ol+1,c1+1, cooh_after_oh_1 = c2+1, cooh_after_oh_2 = c2+1)
									all_lipid_dict['Ketone'].append(ketone_name)
									all_lipid_dict['Isocyanide'].append(iso_name)
									all_lipid_dict['Amine'].append(amine_name)
									all_lipid_dict['Carboxylic_acid'].append(cooh_name)
									all_lipid_dict['smiles'].append(full_smiles)
									all_lipid_dict['Lipid_name'].append(name)
									all_lipid_dict['quantified_delivery'].append(0)
									
									all_lipid_dict['n_stereoisomers'].append(n_isomers)
									all_lipid_dict['sp3_nitrogens'].append(sp3_ns)
	to_return = pd.DataFrame(all_lipid_dict)
	return to_return

def generate_symmetric_cooh_linker_4cr_ketones(fraction = 2, fixed_number_to_screen = None):
	all_lipid_dict = {'Lipid_name':[],'Amine':[],'Ketone':[],'Isocyanide':[],'Carboxylic_acid':[],'smiles':[],'n_stereoisomers':[],'sp3_nitrogens':[],'quantified_delivery':[]}
	amis = {'Started_amines':[]}
	good_cooh_linkers = ['C(=O)CCCC(=O)^','C(=O)CCCCCCCCCC(=O)^','C(=O)C5(CCCCC5)C(=O)^','C(=O)/C=C\C=C/C(=O)^','C(=O)C/C=C/CC(=O)^','C(=O)C[C@@]12C[C@H]3C[C@@H](C1)C[C@@](C3)(CC(=O)^)C2','C(=O)/C=C\C(=O)^']
	if not fixed_number_to_screen is None:
		total_possible = len(amines)*len(good_cooh_linkers)*len(isos)*len(ohs)*len(ohs)
		fraction = float(fixed_number_to_screen)/float(total_possible)
	for a, ami in enumerate(amines):
		# print('on to amine #',a,': ',ami)
		amis['Started_amines'].append(ami)
		pd.DataFrame(amis).to_csv('Files_for_screen/Completed_simple_symmetric_amines.csv',index = False)
		for cl, coohlinker in enumerate(cooh_linkers):
			if coohlinker in good_cooh_linkers:
				for i, iso in enumerate(isos):
					for o1, oh1 in enumerate(ohs):
						for o2, oh2 in enumerate(ohs):
							if random.random()<fraction:
								# print(ami,', ',str(a))
								# print(oh1,', ',str(o1))
								# print(oh2,', ',str(o2))
								# print(iso,', ',str(i))
								# print(cooh,', ',str(c))
								# print(cooh1)
								# print(cooh2)
								full_smiles = generate_4cr_ketone_smiles(ami, iso, oh1, oh1, coohlinker, oh_after_cooh = oh2)
								mol = Chem.MolFromSmiles(full_smiles)
								n_isomers = len(tuple(EnumerateStereoisomers(mol)))
								sp3_ns = 0
								for x in mol.GetAtoms():
									if str(x.GetSymbol())=='N' and str(x.GetHybridization()) == 'SP3':
										sp3_ns += 1
								if is_valid_structure(n_isomers, sp3_ns):
								# print(full_smiles)
									name, amine_name, iso_name, ketone_name, cooh_name = generate_4cr_ketone_smiles_name_and_components(a+1,i+1,o1+1,o1+1,cl+1, oh_after_cooh = o2+1)
									all_lipid_dict['Ketone'].append(ketone_name)
									all_lipid_dict['Isocyanide'].append(iso_name)
									all_lipid_dict['Amine'].append(amine_name)
									all_lipid_dict['Carboxylic_acid'].append(cooh_name)
									all_lipid_dict['smiles'].append(full_smiles)
									all_lipid_dict['Lipid_name'].append(name)
									all_lipid_dict['quantified_delivery'].append(0)
									
									all_lipid_dict['n_stereoisomers'].append(n_isomers)
									all_lipid_dict['sp3_nitrogens'].append(sp3_ns)
	to_return = pd.DataFrame(all_lipid_dict)
	return to_return

def generate_all_the_symmetric_things(fraction = 2, fixed_number_to_screen = None):
	all_lipid_dict = {'Lipid_name':[],'Amine':[],'Ketone':[],'Isocyanide':[],'Carboxylic_acid':[],'smiles':[]}
	amis = {'Started_amines':[]}
	# good_oh_linkers = ['OCC#CCO*','OC/C=C\CO*','OCCCCO*','OCCCCCCO','OCC(CO*)(CO*)CO*','OCC(CC)(CO*)CO*','OCC(C)(CO*)CO*']
	# good_cooh_linkers = ['C(=O)CCCC(=O)^','C(=O)CCCCCCCCCC(=O)^','C(=O)C5(CCCCC5)C(=O)^','C(=O)/C=C\C=C/C(=O)^','C(=O)C/C=C/CC(=O)^','C(=O)C[C@@]12C[C@H]3C[C@@H](C1)C[C@@](C3)(CC(=O)^)C2','C(=O)/C=C\C(=O)^']
	if not fixed_number_to_screen is None:
		total_possible = len(amines)*len(isos)*len(oh_linkers)*len(cooh_linkers)*len(coohs)*len(ohs)
		fraction = float(fixed_number_to_screen)/float(total_possible)
	print('Keeping fraction: ',fraction)
	print('Approximate total number of lipids: ',total_possible)
	for a, ami in enumerate(amines):
		amis['Started_amines'].append(ami)
		pd.DataFrame(amis).to_csv('Files_for_screen/full_library_progress.csv',index = False)
		for i, iso in enumerate(isos):
			for o1, oh1 in enumerate(ohs):
				for c1, cooh1 in enumerate(coohs):
					if random.random()<fraction:
						full_smiles = generate_4cr_ketone_smiles(ami, iso, oh1, oh1, cooh1)
						name, amine_name, iso_name, ketone_name, cooh_name = generate_4cr_ketone_smiles_name_and_components(a+1,i+1,o1+1,o1+1,c1+1)
						all_lipid_dict['Ketone'].append(ketone_name)
						all_lipid_dict['Isocyanide'].append(iso_name)
						all_lipid_dict['Amine'].append(amine_name)
						all_lipid_dict['Carboxylic_acid'].append(cooh_name)
						all_lipid_dict['smiles'].append(full_smiles)
						all_lipid_dict['Lipid_name'].append(name)
				for cl, coohlinker in enumerate(cooh_linkers):
					for o3, oh3 in enumerate(ohs):
						if random.random()<fraction:
							full_smiles = generate_4cr_ketone_smiles(ami, iso, oh1, oh1, coohlinker, oh_after_cooh = oh3)
							name, amine_name, iso_name, ketone_name, cooh_name = generate_4cr_ketone_smiles_name_and_components(a+1,i+1,o1+1,o1+1,cl+1, oh_after_cooh = o3+1)
							all_lipid_dict['Ketone'].append(ketone_name)
							all_lipid_dict['Isocyanide'].append(iso_name)
							all_lipid_dict['Amine'].append(amine_name)
							all_lipid_dict['Carboxylic_acid'].append(cooh_name)
							all_lipid_dict['smiles'].append(full_smiles)
							all_lipid_dict['Lipid_name'].append(name)
					# Screen cooh linker-containing
			for ol1, ohlinker1 in enumerate(oh_linkers):
				for c1, cooh1 in enumerate(coohs):
					for c2, cooh2 in enumerate(coohs):
						if random.random()<fraction:
							full_smiles = generate_4cr_ketone_smiles(ami, iso, ohlinker1, ohlinker1, cooh1, cooh_after_oh_1 = cooh2, cooh_after_oh_2 = cooh2)
							name, amine_name, iso_name, ketone_name, cooh_name = generate_4cr_ketone_smiles_name_and_components(a+1,i+1,ol1+1,ol1+1,c1+1, cooh_after_oh_1 = c2+1, cooh_after_oh_2 = c2+1)
							all_lipid_dict['Ketone'].append(ketone_name)
							all_lipid_dict['Isocyanide'].append(iso_name)
							all_lipid_dict['Amine'].append(amine_name)
							all_lipid_dict['Carboxylic_acid'].append(cooh_name)
							all_lipid_dict['smiles'].append(full_smiles)
							all_lipid_dict['Lipid_name'].append(name)
					for cl, coohlinker in enumerate(cooh_linkers):
						for o1, oh1 in enumerate(ohs):
							if random.random()<fraction:
								full_smiles = generate_4cr_ketone_smiles(ami, iso, ohlinker1, ohlinker1, coohlinker, oh_after_cooh = oh1, cooh_after_oh_1 = cooh1, cooh_after_oh_2 = cooh1)
								name, amine_name, iso_name, ketone_name, cooh_name = generate_4cr_ketone_smiles_name_and_components(a+1,i+1,ol1+1,ol1+1,cl+1, oh_after_cooh = o1+1, cooh_after_oh_1 = c1+1, cooh_after_oh_2 = c1+1)
								all_lipid_dict['Ketone'].append(ketone_name)
								all_lipid_dict['Isocyanide'].append(iso_name)
								all_lipid_dict['Amine'].append(amine_name)
								all_lipid_dict['Carboxylic_acid'].append(cooh_name)
								all_lipid_dict['smiles'].append(full_smiles)
								all_lipid_dict['Lipid_name'].append(name)
	# for key in all_lipid_dict:
		# print(key,': length ',len(all_lipid_dict[key]))
	to_return = pd.DataFrame(all_lipid_dict)
	to_return.to_csv('Files_for_screen/unfiltered_all_lipids.csv', index = False)
	to_return['quantified_delivery'] = [0]*len(to_return)
	stereos = []
	for r, row in to_return.iterrows():
		mol = Chem.MolFromSmiles(row['smiles'])
		n_isomers = len(tuple(EnumerateStereoisomers(mol)))
		stereos.append(n_isomers)
	to_return['n_stereoisomers'] = stereos
	to_return = to_return[to_return['n_stereoisomers']<3]
	return to_return

def generate_all_things_subset(number_to_screen):
	all_lipid_dict = {'Lipid_name':[],'Amine':[],'Ketone':[],'Isocyanide':[],'Carboxylic_acid':[],'smiles':[], 'n_stereoisomers':[],'quantified_delivery':[], 'chiral_smiles':[]}
	# pick random amine and linker
	
	# pick random carboxyl: first, make a random choice
	while len(all_lipid_dict['smiles'])<number_to_screen:
		if len(all_lipid_dict['smiles'])%10000 == 5:
			print(len(all_lipid_dict['smiles']))
		a = random.choice(range(len(amines)))
		i = random.choice(range(len(isos)))
		amine = amines[a]
		iso = isos[i]
		use_dicarboxy = random.random() > float(len(coohs))/(len(cooh_linkers)*len(ohs))
		if use_dicarboxy:
			c = random.choice(range(len(cooh_linkers)))
			oha = random.choice(range(len(ohs)))
			cooh = cooh_linkers[c]
			ohafter = ohs[oha]
		else:
			c = random.choice(range(len(coohs)))
			cooh = coohs[c]
			ohafter = None
			oha = None

		use_dioh1 = random.random() > float(len(ohs))/(len(oh_linkers)*len(coohs))
		if use_dioh1:
			o1 = random.choice(range(len(oh_linkers)))
			ca1 = random.choice(range(len(coohs)))
			oh1 = oh_linkers[o1]
			cafter1 = coohs[ca1]
		else:
			o1 = random.choice(range(len(ohs)))
			oh1 = ohs[o1]
			ca1 = None
			cafter1 = None

		use_dioh2 = random.random() > float(len(ohs))/(len(oh_linkers)*len(coohs))
		if use_dioh2:
			o2 = random.choice(range(len(oh_linkers)))
			ca2 = random.choice(range(len(coohs)))
			oh2 = oh_linkers[o2]
			cafter2 = coohs[ca2]
		else:
			o2 = random.choice(range(len(ohs)))
			oh2 = ohs[o2]
			ca2 = None
			cafter2 = None
		full_smiles = generate_4cr_ketone_smiles(amine,iso,oh1,oh2,cooh,oh_after_cooh = ohafter, cooh_after_oh_1 = cafter1, cooh_after_oh_2 = cafter2, dechiralize = False)
		name, amine_name, iso_name, ketone_name, cooh_name = generate_4cr_ketone_smiles_name_and_components(a,i,o1,o2,c, oh_after_cooh = oha, cooh_after_oh_1 = ca1,cooh_after_oh_2 = ca2)
		mol = Chem.MolFromSmiles(full_smiles)
		n_isomers = len(tuple(EnumerateStereoisomers(mol)))
		if is_valid_structure(n_isomers):
			all_lipid_dict['Ketone'].append(ketone_name)
			all_lipid_dict['Isocyanide'].append(iso_name)
			all_lipid_dict['Amine'].append(amine_name)
			all_lipid_dict['Carboxylic_acid'].append(cooh_name)
			all_lipid_dict['chiral_smiles'].append(full_smiles)
			all_lipid_dict['smiles'].append(full_smiles.replace('@',''))
			all_lipid_dict['Lipid_name'].append(name)
			all_lipid_dict['quantified_delivery'].append(0)
			all_lipid_dict['n_stereoisomers'].append(n_isomers)
	to_return = pd.DataFrame(all_lipid_dict).drop_duplicates()
	to_return.to_csv('Files_for_screen/filtered_lipids_subset_'+str(number_to_screen)+'.csv', index = False)
	return to_return

def generate_all_aldehydes_subset(number_to_screen):
	all_lipid_dict = {'Lipid_name':[],'Amine':[],'Aldehyde':[],'Isocyanide':[],'Carboxylic_acid':[],'smiles':[], 'quantified_delivery':[], 'chiral_smiles':[]}
	# pick random amine and linker
	
	# pick random carboxyl: first, make a random choice
	for blah in range(number_to_screen):
		a = random.choice(range(len(amines)))
		i = random.choice(range(len(isos)))
		amine = amines[a]
		iso = isos[i]
		if len(all_lipid_dict['smiles'])%10000 == 5:
			print(len(all_lipid_dict['smiles']))
		use_dicarboxy = random.random() > float(len(coohs))/(len(cooh_linkers)*len(ohs))
		if use_dicarboxy:
			c = random.choice(range(len(cooh_linkers)))
			oha = random.choice(range(len(ohs)))
			cooh = cooh_linkers[c]
			ohafter = ohs[oha]
		else:
			c = random.choice(range(len(coohs)))
			cooh = coohs[c]
			ohafter = None
			oha = None

		use_dicarboxy = random.random() > float(len(coohs))/(len(cooh_linkers)*len(ohs))
		if use_dicarboxy:
			cal = random.choice(range(len(cooh_linkers)))
			ohaal = random.choice(range(len(ohs)))
			coohal = cooh_linkers[cal]
			ohafteral = ohs[ohaal]
		else:
			cal = random.choice(range(len(coohs)))
			coohal = coohs[cal]
			ohafteral = None
			ohaal = None

		full_smiles = generate_4cr_aldehyde_smiles(amine,iso,coohal,cooh,oh_after_cooh = ohafter, oh_after_coohal = ohafteral)
		name, amine_name, iso_name, aldehyde_name, cooh_name = generate_4cr_aldehyde_smiles_name_and_components(a,i,cal,c, oh_after_cooh = oha, oh_after_coohal = ohaal)
		all_lipid_dict['Aldehyde'].append(aldehyde_name)
		all_lipid_dict['Isocyanide'].append(iso_name)
		all_lipid_dict['Amine'].append(amine_name)
		all_lipid_dict['Carboxylic_acid'].append(cooh_name)
		all_lipid_dict['chiral_smiles'].append(full_smiles)
		all_lipid_dict['smiles'].append(full_smiles.replace('@',''))
		all_lipid_dict['Lipid_name'].append(name)
		all_lipid_dict['quantified_delivery'].append(0)
	to_return = pd.DataFrame(all_lipid_dict).drop_duplicates()
	to_return.to_csv('Files_for_screen/aldehyde_subset_'+str(number_to_screen)+'.csv', index = False)
	return to_return

def generate_all_the_things(fraction = 2, fixed_number_to_screen = None):
	all_lipid_dict = {'Lipid_name':[],'Amine':[],'Ketone':[],'Isocyanide':[],'Carboxylic_acid':[],'smiles':[]}
	amis = {'Started_amines':[]}
	# good_oh_linkers = ['OCC#CCO*','OC/C=C\CO*','OCCCCO*','OCCCCCCO','OCC(CO*)(CO*)CO*','OCC(CC)(CO*)CO*','OCC(C)(CO*)CO*']
	# good_cooh_linkers = ['C(=O)CCCC(=O)^','C(=O)CCCCCCCCCC(=O)^','C(=O)C5(CCCCC5)C(=O)^','C(=O)/C=C\C=C/C(=O)^','C(=O)C/C=C/CC(=O)^','C(=O)C[C@@]12C[C@H]3C[C@@H](C1)C[C@@](C3)(CC(=O)^)C2','C(=O)/C=C\C(=O)^']
	if not fixed_number_to_screen is None:
		total_possible = len(amines)*len(isos)*len(oh_linkers)*len(oh_linkers)/2*len(cooh_linkers)*len(coohs)*len(ohs)*len(coohs)
		fraction = float(fixed_number_to_screen)/float(total_possible)
	print('Keeping fraction: ',fraction)
	print('Approximate total number of lipids: ',total_possible)
	for a, ami in enumerate(amines):
		amis['Started_amines'].append(ami)
		pd.DataFrame(amis).to_csv('Files_for_screen/full_symmetric_library_progress.csv',index = False)
		for i, iso in enumerate(isos):
			for o1, oh1 in enumerate(ohs):
				for o2, oh2 in enumerate(ohs):
					if o2 >= o1:
						for c1, cooh1 in enumerate(coohs):
							if random.random()<fraction:
								full_smiles = generate_4cr_ketone_smiles(ami, iso, oh1, oh2, cooh1)
								name, amine_name, iso_name, ketone_name, cooh_name = generate_4cr_ketone_smiles_name_and_components(a+1,i+1,o1+1,o2+1,c1+1)
								all_lipid_dict['Ketone'].append(ketone_name)
								all_lipid_dict['Isocyanide'].append(iso_name)
								all_lipid_dict['Amine'].append(amine_name)
								all_lipid_dict['Carboxylic_acid'].append(cooh_name)
								all_lipid_dict['smiles'].append(full_smiles)
								all_lipid_dict['Lipid_name'].append(name)
						for cl, coohlinker in enumerate(cooh_linkers):
							for o3, oh3 in enumerate(ohs):
								if random.random()<fraction:
									full_smiles = generate_4cr_ketone_smiles(ami, iso, oh1, oh2, coohlinker, oh_after_cooh = oh3)
									name, amine_name, iso_name, ketone_name, cooh_name = generate_4cr_ketone_smiles_name_and_components(a+1,i+1,o1+1,o2+1,cl+1, oh_after_cooh = o3+1)
									all_lipid_dict['Ketone'].append(ketone_name)
									all_lipid_dict['Isocyanide'].append(iso_name)
									all_lipid_dict['Amine'].append(amine_name)
									all_lipid_dict['Carboxylic_acid'].append(cooh_name)
									all_lipid_dict['smiles'].append(full_smiles)
									all_lipid_dict['Lipid_name'].append(name)
							# Screen cooh linker-containing
				for ol, ohlinker in enumerate(oh_linkers):
					for c1, cooh1 in enumerate(coohs):
						for c2, cooh2 in enumerate(coohs):
							if random.random()<fraction:
								full_smiles = generate_4cr_ketone_smiles(ami, iso, oh1, ohlinker, cooh2, cooh_after_oh_2 = cooh1)
								name, amine_name, iso_name, ketone_name, cooh_name = generate_4cr_ketone_smiles_name_and_components(a+1,i+1,o1+1,ol+1,c2+1, cooh_after_oh_2 = c1+1)
								all_lipid_dict['Ketone'].append(ketone_name)
								all_lipid_dict['Isocyanide'].append(iso_name)
								all_lipid_dict['Amine'].append(amine_name)
								all_lipid_dict['Carboxylic_acid'].append(cooh_name)
								all_lipid_dict['smiles'].append(full_smiles)
								all_lipid_dict['Lipid_name'].append(name)
						for cl, coohlinker in enumerate(cooh_linkers):
							for o3, oh3 in enumerate(ohs):
								if random.random()<fraction:
									full_smiles = generate_4cr_ketone_smiles(ami, iso, oh1, ohlinker, coohlinker, cooh_after_oh_2 = cooh1, oh_after_cooh = oh3)
									name, amine_name, iso_name, ketone_name, cooh_name = generate_4cr_ketone_smiles_name_and_components(a+1,i+1,o1+1,ol+1,cl+1, oh_after_cooh = o3+1, cooh_after_oh_2 = c1+1)
									all_lipid_dict['Ketone'].append(ketone_name)
									all_lipid_dict['Isocyanide'].append(iso_name)
									all_lipid_dict['Amine'].append(amine_name)
									all_lipid_dict['Carboxylic_acid'].append(cooh_name)
									all_lipid_dict['smiles'].append(full_smiles)
									all_lipid_dict['Lipid_name'].append(name)
			for ol1, ohlinker1 in enumerate(oh_linkers):
				for ol2, ohlinker2 in enumerate(oh_linkers):
					if ol2 >= ol1:
						for c1, cooh1 in enumerate(coohs):
							for c2, cooh2 in enumerate(coohs):
								for c3, cooh3 in enumerate(coohs):
									if random.random()<fraction:
										full_smiles = generate_4cr_ketone_smiles(ami, iso, ohlinker1, ohlinker2, cooh1, cooh_after_oh_1 = cooh2, cooh_after_oh_2 = cooh3)
										name, amine_name, iso_name, ketone_name, cooh_name = generate_4cr_ketone_smiles_name_and_components(a+1,i+1,ol1+1,ol2+1,c1+1, cooh_after_oh_1 = c2+1, cooh_after_oh_2 = c3+1)
										all_lipid_dict['Ketone'].append(ketone_name)
										all_lipid_dict['Isocyanide'].append(iso_name)
										all_lipid_dict['Amine'].append(amine_name)
										all_lipid_dict['Carboxylic_acid'].append(cooh_name)
										all_lipid_dict['smiles'].append(full_smiles)
										all_lipid_dict['Lipid_name'].append(name)
								for cl, coohlinker in enumerate(cooh_linkers):
									for o1, oh1 in enumerate(ohs):
										if random.random()<fraction:
											full_smiles = generate_4cr_ketone_smiles(ami, iso, ohlinker1, ohlinker2, coohlinker, oh_after_cooh = oh1, cooh_after_oh_1 = cooh1, cooh_after_oh_2 = cooh2)
											name, amine_name, iso_name, ketone_name, cooh_name = generate_4cr_ketone_smiles_name_and_components(a+1,i+1,ol1+1,ol2+1,cl+1, oh_after_cooh = o1+1, cooh_after_oh_1 = c1+1, cooh_after_oh_2 = c2+1)
											all_lipid_dict['Ketone'].append(ketone_name)
											all_lipid_dict['Isocyanide'].append(iso_name)
											all_lipid_dict['Amine'].append(amine_name)
											all_lipid_dict['Carboxylic_acid'].append(cooh_name)
											all_lipid_dict['smiles'].append(full_smiles)
											all_lipid_dict['Lipid_name'].append(name)
	# for key in all_lipid_dict:
		# print(key,': length ',len(all_lipid_dict[key]))
	to_return = pd.DataFrame(all_lipid_dict)
	to_return.to_csv('Files_for_screen/unfiltered_all_lipids.csv', index = False)
	to_return['quantified_delivery'] = [0]*len(to_return)
	stereos = []
	for r, row in to_return.iterrows():
		mol = Chem.MolFromSmiles(row['smiles'])
		n_isomers = len(tuple(EnumerateStereoisomers(mol)))
		stereos.append(n_isomers)
	to_return['n_stereoisomers'] = stereos
	to_return = to_return[to_return['n_stereoisomers']<3]
	return to_return

def generate_symmetric_all_both_linkers_4cr_ketones(fraction = 2, fixed_number_to_screen = None):
	all_lipid_dict = {'Lipid_name':[],'Amine':[],'Ketone':[],'Isocyanide':[],'Carboxylic_acid':[],'smiles':[],'n_stereoisomers':[],'sp3_nitrogens':[],'quantified_delivery':[]}
	amis = {'Started_amines':[]}
	# good_oh_linkers = ['OCC#CCO*','OC/C=C\CO*','OCCCCO*','OCCCCCCO','OCC(CO*)(CO*)CO*','OCC(CC)(CO*)CO*','OCC(C)(CO*)CO*']
	# good_cooh_linkers = ['C(=O)CCCC(=O)^','C(=O)CCCCCCCCCC(=O)^','C(=O)C5(CCCCC5)C(=O)^','C(=O)/C=C\C=C/C(=O)^','C(=O)C/C=C/CC(=O)^','C(=O)C[C@@]12C[C@H]3C[C@@H](C1)C[C@@](C3)(CC(=O)^)C2','C(=O)/C=C\C(=O)^']
	if not fixed_number_to_screen is None:
		total_possible = len(amines)*len(good_oh_linkers)*len(good_cooh_linkers)*len(isos)*len(coohs)*len(coohs)
		fraction = float(fixed_number_to_screen)/float(total_possible)
	print('Keeping fraction: ',fraction)
	for a, ami in enumerate(amines):
		print('on to amine #',a,': ',ami)
		amis['Started_amines'].append(ami)
		pd.DataFrame(amis).to_csv('Files_for_screen/Completed_simple_symmetric_amines.csv',index = False)
		for cl, coohlinker in enumerate(cooh_linkers):
			# if coohlinker in good_cooh_linkers:
			for ol, ohlinker in enumerate(oh_linkers):
				# if ohlinker in good_oh_linkers:
				for i, iso in enumerate(isos):
					for o1, oh1 in enumerate(ohs):
						for c1, cooh1 in enumerate(coohs):
							if random.random()<fraction:
								full_smiles = generate_4cr_ketone_smiles(ami, iso, ohlinker, ohlinker, coohlinker, oh_after_cooh = oh1, cooh_after_oh_1 = cooh1, cooh_after_oh_2 = cooh1)
								# print('COOH: ',cooh1)
								# print('Full lipid: ',full_smiles)
								mol = Chem.MolFromSmiles(full_smiles)
								n_isomers = len(tuple(EnumerateStereoisomers(mol)))
								sp3_ns = 0
								for x in mol.GetAtoms():
									if str(x.GetSymbol())=='N' and str(x.GetHybridization()) == 'SP3':
										sp3_ns += 1
								if is_valid_structure(n_isomers, sp3_ns):
								# print(full_smiles)
									name, amine_name, iso_name, ketone_name, cooh_name = generate_4cr_ketone_smiles_name_and_components(a+1,i+1,ol+1,ol+1,cl+1, oh_after_cooh = o1+1, cooh_after_oh_1 = c1+1, cooh_after_oh_2 = c1+1)
									all_lipid_dict['Ketone'].append(ketone_name)
									all_lipid_dict['Isocyanide'].append(iso_name)
									all_lipid_dict['Amine'].append(amine_name)
									all_lipid_dict['Carboxylic_acid'].append(cooh_name)
									all_lipid_dict['smiles'].append(full_smiles)
									all_lipid_dict['Lipid_name'].append(name)
									all_lipid_dict['quantified_delivery'].append(0)
									
									all_lipid_dict['n_stereoisomers'].append(n_isomers)
									all_lipid_dict['sp3_nitrogens'].append(sp3_ns)
	to_return = pd.DataFrame(all_lipid_dict)
	return to_return

def generate_symmetric_both_linkers_4cr_ketones(fraction = 2, fixed_number_to_screen = None):
	all_lipid_dict = {'Lipid_name':[],'Amine':[],'Ketone':[],'Isocyanide':[],'Carboxylic_acid':[],'smiles':[],'n_stereoisomers':[],'sp3_nitrogens':[],'quantified_delivery':[]}
	amis = {'Started_amines':[]}
	good_oh_linkers = ['OCC#CCO*','OC/C=C\CO*','OCCCCO*','OCCCCCCO','OCC(CO*)(CO*)CO*','OCC(CC)(CO*)CO*','OCC(C)(CO*)CO*']
	good_cooh_linkers = ['C(=O)CCCC(=O)^','C(=O)CCCCCCCCCC(=O)^','C(=O)C5(CCCCC5)C(=O)^','C(=O)/C=C\C=C/C(=O)^','C(=O)C/C=C/CC(=O)^','C(=O)C[C@@]12C[C@H]3C[C@@H](C1)C[C@@](C3)(CC(=O)^)C2','C(=O)/C=C\C(=O)^']
	if not fixed_number_to_screen is None:
		total_possible = len(amines)*len(good_oh_linkers)*len(good_cooh_linkers)*len(isos)*len(coohs)*len(coohs)
		fraction = float(fixed_number_to_screen)/float(total_possible)
	print('Keeping fraction: ',fraction)
	for a, ami in enumerate(amines):
		# print('on to amine #',a,': ',ami)
		amis['Started_amines'].append(ami)
		pd.DataFrame(amis).to_csv('Files_for_screen/Completed_simple_symmetric_amines.csv',index = False)
		for cl, coohlinker in enumerate(cooh_linkers):
			if coohlinker in good_cooh_linkers:
				for ol, ohlinker in enumerate(oh_linkers):
					if ohlinker in good_oh_linkers:
						for i, iso in enumerate(isos):
							for o1, oh1 in enumerate(ohs):
								for c1, cooh1 in enumerate(coohs):
									if random.random()<fraction:
										# print(ami,', ',str(a))
										# print(oh1,', ',str(o1))
										# print(oh2,', ',str(o2))
										# print(iso,', ',str(i))
										# print(cooh,', ',str(c))
										# print(cooh1)
										# print(cooh2)
										full_smiles = generate_4cr_ketone_smiles(ami, iso, ohlinker, ohlinker, coohlinker, oh_after_cooh = oh1, cooh_after_oh_1 = cooh1, cooh_after_oh_2 = cooh1)
										mol = Chem.MolFromSmiles(full_smiles)
										n_isomers = len(tuple(EnumerateStereoisomers(mol)))
										sp3_ns = 0
										for x in mol.GetAtoms():
											if str(x.GetSymbol())=='N' and str(x.GetHybridization()) == 'SP3':
												sp3_ns += 1
										if is_valid_structure(n_isomers, sp3_ns):
										# print(full_smiles)
											name, amine_name, iso_name, ketone_name, cooh_name = generate_4cr_ketone_smiles_name_and_components(a+1,i+1,ol+1,ol+1,cl+1, oh_after_cooh = o1+1, cooh_after_oh_1 = c1+1, cooh_after_oh_2 = c1+1)
											all_lipid_dict['Ketone'].append(ketone_name)
											all_lipid_dict['Isocyanide'].append(iso_name)
											all_lipid_dict['Amine'].append(amine_name)
											all_lipid_dict['Carboxylic_acid'].append(cooh_name)
											all_lipid_dict['smiles'].append(full_smiles)
											all_lipid_dict['Lipid_name'].append(name)
											all_lipid_dict['quantified_delivery'].append(0)
											
											all_lipid_dict['n_stereoisomers'].append(n_isomers)
											all_lipid_dict['sp3_nitrogens'].append(sp3_ns)
	to_return = pd.DataFrame(all_lipid_dict)
	return to_return


def generate_simple_symmetric_4cr_ketones(fraction = 2, fixed_number_to_screen = None):
	all_lipid_dict = {'Lipid_name':[],'Amine':[],'Ketone':[],'Isocyanide':[],'Carboxylic_acid':[],'smiles':[],'quantified_delivery':[]}
	amis = {'Started_amines':[]}
	if not fixed_number_to_screen is None:
		total_possible = len(amines)*len(ohs)*len(isos)*len(coohs)
		fraction = float(fixed_number_to_screen)/float(total_possible)
	for a, ami in enumerate(amines):
		# print('on to amine #',a,': ',ami)
		amis['Started_amines'].append(ami)
		pd.DataFrame(amis).to_csv('Files_for_screen/Completed_simple_symmetric_amines.csv',index = False)
		for o1, oh1 in enumerate(ohs):
			# print('on to oh #',o1,': ',oh1)
			for i, iso in enumerate(isos):
				for c, cooh in enumerate(coohs):
					if random.random()<fraction:
						try:
							full_smiles = generate_4cr_ketone_smiles(ami, iso, oh1, oh1, cooh)
						except:
							print(o1,', ',oh1)
							print(c,', ',cooh)
							print(i,', ',iso)
							print(a,', ',ami)
						name, amine_name, iso_name, ketone_name, cooh_name = generate_4cr_ketone_smiles_name_and_components(a+1,i+1,o1+1,o1+1,c+1)
						all_lipid_dict['Ketone'].append(ketone_name)
						all_lipid_dict['Isocyanide'].append(iso_name)
						all_lipid_dict['Amine'].append(amine_name)
						all_lipid_dict['Carboxylic_acid'].append(cooh_name)
						all_lipid_dict['smiles'].append(full_smiles)
						all_lipid_dict['Lipid_name'].append(name)
						all_lipid_dict['quantified_delivery'].append(0)
	to_return = pd.DataFrame(all_lipid_dict)
	return to_return

def reduce_folders(fixed_number_to_screen, new_folder_prefix, folders_to_reduce, files_to_reduce = ['test_extra_x.csv','test_metadata.csv','test_weights.csv','test.csv']):
	length_tester = pd.read_csv('Files_for_screen/'+folders_to_reduce[0]+'/test_weights.csv')
	length = len(length_tester)
	print('fixed number to screen: ',fixed_number_to_screen)
	mask = set(random.sample(range(length),int(fixed_number_to_screen)))
	for i, folder in enumerate(folders_to_reduce):
		path_if_none('Files_for_screen/'+new_folder_prefix+folder)
		for fname in files_to_reduce:
			df = pd.read_csv('Files_for_screen/'+folder + '/' + fname)
			df = df[[i in mask for i in range(length)]]
			df.to_csv('Files_for_screen/'+new_folder_prefix+folder+'/'+fname, index = False)


def generate_screening_files(mol_dict,extra_x_df,fpaths):
	for i,row in extra_x_df.iterrows():
		path_if_none('Files_for_screen/'+fpaths[i])
		pd.DataFrame({'SampleWeight':[1 for _ in range(len(mol_dict))]}).to_csv('Files_for_screen/'+fpaths[i]+'/test_weights.csv', index = False)
		mol_dict.to_csv('Files_for_screen/'+fpaths[i]+'/test_metadata.csv',index = False)
		mol_dict[['smiles','quantified_delivery']].to_csv('Files_for_screen/'+fpaths[i]+'/test.csv',index = False)
		rowdf = pd.DataFrame({})
		for col in extra_x_df.columns:
			rowdf[col] = [row[col]]*len(mol_dict)
		# for j in range(len(mol_dict)):
		# 	rowdf = rowdf.append(row)
		rowdf.to_csv('Files_for_screen/'+fpaths[i]+'/test_extra_x.csv',index = False)

def path_if_none(newpath):
	if not os.path.exists(newpath):
		os.makedirs(newpath)


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

def main(argv):
	# args = sys.argv[1:]
	library_type = argv[1]
	library_generation_complete = False
	subset = False
	fraction = 2
	fixed_number_to_screen = None
	if len(argv)>2:
		extra_param = argv[2]
		if type(extra_param)==type(True):
			library_generation_complete = argv[2]
		elif float(extra_param)<1:
			fraction = extra_param
			subset = True
		elif float(extra_param)>10:
			fixed_number_to_screen = extra_param
			subset = True
	if library_type == 'simple_4cr_ketones':
		if not library_generation_complete:
			library_df = generate_simple_4cr_ketones()
		library_df_name = 'Simple_4cr_ketones.csv'
		library_head = 'sim_4cr_k_'

	elif library_type == 'simple_symmetric_4cr_ketones':
		if not library_generation_complete:
			library_df = generate_simple_symmetric_4cr_ketones(fraction = fraction, fixed_number_to_screen = fixed_number_to_screen)
			if not fixed_number_to_screen is None:
				actual_length = len(library_df)
				print('targeted: ',fixed_number_to_screen,' but actual made: ',actual_length)
				library_df = generate_simple_symmetric_4cr_ketones(fraction = fraction, fixed_number_to_screen = float(fixed_number_to_screen)*float(fixed_number_to_screen)/actual_length)
				print('now targeted: ',fixed_number_to_screen,' but actually made: ',len(library_df))
		library_df_name = 'Simple_symmetric_4cr_ketones_followup.csv'
		library_head = 'follow_sim_sym_4cr_k_'

	elif library_type == 'symmetric_oh_linker_4cr_ketones':
		if not library_generation_complete:
			library_df = generate_symmetric_oh_linker_4cr_ketones(fraction = fraction, fixed_number_to_screen = fixed_number_to_screen)
			if not fixed_number_to_screen is None:
				actual_length = len(library_df)
				print('targeted: ',fixed_number_to_screen,' but actual made: ',actual_length)
				library_df = generate_symmetric_oh_linker_4cr_ketones(fraction = fraction, fixed_number_to_screen = float(fixed_number_to_screen)*float(fixed_number_to_screen)/actual_length)
				print('now targeted: ',fixed_number_to_screen,' but actually made: ',len(library_df))
		library_df_name = 'Symmetric_oh_linkers.csv'
		library_head = 'sym_oh_linker_4cr_k_'

	elif library_type == 'all_things_subset':
		library_df = generate_all_things_subset(int(fixed_number_to_screen))
		library_df_name = 'all_things_subset_'+str(fixed_number_to_screen)+'.csv'
		library_head = 'all_4cr_k_sub_'+str(fixed_number_to_screen)+'_'

	elif library_type == 'aldehydes_subset':
		library_df = generate_all_aldehydes_subset(int(fixed_number_to_screen))
		library_df_name = 'aldehydes_subset_'+str(fixed_number_to_screen)+'.csv'
		library_head = 'all_4cr_a_sub_'+str(fixed_number_to_screen)+'_'

	elif library_type == 'all_the_things':
		library_df = generate_all_the_things(fraction = fraction, fixed_number_to_screen = fixed_number_to_screen)
		library_df_name = 'all_the_things.csv'
		library_head = 'all_4cr_k_'

	elif library_type == 'all_the_symmetric_things':
		library_df = generate_all_the_symmetric_things(fraction = fraction, fixed_number_to_screen = fixed_number_to_screen)
		library_df_name = 'all_the_symmetric_things.csv'
		library_head = 'all_symmetric_4cr_k_'

	elif library_type == 'symmetric_cooh_linker_4cr_ketones':
		if not library_generation_complete:
			library_df = generate_symmetric_cooh_linker_4cr_ketones(fraction = fraction, fixed_number_to_screen = fixed_number_to_screen)
			if not fixed_number_to_screen is None:
				actual_length = len(library_df)
				print('targeted: ',fixed_number_to_screen,' but actual made: ',actual_length)
				library_df = generate_symmetric_cooh_linker_4cr_ketones(fraction = fraction, fixed_number_to_screen = float(fixed_number_to_screen)*float(fixed_number_to_screen)/actual_length)
				print('now targeted: ',fixed_number_to_screen,' but actually made: ',len(library_df))
		library_df_name = 'Symmetric_cooh_linkers.csv'
		library_head = 'sym_oh_linker_4cr_k_'

	elif library_type == 'symmetric_both_linkers_4cr_ketones':
		if not library_generation_complete:
			library_df = generate_symmetric_both_linkers_4cr_ketones(fraction = fraction, fixed_number_to_screen = fixed_number_to_screen)
			if not fixed_number_to_screen is None:
				actual_length = len(library_df)
				print('targeted: ',fixed_number_to_screen,' but actual made: ',actual_length)
				library_df = generate_symmetric_both_linkers_4cr_ketones(fraction = fraction, fixed_number_to_screen = float(fixed_number_to_screen)*float(fixed_number_to_screen)/actual_length)
				print('now targeted: ',fixed_number_to_screen,' but actually made: ',len(library_df))
		library_df_name = 'Symmetric_both_linkers.csv'
		library_head = 'sym_both_linkers_4cr_k_'

	elif library_type == 'reduce_library':
		new_folder_prefix = argv[3]
		folders_to_reduce = argv[4:]
		reduce_folders(fixed_number_to_screen, new_folder_prefix, folders_to_reduce)

	if not library_type == 'reduce_library':
		if subset:
			library_df_name = library_df_name[:-4]+'_subset.csv'
			library_head = 'subset_'+library_head
		if library_generation_complete:
			library_df = pd.read_csv(library_df_name)
		else:
			# print(library_df.head())
			library_df.to_csv(library_df_name, index = False)
		condition_names = list(pd.read_csv('Extra_x_files/condition_names.csv')['Condition_name'])
		fpaths = [library_head + con for con in condition_names]
		extra_x_df = pd.read_csv('Extra_x_files/all_extra_xs.csv')
		generate_screening_files(library_df,extra_x_df,fpaths)		

if __name__ == '__main__':
	main(sys.argv)


# generate_structure_plus_activity_file()
# generate_data_files()





