from PIL import Image
import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors, MolFromSmiles


library_type = 'Li_thiolyne'
image = Image.open(library_type + '_heatmap_grayscale.png')
pix = image.load()
# print(image.size)
# print(pix[64,64])

key_x = 1395
key_top = 90
key_bot = 837
value_top = 13
value_bot = 0.0

color_to_lum_dict = {}
for y in range(key_top, key_bot):
	scale = 1.0 - (y-key_top)/(key_bot-key_top)
	# print(scale)
	lum_value = value_bot + (value_top-value_bot)*scale
	color_to_lum_dict[pix[key_x,y]] = lum_value
	# print(lum_value)


# print(color_to_lum_dict[(251,5,10,255)])
hmp_colors = [pix[key_x,y] for y in range(key_top, key_bot)]

def value_from_color(color, ctol_dict):
	min_dist = -1
	min_key = None
	for k in ctol_dict.keys():
		dist = np.linalg.norm(np.array(color) - np.array(k))
		if min_dist < -0.5 or dist<min_dist:
			min_key = k
			min_dist = dist
	return ctol_dict[min_key]

def get_molecular_weights():
	df = pd.read_csv('Li_thiolyne_structures_with_activities.csv')
	mol_weights = []
	for smiles in df.smiles:
		m = MolFromSmiles(smiles)
		mol_weights.append(Descriptors.MolWt(m))
	df['MolWt'] = mol_weights
	df.to_csv('Li_thiolyne_structures_with_activities.csv', index = False)


def get_matrix_data(start_pixel, length, height, values_wide, values_high):
	to_return = [[0 for v in range(values_wide)] for v in range(values_high)]
	end_pixel = [start_pixel[0]+length, start_pixel[1]+height]
	# print(end_pixel)
	width_per_point = float(end_pixel[0]-start_pixel[0])/values_wide
	height_per_point = float(end_pixel[1] - start_pixel[1])/values_high
	# print(width_per_point)
	# print(height_per_point)
	for down in range(values_high):
		for right in range(values_wide):
			color_x = start_pixel[0]+width_per_point*(right+0.5)
			color_y = start_pixel[1]+height_per_point*(down+0.5)
			# print(color_x,', ', color_y)
			color = pix[color_x, color_y]
			for i in range(-2,2):
				for j in range(-2,2):
					pix[color_x+i, color_y+j] = (0)
			to_return[down][right] = (value_from_color(color, color_to_lum_dict))
	return to_return

submatrix_length = 1162-158
submatrix_height = 867-60
submatrix_intervals = 127.5-28


all_vals = []
for i in range(1):
	vals = get_matrix_data([158,60+i*submatrix_intervals], submatrix_length, submatrix_height, 8,14)
	all_vals = all_vals + vals


# all_vals = get_matrix_data([226,28], submatrix_length, submatrix_height, 25,8)

# vals = get_matrix_data([51,51],submatrix_length, submatrix_height,12,4)
# image.save(library_type + '_heatmap_grayscale_with_dots.png',format = 'png')
# print(len(all_vals))
# print(len(all_vals[0]))
# print(all_vals)
# print(vals)
# df = pd.DataFrame(all_vals)
# df.to_csv(library_type + '_delivery_values.csv')

get_molecular_weights()

