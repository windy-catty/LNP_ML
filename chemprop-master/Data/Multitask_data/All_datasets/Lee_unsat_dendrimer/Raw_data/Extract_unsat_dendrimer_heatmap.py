from PIL import Image
import numpy as np
import pandas as pd

header = 'unsat_dendrimer'
image = Image.open(header + '_heatmap.png')
pix = image.load()
# print(image.size)
# print(pix[64,64])

# key_x = 840
# key_top = 118
# key_bot = 603
# value_top = 50400000.0
# value_bot = 1

# color_to_lum_dict = {}
# for y in range(key_top, key_bot):
# 	scale = 1.0 - (y-key_top)/(key_bot-key_top)
# 	# print(scale)
# 	lum_value = value_bot + (value_top-value_bot)*scale
# 	color_to_lum_dict[pix[key_x,y]] = lum_value
# 	# Set white as -100
# 	color_to_lum_dict[255,255,255,255] = -100
	# print(lum_value)

color_to_lum_dict = {}
color_to_lum_dict[255,255,255,255] = 0
color_to_lum_dict[245,194,137,255] = 1
color_to_lum_dict[241,163,84,255] = 2
color_to_lum_dict[237,107,45,255] = 3
color_to_lum_dict[234,50,35,255] = 4



# print(color_to_lum_dict[(251,5,10,255)])
# hmp_colors = [pix[key_x,y] for y in range(key_top, key_bot)]

def value_from_color(color, ctol_dict):
	min_dist = -1
	min_key = None
	for k in ctol_dict.keys():
		dist = np.linalg.norm(np.array(color[:3]) - np.array(k[:3]))
		if min_dist < -0.5 or dist<min_dist:
			min_key = k
			min_dist = dist
	return ctol_dict[min_key]



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
					pix[color_x+i, color_y+j] = (0,0,0,255)
			to_return[down][right] = value_from_color(color, color_to_lum_dict)
	return to_return

# submatrix_length = 828-88
# submatrix_heights = [281-128, 468-286, 603-474, 880-621, 1120-886, 1281-1127, 1314-1288]
# vert_origins = [128,286,474,621,886,1127,1288]
# values_high = [6,7,5,10,9,6,1]
x_start = 370
x_end = 900
y_starts = [75]
y_ends = [918]
values_high = [13]


all_vals = []
for i in range(len(values_high)):
	all_vals = all_vals + get_matrix_data([x_start,y_starts[i]], x_end-x_start, y_ends[i]-y_starts[i], 7,values_high[i])

# vals = get_matrix_data([51,51],submatrix_length, submatrix_height,12,4)
image.save(header + '_heatmap_with_dots.png',format = 'png')
# print(len(all_vals))
# print(len(all_vals[0]))
# print(all_vals)
# print(vals)
df = pd.DataFrame(all_vals)
df.to_csv(header + '_lum_results.csv')

