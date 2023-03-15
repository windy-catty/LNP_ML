from PIL import Image
import numpy as np
import pandas as pd

image = Image.open('384_screen_image.png')
pix = image.load()
# print(image.size)
# print(pix[64,64])

key_x = 411
key_top = 574
key_bot = 935
value_top = 1000.0
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

submatrix_length = 370-51
submatrix_height = 157-51

submatrix_intervals = 168-51

all_vals = []
for i in range(8):
	vals = get_matrix_data([51,51+i*submatrix_intervals], submatrix_length, submatrix_height, 12,4)
	all_vals = all_vals + vals

# vals = get_matrix_data([51,51],submatrix_length, submatrix_height,12,4)
image.save('384_screen_image_with_dots.png',format = 'png')
# print(len(all_vals))
# print(len(all_vals[0]))
# print(all_vals)
# print(vals)
df = pd.DataFrame(all_vals)
df.to_csv('lum_values.csv')

