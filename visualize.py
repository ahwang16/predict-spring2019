# visualize.py

# Some methods to visualize SLIDE data

import slide
import pickle as pkl
import json

def displaydict(infile, index):
	name, ext = infile.split('.')

	idioms = None

	# infile is dict --> str(category) : set(str(idiom))
	if ext == 'pkl':
		with open(infile, 'rb') as i:
			idioms = pkl.load(i)
	else:
		with open(infile, 'r') as i:
			idioms.json.load(i)

	count = 1
	for category in idioms:
		print(category)
		slide.displaysubset(index, idioms[category],
				    '{}_{}_{}.png'.format(name, index, category))
		count += 1


if __name__ == '__main__':
	files = ['ner.pkl', 'pos.pkl']

	for x in range(1, 4):
		for f in files:
			print(f, x)
			displaydict(f, x)

	
