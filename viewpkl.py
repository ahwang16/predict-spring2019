import pickle as pkl

with open('idiomsfiltered.json', 'r') as w:
	x = pkl.load(w)

print(x)
