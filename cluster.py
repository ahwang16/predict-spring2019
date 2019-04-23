# cluster.py

# a file of helper files ot parse and cluster idioms in the SLIDE lexicon

from collections import defaultdict
import spacy, json
import pickle as pkl
# from nltk.corpus import stopwords

# stopWords = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

# a class of connected nodes that represents lexical overlap in corpus
class Graph():
	def __init__(self):
		self.nodes = [Node(None, None, None)] # list of all nodes in the graph
		self.heads = set() # set of indices of head nodes (first word of each idiom) in self.nodes
		self.indices = {"":0} # word -> index in self.nodes
		self.index = 1 # current index for use with self.indices/self.nodes


	def add(self, token, terminal, prev="", idiom=None, isHead=False):
		p = self.indices[prev]
		if token not in self.indices:
			self.indices[token] = self.index # create new mapping with current index
			self.nodes.append(Node(idiom, token, terminal)) # add new token node to list of nodes
			self.nodes[p].addnext(self.index) # prev token node points at current
			self.index += 1

			if isHead:
				self.heads.add(self.index-1)

		else:
			current = self.indices[token]
			self.nodes[p].addnext(current)

			if isHead:
				self.heads.add(current)

			if terminal:
				self.nodes[current].addidiom(idiom)


	def load(self, IDIOMS):
		with open(IDIOMS, 'r') as data:
			idioms = json.load(data)

		for idiom in idioms:
			i = nlp(idiom)
			if len(i) == 0:
				continue

			self.add(i[0].text.strip(), False, idiom={idiom}, isHead=True) # first word is the head node
			prev = i[0].text.strip()


			if len(i) == 1:
				self.nodes[self.index-1].terminal = True
				continue

			for x in range(1, len(i) - 1):
				self.add(i[x].text.strip(), False, prev, idiom={idiom})
				prev = i[x].text.strip()

			self.add(i[-1].text.strip(), True, prev, idiom={idiom})


	# DFS implementation to cluster idioms on lexical overlap
	def cluster(self):
		clusters = []
		#print(self.heads)
		for head in self.heads:
			stack = [head] # frontier implemented as stack
			explored = set()
			frontier = set()
			frontier.add(head) # for searching through frontier
			cluster = set()

			while len(stack):
				#print(stack)
				node = stack.pop()
				frontier.remove(node)
				explored.add(node)
				
				if self.nodes[node].terminal:
					cluster = cluster | self.nodes[node].idiom

				for n in self.nodes[node].nextnodes:
					if n not in frontier and n not in explored:
						stack.append(n)
						frontier.add(n)

			clusters.append(cluster)

		return clusters


class Node():
	def __init__(self, idiom, token, terminal):
		self.nextnodes = [] # indices of nodes that immediately follow self
		self.idiom = idiom # source of token this node represents
		self.token = token
		self.terminal = terminal


	def __str__(self):
		return "{} ({})".format(self.token, self.idiom)


	def addnext(self, prev):
		self.nextnodes.append(prev)


	def addidiom(self, idiom):
		self.idiom |= idiom


# returns dictionary of lemmatized words and frequencies throughout entire lexicon
def count(IDIOMS):
	words = defaultdict(int)

	with open(IDIOMS, 'r') as idioms:
		data = json.load(idioms)

	for idiom in data:
		doc = nlp(idiom)

		for word in doc:
			words[word.lemma_] += 1

	#with open('counts.pkl', 'wb') as n:
	#	pkl.dump(words, n)

	with open('counts.json', 'w') as n:
		json.dump(words, n)

	return words


# store idioms in memory with option to remove stop words
def parse(stop=False):
	with open("./IBM_Debater_(R)_SLIDE_LREC_2018/idiomLexicon.tsv", "r") as infile:
		next(infile)
		idioms = []
		for line in infile:
			l = line.split('\t')
			if l[11] != "X":
				sent = nlp(l[0])

				newsent = ""
				#print(l[0])
				for token in sent:
					if stop:
						if not token.is_stop:
							newsent += token.text + token.whitespace_
					else:
						newsent += token.text + token.whitespace_

				idioms.append(newsent)

	with open('idioms.json', 'w') as n:
		json.dump(idioms, n)

	return idioms


# https://spacy.io/api/annotation#pos-tagging
def clusterbypos(IDIOMS):
	with open(IDIOMS, 'r') as idioms:
		data = json.load(idioms)

	cluster = defaultdict(set)

	pos = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN',
		   'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X',
		   'SPACE']

	for p in pos:
		cluster[p] = set()

	for idiom in data:
		doc = nlp(idiom)
		for word in doc:
			cluster[word.pos_].add(idiom)

	with open('pos.pkl', 'wb') as n:
		pkl.dump(cluster, n)

	return cluster


# https://spacy.io/api/annotation#named-entities
def clusterbyner(IDIOMS):
	with open(IDIOMS, 'r') as idioms:
		data = json.load(idioms)

	cluster = defaultdict(set) # default value is empty set

	# list of all named entities
	ner = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART",
			"LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"] 
	
	# initialize defaultdict so that all named entities are accounted for even if not seen
	for n in ner:
		cluster[n] = set()

	for idiom in data:
		doc = nlp(idiom)
		for word in doc:
			cluster[word.ent_type_].add(idiom)


	with open('ner.pkl', 'wb') as n:
		pkl.dump(cluster, n)

	return cluster





if __name__ == "__main__":
	IDIOMS = 'idioms.json'

#	print("starting parse")
#	print(parse())
	g = Graph()
	print("starting load")
	g.load('idioms.json')
	print("starting cluster")
	c = g.cluster()
#	print(c)

	with open('clusters.pkl', 'wb') as cfile:
		pkl.dump(c, cfile)
#
#	print(g.indices)


#	print('starting ner')
#	print(clusterbyner(IDIOMS))

	#print('starting parse')
	#IDIOMS = parse()

#	print('starting pos')
#	print(clusterbypos(IDIOMS))

#	print('starting count')
#	print(count(IDIOMS))

