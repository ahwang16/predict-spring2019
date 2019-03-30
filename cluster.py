# cluster.py

# a file of helper files ot parse and cluster idioms in the SLIDE lexicon

from collections import defaultdict
import spacy
import json
# from nltk.corpus import stopwords

# stopWords = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

# a class of connected nodes that represents lexical overlap in corpus
class Graph():
	def __init__(self):
		self.nodes = [Node(None)] # list of all nodes in the graph
		self.heads = [] # list of indices of head nodes (first word of each idiom) in self.nodes
		self.indices = {"":0} # word -> index in self.nodes
		self.index = 0 # current index for use with self.indices/self.nodes


	# separate function to add the first node to the graph
	def addfirst(self, token, idiom):
		self.indices[token] = self.index
		self.nodes.append(Node(idiom))
		self.heads.append(self.index)
		self.index += 1


	def add(self, token, prev="", idiom=None, isHead=False):
		# if the token has not been seen yet
		if token not in self.indices:
			self.indices[token] = self.index # map token to current index
			self.index += 1
		self.nodes.append(Node(idiom)) # add new token node to list of nodes

		# map previous node to current node
		p = self.indices[prev] # index of previous token
		self.nodes[p].addnext(self.index)

		if isHead:
			self.heads.append(self.index)


	def load(self, idioms):
		for idiom in idioms:
			#print('load', idiom)
			i = nlp(idiom)
			print('load', i)

			self.add(i[0].text, isHead=True) # first word is the head node
			prev = i[0].text

			for x in range(1, len(i) - 1):
				self.add(prev, i[x].text)
				prev = i[x].text

			self.add(prev, i[len(i)-1], idiom=idiom)




	# DFS implementation to cluster idioms on lexical overlap
	def cluster(self):
		clusters = []

		for head in self.heads:
			stack = [head] # frontier implemented as stack
			explored = set()
			frontier = set(head) # for searching through frontier
			cluster = set()

			while len(stack):
				node = stack.pop()
				frontier.remove(node)
				explored.add(node)

				if len(self.nodes[node].nextnodes) == 0:
					print(self.nodes[node].idiom)
					cluster.add('cluster', self.nodes[node].idiom)

				for n in self.nodes[node].nextnodes:
					if n not in frontier and n not in explored:
						stack.append(n)
						fronter.add(n)

			clusters.append(cluster)

		return clusters






class Node():
	def __init__(self, idiom):
		self.nextnodes = [] # indices of nodes that immediately follow self
		self.idiom = idiom # source of token this node represents


	def addnext(self, prev):
		self.nextnodes.append(prev)


# returns dictionary of lemmatized words and frequencies throughout entire lexicon
def count():
	words = defaultdict(int)
	with open("./IBM_Debater_(R)_SLIDE_LREC_2018/idiomLexicon.tsv", "r") as infile:
		for line in infile:
			l = line.split('\t')
			if l[11] != "X":
				sent = nlp(l[0])
				print(sent)
				for word in sent:
					words[word.lemma_] += 1

	with open('counts.json', 'w') as c:
		json.dump(words, c)

	return words


# store idioms in memory with option to remove stop words
def parse(stop=False):
	with open("./IBM_Debater_(R)_SLIDE_LREC_2018/idiomLexicon.tsv", "r") as infile:
		idioms = []
		for line in infile:
			l = line.split('\t')
			if l[11] != "X":
				sent = nlp(l[0])

				newsent = ""
				print(sent)
				for token in sent:
					if stop:
						if not token.is_stop:
							newsent += token.text + token.whitespace_
					else:
						newsent += token.text + token.whitespace_

				idioms.append(newsent)

	return idioms


# https://spacy.io/api/annotation#pos-tagging
def clusterbypos():
	pass


def clusterbyner():
	cluster = defaultdict(list) # default value is empty set

	# list of all named entities
	ner = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART",
			"LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"] 
	
	# initialize defaultdict so that all named entities are accounted for even if not seen
	for n in ner:
		cluster[n]

	for idiom in IDIOMS:
		doc = nlp(idiom)
		print(doc)
		for word in doc:
			cluster[word.ent_type_].append(idiom)


	with open('ner.json', 'w') as n:
		json.dump(cluster, n)

	return cluster





if __name__ == "__main__":
	print("starting parse")
	IDIOMS = parse()
	g = Graph()
	print("starting load")
	g.load(IDIOMS)
	print("starting cluster")
	print(g.cluster())


	# print(clusterbyner())

	#print('starting parse')
	#IDIOMS = parse()

	#print('starting ner')
	#clusterbyner()

	#print('starting count')
	#count()

