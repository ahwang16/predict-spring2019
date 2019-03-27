# cluster.py

# a file of helper files ot parse and cluster idioms in the SLIDE lexicon

from collections import defaultdict
import spacy
# from nltk.corpus import stopwords

# stopWords = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")
IDIOMS = parse()

# returns dictionary of lemmatized words and frequencies throughout entire lexicon
def count():
	words = defaultdict(int)
	with open("./IBM_Debater_(R)_SLIDE_LREC_2018/idiomLexicon.tsv", "r") as infile:
		for line in infile:
			l = line.split('\t')
			if l[11] != "X":
				for word in l[0]:
					words[words.lemma_] += 1

	return words


# store idioms only in memory without stop words
def parse():
	with open("./IBM_Debater_(R)_SLIDE_LREC_2018/idiomLexicon.tsv", "r") as infile:
		idioms = set()
		for line in infile:
			l = line.split('\t')
			if l[11] != "X":
				sent = nlp(l[0])

				newsent = ""

				for token in sent:
					if not token.is_stop:
						newsent += (token.text + token.whitespace_)

				idioms.add(newsent)

		return idioms


# https://spacy.io/api/annotation#pos-tagging
def clusterbypos():
	pass


def clusterbyner():
	cluster = defaultdict(set) # default value is empty set

	# list of all named entities
	ner = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART",
			"LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"] 
	
	# initialize defaultdict so that all named entities are accounted for even if not seen
	for n in ner:
		cluster[n]





if __name__ == "__main__":
	words = count()


