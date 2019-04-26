# features to extract from sentences


from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import spacy
from collections import Counter, defaultdict
import os, sys
import numpy as np
import pickle as pkl

nlp = spacy.load("en")


def lexicalfeats(sent, i) :
	"""Returns lexical features for a token in input sentence/phrase
	Helper function for features() function

	Parameters
		- sent (spacy Doc): input sentence first passed through SpaCy
		- i (int): index of current token
	Returns
		- isNumeric (bool): True if current token is a digit, else False
		- pos (str): part of speech of this token
		- verbType (str): type of verb if pos of token is verb, else "nil"
		- modal (bool): True if current token is a modal verb, else False
	Source
		Committed Belief paper
	"""
	token = sent[i]
	isNumeric = token.is_digit
	pos = token.pos_
	verbType = token.tag_ if token.pos_ == "VERB" else "nil"
	modal = token.tag_ == "MD"

	return isNumeric, pos, verbType, modal


def basicfeats(sent):
	"""Returns sentence-level features for input sentence
	Helper function for features() function

	Parameters
		- sent (spacy Doc): input sentence first passed through SpaCy
	Returns
		- length (int): length of sent
		- cap (float): number of capitalized words normalized by length of sent
		- entity (float): number of named entities normalized by length of sent
	Source
		Salience paper
	"""
	length = len(sent)
	cap = 0
	entity = len(sent.ents) / length

	for word in sent:
		if word.shape_[0] == "X":
			cap += 1
	cap /= length
	
	return length, cap, entity


# returns list of dictionaries (feats) and list of labels (int)
def features(filename):
	"""Returns features and gold labels for every sent in filename
	Main feature function for lexical/sentence-level features
	The features extracted for each sentence are stored in a dictionary

	Parameters
		- filename (str): name of file containing claims and labels separated by
		  a tab
	Returns
		- feats (list of dictionaries): features extracted for all sentences
		- labels (list of ints): gold label for each sentence
		- Sentence order is preserved for feats and labels
	Source
		Committed Belief paper
		Salience paper
	"""
	feats = []
	labels = []

	with open(filename, "r") as infile:
		for line in infile:
			claim, label = line.split('\t')

			# skip empty data points
			if len(claim) == 0:
				continue
			
			doc = nlp(claim) # pass sent through SpaCy

			# Initialize features
			isNumeric = False
			pos = defaultdict(int)

			# Count of all possible POS labeled by SpaCy
			postags = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB', 'ADP', 'AUX',
				   'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ', 'PUNCT', 'SYM', 'X']
			for tag in postags:
				pos[tag] = 0
			
			verbType = "nil"
			modal = False

			# Going through each word
			for x in range(len(doc)):
				isNumeric_, pos_, verbType_, modal_ = lexicalfeats(doc, x)
				isNumeric = isNumeric or isNumeric_
				pos[pos_] += 1
				if verbType_ is not "nil":
					verbType = verbType_
				modal = modal or modal_
			
			length, capital, entity = basicfeats(doc)

			# Compiling feature dictionary for current claim
			feat = {
				"isNumeric" : isNumeric,
				"verbType" : verbType,
				"modal" : modal,
				"length" : length,
				"capital" : capital,
				"entity" : entity
			}

			for p in pos:
				feat[p] = pos[p]

			feats.append(feat)
			labels.append(int(label))

	return feats, labels


def loadewe(filename):
	"""Loads Emotional Word Embeddings from text file and saves it as pickle
	file named ewedict.pkl

	Parameters
		- filename (str): name of file containing EWE
	Returns
		- ewedict (dict: str -> numpy array): dictionary mapping word
		  to word embedding as 1x300 numpy array
	Source

	"""
	ewedict = {}
	with open(filename, 'r') as ewe:
		for line in ewe:
			l = line.split()
			ewedict[l[0]] = np.array(list(map(float, l[1:])))

	with open('ewedict.pkl', 'wb') as outfile:
		pkl.dump(ewedict, outfile)

	return ewedict


def svm(X, y, c=1.0, multi=False):
	"""Trains and tests C-Support Vector Classification from sklearn on given
	data (sklearn.svm.svc, penalty parameter of error can be user-controlled,
	gamma=auto, class_weight=balanced)
	Holds out last 500 data points for testing

	Parameters
		- X (np.array): all data/features in matrix form
		- y (list of ints): labels for X
		- c (float): penalty parameter of error term [default 1.0]
		- multi (bool): True if dataset is multiclass --> return precision,
		  recall, and f1 scores for each class; False if dataset is binary -->
		  return precision, recall, and f1 scores for positive label (the
		  default sklearn setting)
	Returns
		- 4-tuple of accuracy, precision, recall, and f1 scores
	"""
	clf = SVC(C=c, gamma='auto', class_weight='balanced')
	clf.fit(X[:-500], y[:-500])
	prediction = clf.predict(X[-500:])

	if multi:
		return (accuracy_score(y[-500:], prediction),
				precision_score(y[-500:], prediction, average=None),
				recall_score(y[-500:], prediction, average=None),
				f1_score(y[-500:], prediction, average=None))
	else:
		return (accuracy_score(y[-500:], prediction),
				precision_score(y[-500:], prediction),
				recall_score(y[-500:], prediction),
				f1_score(y[-500:], prediction))
	
def logreg(X, y, multi=False):
	"""Trains and tests logistic regression classifier from sklearn on given
	data (class_weight=balanced)
	Holds out last 500 data points for testing

	Parameters
		- X (np.array): all data/features in matrix form
		- y (list of ints): labels for X
		- multi (bool): True if dataset is multiclass --> return precision,
		  recall, and f1 scores for each class; False if dataset is binary -->
		  return precision, recall, and f1 scores for positive label (the
		  default sklearn setting)
	Returns
		- 4-tuple of accuracy, precision, recall, and f1 scores
	"""
	clf = LogisticRegression(class_weight='balanced')
	clf.fit(X[:-500], y[:-500])
	prediction = clf.predict(X[-500:])

	if multi:
		return (accuracy_score(y[-500:], prediction),
				precision_score(y[-500:], prediction, average=None),
				recall_score(y[-500:], prediction, average=None),
				f1_score(y[-500:], prediction, average=None))
	else:
		return (accuracy_score(y[-500:], prediction),
				precision_score(y[-500:], prediction),
				recall_score(y[-500:], prediction),
				f1_score(y[-500:], prediction))


def embedfeatures(embeddings, infile):
	"""Iterates through data file and returns emotional word embedding features
	for claims and corresponding list of gold labels
	If no word embedding exists for particular token, append matrix of 0s

	Parameters
		- embeddings (dict: str --> numpy array): dictionary of word embeddings
		- infile (str): name of file containing claim/label data
	Returns
		- feats (list of numpy arrays): mean pooling for word embeddings of each
		  input claim (dimensions: nx300, n = number of claims)
		- labels (list of ints): gold label for each claim represented as
		  matrix in feats
	Source

	"""
	feats = []
	labels = []

	with open(infile, 'r') as data:
		for line in data:
			claim, label = line.split('\t')

			if len(claim) == 0:
				continue

			doc = nlp(claim)

			f = []
			for token in doc:
				try:
					f.append(embeddings[token.text])
				except:
					f.append(np.zeros(300))

			feats.append(np.mean(f, axis=0))
			labels.append(int(label))

	return feats, labels


if __name__ == "__main__":
#	print('loading ewedict')
#	ewedict = loadewe('ewe_uni.txt')

#	feats, y = embedfeatures(ewedict, sys.argv[1])
	
#	y = np.array(labels).reshape(-1, 1)

	# python3 feat.py <data_file_name> <isMultiClass> <features> <optional c>
	X, y = None, None
	if sys.argv[3] == 'ewedict':
		with open('./datasets/ewedict.pkl', 'rb') as infile:
			ewedict = pkl.load(infile)
		X, y = embedfeatures(ewedict, sys.argv[1])
	elif sys.argv[3] == 'lexical':
		feats, y = features(sys.argv[1])
		v = DictVectorizer(sparse=False)
		X = v.fit_transform(feats)
	# concatenate word embedding and lexical features
	else:
		with open('./datasets/ewedict.pkl', 'rb') as infile:
			ewedict = pkl.load(infile)
		f_ewe, y = embedfeatures(ewedict, sys.argv[1])

		f_lex, y = features(sys.argv[1])
		v = DictVectorizer(sparse=False)
		X = v.fit_transform(f_lex)

		feats = np.concatenate((f_ewe, X))



	# feats, y = features(sys.argv[1])

	# v = DictVectorizer(sparse=False)
	# X = v.fit_transform(feats)
	print('running models')
	c = 1.0
	if len(sys.argv) == 5:
		c = sys.argv[4]
#	scores = [svm(feats, y), logreg(feats, y)]
	scores = [svm(X, y, c=c, multi=bool(int(sys.argv[2]))), logreg(X, y , multi=bool(int(sys.argv[2])))]	
	print(scores)
	#for score in scores:
		#print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))



