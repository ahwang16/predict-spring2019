# features to extract from sentences

#from itertools import chain
#import nltk
#from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score
#import pycrfsuite
import spacy
#import parsexml
from collections import Counter, defaultdict
import os, sys
import numpy as np

# taken from committed belief paper
# For the ith word in a sentence, return list of lexical features
def lexicalfeats(sent, i) :
	token = sent[i]
	isNumeric = token.is_digit
	pos = token.pos_
	verbType = token.tag_ if token.pos_ == "VERB" else "nil"
	modal = token.tag_ == "MD"

	return isNumeric, pos, verbType, modal


# taken from salience paper
def basicfeats(sent):
	length = len(sent)

	cap = 0
	entity = len(sent.ents)

	for word in sent:
#		print(word, word.shape_)
		if word.shape_[0] == "X":
			cap += 1
	cap /= length
	entity /= length
	
	return length, cap, entity


# returns list of dictionaries (feats) and list of labels (int)
def features(filename):
	nlp = spacy.load("en")
	feats = []
	labels = []
	with open(filename, "r") as infile:
		for line in infile:
			# NEED TO DO SOME PROCESSING FOR THE LABELS
			claim, label = line.split('\t')
			if len(claim) == 0:
				continue
			#print(claim, label)
			doc = nlp(claim)

			isNumeric = False
			pos = defaultdict(int)
			postags = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB', 'ADP', 'AUX',
				   'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ', 'PUNCT', 'SYM', 'X']
			for tag in postags:
				pos[tag] = 0
				#print(tag)
			verbType = "nil"
			modal = False

			for x in range(len(doc)):
				isNumeric_, pos_, verbType_, modal_ = lexicalfeats(doc, x)
		#		print(doc[x], isNumeric_, pos_, verbType_, modal_)
				isNumeric = isNumeric or isNumeric_
				pos[pos_] += 1
				if verbType_ is not "nil":
					verbType = verbType_
				modal = modal or modal_
			
			length, capital, entity = basicfeats(doc)

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
			labels.append(label)

	return feats, labels


def loadewe(filename):
	ewedict = {}
	with open(filename, 'r') as ewe:
		for line in ewe:
			l = line.split()
			ewedict[l[0]] = np.array(list(map(float, l[1:])))

	return ewedict


def svm(X, y, c=1.0):
	clf = SVC(C=c, gamma='auto')
	#return cross_val_score(clf, X, y, cv=5)
	clf.fit(X[:-500], y[:-500])
	prediction = clf.predict(X[-500:])
	print(prediction)
	acc = accuracy_score(y[-500:], prediction)
	print(acc)
	rec = recall_score(y[-500:], prediction)
	print(rec)
	return np.array(accuracy_score(y[-500:], prediction),
			recall_score(y[-500:], prediction),
			f1_score(y[-500], prediction))
	
def logreg(X, y):
	clf = LogisticRegression()
	#return cross_val_score(clf, X, y, cv=5)
	clf.fit(X[:-500], y[:-500])
	prediction = clf.predict(X[-500:])
	print(prediction)
	return np.array(accuracy_score(y[-500:], prediction),
			recall_score(y[-500:], prediction),
			f1_score(y[-500], prediction))


def embedfeatures(embeddings, infile):
	feats = []
	labels = []
	nlp = spacy.load("en")

	with open(infile, 'r') as data:
		for line in data:
			#print(line)

			claim, label = line.split('\t')

			if len(claim) == 0:
				continue

			doc = nlp(claim)

			f = []
			for token in doc:
				try:
					f.append(ewedict[token.text])
				except:
					f.append(np.zeros(300))

			feats.append(np.mean(f, axis=0))
			#feats.append(max(f))
			labels.append(int(label))

	return feats, labels


if __name__ == "__main__":
	print('loading ewedict')
	ewedict = loadewe('ewe_uni.txt')

	feats, y = embedfeatures(ewedict, sys.argv[1])
	
#	y = np.array(labels).reshape(-1, 1)
#	feats, y = features(sys.argv[1])

#	v = DictVectorizer(sparse=False)
#	X = v.fit_transform(feats)
	print('running models')
	scores = [svm(feats, y), logreg(feats, y)]
	
	print(scores)
	#for score in scores:
		#print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))



