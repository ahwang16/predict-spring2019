# features to extract from sentences

from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import spacy
#import parsexml
from collections import Counter
import os, sys


# taken from committed belief paper
# For the ith word in a sentence, return list of lexical features
def lexicalfeats(sent, i) :
	token = sent[i]
	#	print(sent, i)
	#	print("token:", token, ", token type:", type(token), ", sent type:", type(sent))
	# daughters = {c.text.lower() for c in token.children}
	# ancestors = {h.lemma_.lower() for h in token.ancestors}
	# lemmas = {"tell", "accuse", "insist", "seem", "believe", "say", "find", "conclude", "claim", "trust", "think", "suspect", "doubt", "suppose"}
	# auxdaughter = "nil"
	# moddaughter = "nil"
	# for c in token.children:
	# 	if c.pos_ == "AUX":
	# 		auxdaughter = c.text
	# 	if c.tag_ == "MD":
	# 		moddaughter = c.text

	# feats = {
	# 	"isNumeric" : not token.is_alpha,
	# 	"POS" : token.pos_,
	# 	"verbType" : token.tag_ if token.pos_ == "VERB" else "nil",
	# 	"whichModalAmI" : token if token.tag_ == "MD" else "nil",
	# 	"amVBwithDaughterTo" : token.pos_ == "VERB" and "to" in daughters,
	# 	"haveDaughterPerfect" : ("has" in daughters or "have" in daughters or "had" in daughters),
	# 	"haveDaughterShould" : "should" in daughters,
	# 	"haveDaughterWh" : ("where" in daughters or "when" in daughters or "while" in daughters or "who" in daughters or "why" in daughters),
	# 	"haveReportingAncestor" : (token.pos_ == "VERB" and len(lemmas.intersection(ancestors))),
	# 	"parentPOS" : token.head.pos_,
	# 	"whichAuxIsMyDaughter" : auxdaughter,
	# 	"whichModalIsMyDaughter" : moddaughter
	# }


	isNumeric = not token.is_alpha
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

	# return {
	# 	"sentLength" : length,
	# 	"capital" : cap,
	# 	"entity" : entity
	# }

	return length, capital, entity


if __name__ == "__main__":
	nlp = spacy.load("en")
	doc = nlp(sys.argv[1])

	isNumeric = False
	pos = {}
	postags = ['PUNCT', 'SYM', 'X', 'ADJ', 'VERB', 'CONJ', 'NUM', 'DET', 'ADV',
			   'NOUN', 'PROPN', 'PART', 'PRON', 'SPACE', 'INTJ', ]
	for tag in postags:
		pos[tag] = 0
	verbType = ""
	modal = False

	for x in range(len(doc)):
		isNumeric_, pos_, verbType_, modal_ = lexicalfeats(doc, x)

		isNumeric = isNumeric or isNumeric_
		pos[pos_] += 1
		if verbType_ is not "nil":
			verbType = verbType_
		modal = modal or modal_
	
	length, capital, entity = basicfeats(doc)

	feat = {
		"isNumeric" : isNumeric,
		"verbType" : verbType,
		"modal" : modal
	}

	for p in pos:
		feat[p] = pos[p]
		
