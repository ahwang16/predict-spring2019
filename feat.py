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
	daughters = {c.text.lower() for c in token.children}
	ancestors = {h.lemma_.lower() for h in token.ancestors}
	lemmas = {"tell", "accuse", "insist", "seem", "believe", "say", "find", "conclude", "claim", "trust", "think", "suspect", "doubt", "suppose"}
	auxdaughter = "nil"
	moddaughter = "nil"
	for c in token.children:
		if c.pos_ == "AUX":
			auxdaughter = c.text
		if c.tag_ == "MD":
			moddaughter = c.text

	feats = [
			"isNumeric=%s" % token.is_alpha,
			"POS=" + token.pos_,
			"verbType=" + (token.tag_ if token.pos_ == "VERB" else "nil"),
			"whichModalAmI=" + (token if token.tag_ == "MD" else "nil"),
			"amVBwithDaughterTo=%s" % (token.pos_ == "VERB" and "to" in daughters),
			"haveDaughterPerfect=%s" % ("has" in daughters or "have" in daughters or "had" in daughters),
			"haveDaughterShould=%s" % ("should" in daughters),
			"haveDaughterWh=%s" % ("where" in daughters or "when" in daughters or "while" in daughters or "who" in daughters or "why" in daughters),
			"haveReportingAncestor=%s" % (token.pos_=="VERB" and len(lemmas.intersection(ancestors))!=0),
			"parentPOS=" + token.head.pos_,
			"whichAuxIsMyDaughter=" + auxdaughter,
			"whichModalIsMyDaughter=" + moddaughter
		]

	return feats

'''
	feats = []
	# lexical and syntactic features with no context
	if feature == 0:
		feats = [
			"POS=" + token.pos_,
			"whichModalAmI=" + token.text if token.tag_ == "MD" else "nil",
			"parentPOS=" + token.head.pos_,

		]

	# lexical features with context
	elif feature == 1:
		feats = [
			"POS=" + token.pos_,
			"whichModalAmI=" + token.text if token.tag_ == "MD" else "nil",
			"verbType=" + token.tag_ if token.pos_ == "VERB" else "nil",
			"isNumeric%s" % str(token.is_alpha),
			"haveReportingAncestor=%s" % str(token.pos_=="VERB" and len(lemmas.intersection(ancestors))!=0),
			"whichModalIsMyDaughter=" + moddaughter,
			"whichAuxIsMyDaughter=" + auxdaughter,
			"haveDaughterShould=%s" % str("should" in daughters)
		]

	# lexical features with context and syntactic features with no context
	elif feature == 2:
		feats = [
			"POS=" + token.pos_,
			"whichModalAmI=" + token.text if token.tag_ == "MD" else "nil",
			"parentPOS=" + token.head.pos_,
			"haveReportingAncestor=%s" % str(token.pos_=="VERB" and len(lemmas.intersection(ancestors))!=0),
			"whichModalIsMyDaughter=" + moddaughter,
			"whichAuxIsMyDaughter=" + auxdaughter,
			"haveDaughterShould=%s" % str("should" in daughters)
		]

	# lexical and syntactic features with context
	elif feature == 3:
		feats = [
			"POS=" + token.pos_,
			"whichModalAmI=" + token.text if token.tag_ == "MD" else "nil",
			"parentPOS=" + token.head.pos_,
			"haveReportingAncestor=%s" % str(token.pos_=="VERB" and len(lemmas.intersection(ancestors))!=0),
			"whichModalIsMyDaughter=" + moddaughter,
			"haveDaughterPerfect=%s" % str("has" in daughters or "have" in daughters or "had" in daughters),
			"whichAuxIsMyDaughter=" + auxdaughter,
			"haveDaughterWh=%s" % str("where" in daughters or "when" in daughters or "while" in daughters or "who" in daughters or "why" in daughters),
			"haveDaughterShould=%s" % str("should" in daughters)
		]
'''


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

	return ["sentLength=" + str(length), "capital=" + str(cap), "entity=" + str(entity)]







if __name__ == "__main__":
	nlp = spacy.load("en")
	doc = nlp(sys.argv[1])

	for x in range(len(doc)):
		print(lexicalfeats(doc, x))
	
	print(basicfeats(doc))	
