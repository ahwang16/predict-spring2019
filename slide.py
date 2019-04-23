# slide.py

'''
Method to calculate pleasantness score of phrase modified from sentiment
paper. Use DAL to assign score for each word, finite state machine to handle
negations, normalize by length of phrase.
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
import json
import spacy
import pickle as pkl


nlp = spacy.load('en_core_web_sm')

with open('./datasets/dal.json', 'r') as infile:
	daldict = json.load(infile)

# load DAL
def dal():
	with open("dict_of_affect.txt", "r") as dal:
		for line in dal :
				linesplit = line.split()
				daldict[linesplit[0]] = [float(linesplit[1]), float(linesplit[2]), float(linesplit[3])]

	with open('dal.json', 'w') as dal:
		json.dump(daldict, dal)

	return daldict


# get scores from DAL
# 0 = pleasant, 1 = activation, 2 = imagery
def assignscore(sent, index):
	tokens = nlp(sent)
	vec = []
	for t in tokens:
		try:
			vec.append(daldict[t.text][index])
		except:
			vec.append(tryagain(t.text, index))

	return vec


def tryagain(word, index):
	syns = wn.synsets(word)

	if len(syns) == 0:
		return 0

	for syn in syns:
		for lemma in syn.lemmas():
			try:
				return daldict[lemma.name()][index]
			except:
				pass

	for syn in syns:
		for lemma in syn.lemmas():
			ants = lemma.antonyms()
			if ants:
				for ant in ants:
					try:
						return (-1 * daldict[ant.name()][index])
					except:
						pass

	return 0





#	setlen = len(syns)
#	synlen = len(syns[0])
#	x, y = 0

#	while x < setlen:
#		while y < synlen:
#			try:
#				return daldict[syns[x].lemmas()[y].name()][index]
#			except:
#				y += 1
#		y = 0
#		x += 1
#		synlen = len(syns[x])

#	for syn in syns:
#		for l in syn.lemmas():
#			ants = l.antonyms()
#			if ants:
#				for a in ants:
#					try:
#						return (-1 * daldict[a.name()][index])
#					except:
#						pass
#	return 0


# finite state machine to RETAIN or INVERT (negation)
# Invert state: sign of pleasantness score is inverted
# Retain state: sign of pleasantness score stays the same
# sent: sentence (string)
# scores: vector of DAL pleasantness scores
# return updated pleasant scores
def fsm_negate(sent, scores):
	# search for list of negation words --> look at 2005 paper on contextual polarity
	# expand example of negation words https://www.grammarly.com/blog/negatives/
	# on the word wont: one's customary behavior in a particular situation.
	# "Constance, as was her wont, had paid her little attention" --> infrequent
	negate = {"not", "no", "never", "cannot", "didn't", "can't", "cant", "didnt", "couldnt",
				"shouldnt", "couldn't", "shouldn't", "nobody", "nothing", "nowhere", "neither",
				"nor", "none", "doesn't", "doesnt", "isn't", "isnt", "wasn't", "wasnt",
				"wouldn't", "wouldnt", "won't", "wont"}

	# comparative degree adjectives http://www.sparklebox.co.uk/literacy/vocabulary/word-lists/comparatives-superlatives/#.W8E_2xNKjyw
	comp_adj = {"worse", "better", "angrier", "bigger", "blacker", "blander", "bluer", "bolder", "bossier",
				"braver", "briefer", "brighter", "broader", "busier", "calmer", "cheaper", "chewier", "chubbier",
				"classier", "cleaner", "cleverer", "closer", "cloudier", "clumsier", "coarser", "colder",
				"cooler", "crazier", "creamier", "creepier", "crispier", "crunchier", "curly", "curvier",
				"cuter", "damper", "darker", "deadlier", "deeper", "denser", "dirtier", "drier", "duller",
				"dumber", "dustier", "earlier", "easier", "fainter", "fairer", "fancier", "farther",
				"faster", "fatter", "fewer", "fiercer", "filthier", "finer", "firmer", "fitter", "flakier", "flatter",
				"fresher", "friendlier", "fuller", "funnier", "gentler", "gloomier", "greasier", "greater", "greedier",
				"grosser", "hairier", "handier", "happier", "harder", "harsher", "healthier", "heavier", "higher",
				"hipper", "hotter", "humbler", "hungrier", "icier", "itchier", "juicier", "kinder", "larger", "later",
				"lazier", "lighter", "likelier", "littler", "livelier", "longer", "louder", "lovelier", "lower", "madder",
				"meaner", "messier", "milder", "moister", "narrower", "nastier", "naughtier", "nearer", "neater", "needier",
				"newer", "nicer", "noisier", "odder", "oilier", "older", "elder", "plainer", "politer",
				"poorer", "prettier", "prouder", "purer", "quicker", "quieter", "rarer", "rawer", "richer",
				"riper", "riskier", "roomier", "rougher", "ruder", "rustier", "sadder", "safer", "saltier", "saner",
				"scarier", "shallower", "sharper", "shinier", "shorter", "shyer", "sillier", "simpler", "sincerer",
				"skinnier", "sleepier", "slimmer", "slimier", "slower", "smaller", "smarter", "smellier", "smokier",
				"smoother", "softer", "sooner", "sorer", "sorrier", "sourer", "spicier", "steeper", "stingier",
				"stranger", "stricter", "stronger", "sunnier", "sweatier", "sweeter", "taller", "tanner", "tastier",
				"thicker", "thinner", "thirstier", "tinier", "tougher", "truer", "uglier", "warmer", "weaker",
				"wealthier", "weirder", "wetter", "wider", "wilder", "windier", "wiser", "worldlier", "worthier", "younger"}

	# state is True for RETAIN and False for INVERT
	# start with RETAIN
	state = False
	index = 0 # to reference corresponding pleasantness score in scores
	for word in sent.split():
		# RETAIN: leave score
		# switch to INVERT if current word is a negation
		if state: # RETAIN
			state = word not in negate

		# INVERSE: negate score
		# switch to RETAIN if current word is but or a comparative degree adjective
		else:
			scores[index] *= -1
			state = word=="but" or word in comp_adj

		index += 1

	return scores


# :input: array of pleasantness scores
# Z-normalize scores using mean and stdev found in manual (Whissel, 1989)
# boost score by multiplying by normalized score distance from mean
# https://www.god-helmet.com/wp/whissel-dictionary-of-affect/index.htm
# pleasantness: mean 1.85, stdev 0.36
# :return: array of Z-normalized scores
def normalize_dal(p, index):
	if sum(p) == 0:
		return None

	# p a i
	mean = {0: 1.85, 1: 1.67, 2: 1.52}
	stdev = {0: 0.36, 1: 0.36, 2: 0.63}

	meanp, stdevp = mean[index], stdev[index]
	for x in range(len(p)):
		p[x] = (p[x] - meanp) / stdevp
		p[x] *= (abs(p[x] - meanp) / stdevp)

	return p


# update DAL scores with FSM
# :return: single value for entire phrase (sum of values normalized by phrase length)
def dal_score(sent, index):
	scores = normalize_dal(fsm_negate(sent, assignscore(sent, index)), index)
	
	if scores:
		return sum(scores) / len(scores)
	return None


# json-dump dictionary --> str(idiom) : (float(positive_percent), str(sentiment))
def parsepositive():
	with open("./IBM_Debater_(R)_SLIDE_LREC_2018/idiomLexicon.tsv", "r") as infile:
		idioms = {}

		next(infile)

		for line in infile:
			l = line.split('\t')
			print(l[0])
			if l[11] != 'X':
				score = float(l[7]) - float(l[9]) - float(l[8])

				idioms[l[0]] = (score, l[10])

	with open('idiomLexicon.json', 'w') as outfile:
		json.dump(idioms, outfile)

	return idioms


# assign and save DAL scores
# json-dump dictionary --> str(idiom) : ((p, a, i scores), float(positive_percent), str(sentiment))
def savescore():
	with open('idiomLexicon.json', 'r') as infile:
		idioms = json.load(infile)

	idiomsnew = {}
	count = 0
	for idiom in idioms:
#		if count == 10:
#			break
		print(str(count))
		print(idiom)
		score = (dal_score(idiom, 0), dal_score(idiom, 1), dal_score(idiom, 2))

		idiomsnew[idiom] = (score, idioms[idiom][0], idioms[idiom][1])
		count += 1
	with open('idiomLexiconScored.json', 'w') as outfile:
		json.dump(idiomsnew, outfile)


# index: 0 = pleasant, 1 = activation, 2 = imagery
def displayallraw(index, outfile):
	with open("./IBM_Debater_(R)_SLIDE_LREC_2018/idiomLexicon.tsv", "r") as infile:
		s_scores = []
		d_scores = []
		idioms = []
		s_senti = []

		sentiment = { "positive" : "r",
			      "negative" : "b",
			      "neutral" : "g",
			      "inappropriate" : "y" }

		label = { 0: "pleasantness", 1: "activation", 2: "imagery" }

		for line in infile:
			l = line.split("\t")

			if l[11] != "X":
				s = float(l[7]) - float(l[9]) - float(l[8])
				d = dal_score(l[0], index)
				
				if d is None:
					continue

				s_scores.append(s)
				d_scores.append(d)
				idioms.append(l[0])
				s_senti.append(sentiment[l[10]])

				#print(l[0], s, d, l[10])

	plt.scatter(s_scores, d_scores, c=s_senti)
	plt.xlabel("SLIDE positive percent")
	plt.ylabel("DAL {} index".format(label[index]))
	plt.savefig(outfile)


def displaysubset(index, idioms, outfile, title):
	# idiomLexicon.json is a json-dumped dictionary
	# idiom : ((p, i, a), positive_score, sentiment)
	with open('./datasets/idiomLexiconScored.json', 'r') as infile:
		record = json.load(infile)

	s_scores = []
	d_scores = []

	sentiment = { "positive" : "r",
		      "negative" : "b",
		      "neutral" : "g",
		      "inappropriate" : "y" }

	s_senti = []

	label = { 0: "pleasantness", 1: "imagery", 2: "activation" }

	for idiom in idioms:
		print(idiom)
		d = record[idiom][0][index]

		if d is None:
			continue

		d = float(d)

		s_scores.append(float(record[idiom][1]))
		d_scores.append(d)
		s_senti.append(sentiment[record[idiom][2]])

	plt.scatter(s_scores, d_scores, c=s_senti)
	plt.title(title)
	plt.xlabel('SLIDE positive percent')
	plt.ylabel('DAL {} index'.format(label[index]))
	plt.savefig(outfile)



if __name__ == "__main__":
	# dal()
	# parse()

	# parsepositive()


#	savescore()

	with open('./datasets/pos.pkl', 'rb') as ner:
		data = pkl.load(ner)

	# data is dictionary of named_entity -> [idioms]
	for x in range(3):
		for namedent in data:
			displaysubset(x, data[namedent], 'pos_{}_{}.png'.format(x, namedent), namedent)




