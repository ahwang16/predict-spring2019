# slide.py

'''
Method to calculate pleasantness score of phrase modified from sentiment
paper. Use DAL to assign score for each word, finite state machine to handle
negations, normalize by length of phrase.
'''

daldict = {}


# load DAL
def dal():
	with open("dict_of_affect.txt", "r") as dal:
		for line in dal :
				linesplit = line.split()
				daldict[linesplit[0]] = [float(linesplit[1]), float(linesplit[2]), float(linesplit[3])]
	return None


# get pleasant scores from DAL
def assign_pleasant(sent):
	tokens = sent.split()
	pleasant = []

	for t in tokens:
		try:
			pleasant.append(daldict[t][0])
			length += 1
		except:
			pleasant.append(0)

	return pleasant


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
	negate = ["not", "no", "never", "cannot", "didn't", "can't", "cant", "didnt", "couldnt",
				"shouldnt", "couldn't", "shouldn't", "nobody", "nothing", "nowhere", "neither",
				"nor", "none", "doesn't", "doesnt", "isn't", "isnt", "wasn't", "wasnt",
				"wouldn't", "wouldnt", "won't", "wont"]

	# comparative degree adjectives http://www.sparklebox.co.uk/literacy/vocabulary/word-lists/comparatives-superlatives/#.W8E_2xNKjyw
	comp_adj = ["worse", "better", "angrier", "bigger", "blacker", "blander", "bluer", "bolder", "bossier",
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
				"wealthier", "weirder", "wetter", "wider", "wilder", "windier", "wiser", "worldlier", "worthier", "younger"]

	# state is True for RETAIN and False for INVERT
	# start with RETAIN
	state = False

	index = 0 # to reference corresponding pleasantness score in scores
	for word in sent:
		# INVERT: negate score
		# switch to RETAIN if current word is but or a comparative degree adjective
		if state:
			scores[index] *= -1
			state = not (word=="but" or word in comp_adj)
		
		# RETAIN: leave score
		# switch to INVERT if current word is a negation
		else:
			state = word in comp_adj

		index += 1

	return scores


# :input: array of pleasantness scores
# Z-normalize scores using mean and stdev found in manual (Whissel, 1989)
# boost score by multiplying by normalized score distance from mean
# https://www.god-helmet.com/wp/whissel-dictionary-of-affect/index.htm
# pleasantness: mean 1.85, stdev 0.36
# :return: array of Z-normalized scores
def normalize_dal(p):
	meanp, stdevp = 1.85, 0.36
	for x in range(len(p)):
		p[x] = (p[x] - meanp) / stdevp
		p[x] *= abs(p[x] - meanp)

	return p


# update DAL scores with FSM
# :return: single value for entire phrase (sum of values normalized by phrase length)
def dal_score(sent):
	scores = normalize_dal(fsm_negate(sent, assign_pleasant(sent)))
	return sum(scores) / len(scores)


def parse():
	with open("./IBM_Debater_(R)_SLIDE_LREC_2018/idiomLexicon.tsv", "r") as infile:
		s_scores = []
		d_scores = []
		idioms = []

		next(infile)

		for line in infile:
			l = line.split("\t")

			if l[11] != "X":
				s = float(l[7])
				d = dal_score(l[0])

				s_scores.append(s)
				d_scores.append(d)
				idioms.append(l[0])

				print(l[0], s, d)


if __name__ == "__main__":
	dal()
	parse()










