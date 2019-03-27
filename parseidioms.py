# parse idioms from SLIDE

# function to get separate text file of just idioms
def just_idioms():
	with open('./IBM_Debater_(R)_SLIDE_LREC_2018/idiomLexicon.tsv', 'r') as lexicon:
		with open('./IBM_Debater_(R)_SLIDE_LREC_2018/idioms.txt', 'w') as outfile:
			next(lexicon)
			for l in lexicon:
				outfile.write(l.split('\t')[0] + '\n')


# check for idioms in claims data
# claims data is space-separated [claim] [label]
def check_idioms():
	with open('./IBM_Debater_(R)_SLIDE_LREC_2018/idioms.txt', 'r') as idioms:
		with open('claimnonclaim.txt', 'r') as claims:
			with open('idiomclaims.txt', 'w') as outfile:
				for i in idioms:
#					print(i)
					for c in claims:
						print(c)
						idiom = i.split('t')[0]
						if idiom in c.split()[0]:
							outfile.write(c + '\t' + idiom)


if __name__ == "__main__":
	#just_idioms()
	#check_idioms()
