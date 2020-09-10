#######
# File: Quiz6_drb160130.py
# Author: Dorian Benitez (drb160130)
# Date: 9/9/2020
# Purpose: CS 4395.001 - Quiz 6 (Relationships between Words)
#######

import math
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.wsd import lesk
from nltk.book import *
from nltk.corpus import wordnet as wn

# 1.    Print a hypernym of ‘reptile.n.01’
print('Hypernym: ', wn.synset('reptile.n.01').hypernyms())

# 2.    Print a hyponym of ‘reptile.n.01’
print('Hyponym: ', wn.synset('reptile.n.01').hyponyms()[0])

# 3.    Output the path_similarity of ‘shoot.v.01’ and ‘gun_down.v.01’
shoot = wn.synset('shoot.v.01')
gun_down = wn.synset('gun_down.v.01')
print('Path Similarity: ', shoot.path_similarity(gun_down))

# 4.    Output the Wu-Palmer similarity of ‘shoot.v.01’ and ‘gun_down.v.01’
print('Wu-Palmer Similarity: ', wn.wup_similarity(shoot, gun_down))

# 5.    Find a holonym of ‘kitchen.n.01’ and print the definition of that holonym
print('Kitchen Holonym Definition: ', wn.synset('kitchen.n.01').part_holonyms()[0].definition())

# 6.    Find entailments of synset ‘snore.v.01’ using the “.entailments()” method
print(wn.synset('snore.v.01').entailments())

# 7.    Use morphy to find the root form of ‘snoring’
print(wn.morphy('snoring', wn.VERB))

# 8.    Using the Lesk algorithm in NLTK, find the most likely synset of ‘arm’ in the sentence below,
#       and print the definition of that synset.
sent = ['The', 'soldier', 'was', 'convicted', 'of', 'selling', 'arms', 'in', 'the', 'war']
print(lesk(sent, 'arms'))

# 9.    Using VADER sentiment analysis, print the polarity of the statement above.
sentence = 'The soldier was convicted of selling arms in the war'
vs = SentimentIntensityAnalyzer().polarity_scores(sentence)
print(sentence, '\n\t', str(vs))

# 10.   Using NLTK Text object text6 Monty Python, calculate the pmi of ‘fire arrows’.
#       Is this likely to be a collocation?
text = ' '.join(text6.tokens)
voc = len(set(text6))
ab = text.count('fire arrows') / voc
a = text.count('fire') / voc
b = text.count('arrows') / voc
pmi = math.log2(ab / (a * b))
print('PMI: ', pmi)


