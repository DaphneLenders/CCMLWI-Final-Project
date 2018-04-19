import pandas as pd
import re
from nltk import wordpunct_tokenize, pos_tag
from textblob import TextBlob
import InverseDocumentFrequentizer

possibleTags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'RB',
                'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']


#This function calculates what the average word length of all the words in a text is
def calcAverageWordLength(tokens):
    lengthSum = 0
    for token in tokens:
        lengthSum += len(token)
    return lengthSum/len(tokens)

def extract_features(text):
    features = []
    text = str(text)

    tokens = wordpunct_tokenize(text)

    # number of tokens
    features.append(len(tokens))

    # average word length
    features.append(calcAverageWordLength(tokens))

    # sentiment and polarity
    blob = TextBlob(text)
    features.append(blob.sentiment[0])
    features.append(blob.sentiment[1])

    # pos tags frequencies
    tagged_words = pos_tag(tokens)
    for posTag in possibleTags:
        features.append(len([x for x in tagged_words if x[1] == posTag]))

    # word frequencies
    IDF = InverseDocumentFrequentizer.getIDF()
    for key, value in IDF.items():
        tf_idf = text.count(key) * value
        features.append(tf_idf)


    return features

