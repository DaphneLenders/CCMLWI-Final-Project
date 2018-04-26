import pandas as pd
import re
from nltk import wordpunct_tokenize, pos_tag
from textblob import TextBlob
from collections import defaultdict
import InverseDocumentFrequentizer
from nltk.stem.snowball import SnowballStemmer


possibleTags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'RB',
                'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
characters = ['.', '!', '?', ',', ';', ':', '-']
stop_words_all = InverseDocumentFrequentizer.stop_words_all


#This function calculates what the average word length of all the words in a text is
def calcAverageWordLength(tokens):
    lengthSum = 0
    for token in tokens:
        lengthSum += len(token)
    return lengthSum/len(tokens)

def make_bigrams(words):
    length = len(words)
    bigrams = []
    for i in range(length - 1):
        if not (words[i].isdigit() or words[i+1].isdigit()):
            bigrams.append((words[i], words[i + 1]))
    return bigrams

def make_stemmed_bigrams(words):
    length = len(words)
    bigrams = []
    for i in range(length - 1):
        if not (words[i].isdigit() or words[i+1].isdigit()):
            bigrams.append((words[i], words[i + 1]))
    return bigrams

def make_trigrams(words):
    length = len(words)
    trigrams = []
    for i in range(length - 2):
        if not (words[i].isdigit() or words[i + 1].isdigit() or words[i+2].isdigit()):
            trigrams.append((words[i], words[i + 1], words[i+2]))
    return trigrams

def make_function_word_trigrams(words):
    length = len(words)
    trigrams = []
    for i in range(length - 2):
        if (words[i] in stop_words_all and words[i + 1] in stop_words_all and words[i + 2] in stop_words_all):
            trigrams.append((words[i], words[i + 1], words[i + 2]))
    return trigrams
# Given all n-grams in a song, and some queried n-gram this function calculates how often the queried n-gram occurs
# in the code
def calc_ngram_occurence(ngrams, queriedNgram):
    numberOfngramOccurences = 0
    for ngram in ngrams:
        if ngram==queriedNgram:
            numberOfngramOccurences += 1
    return numberOfngramOccurences


def extract_features(text):
    features = []
    text = str(text)

    tokens = wordpunct_tokenize(text)
    words_only = re.compile('\w+').findall(text)

    stemmer = SnowballStemmer("english")
    stemmed_words = [stemmer.stem(x) for x in words_only]

    # number of tokens
    features.append(len(tokens))

    # average word length
    features.append(calcAverageWordLength(tokens))

    # sentiment and polarity
    blob = TextBlob(text)
    features.append(blob.sentiment[0])
    features.append(blob.sentiment[1])

    '''
    # pos tags frequencies
    tagged_words = pos_tag(tokens)
    for posTag in possibleTags:
        features.append(len([x for x in tagged_words if x[1] == posTag]))
    
    
    # word frequencies
    IDF = InverseDocumentFrequentizer.getIDF()
    for key, value in IDF.items():
        #tf_idf = text.count(key) * value
        features.append(text.count(key))
    '''
    bigrams = make_bigrams(stemmed_words)
    all_bigrams = InverseDocumentFrequentizer.getBigrams()
    for key, value in all_bigrams.items():
        number_of_bigram_occurences = calc_ngram_occurence(bigrams, key)
        features.append(number_of_bigram_occurences)
    '''
    trigrams = make_function_word_trigrams(words_only)
    all_trigrams = InverseDocumentFrequentizer.getFunctionWordTrigrams()
    for key, value in all_trigrams.items():
        number_of_trigram_occurences = calc_ngram_occurence(trigrams, key)
        features.append(number_of_trigram_occurences)
    '''
    # character frequencies
    for character in characters:
        number_of_character_occurences = text.count(character)
        features.append(number_of_character_occurences)
    '''
    character_trigrams = make_trigrams(list(text))
    all_character_trigrams = InverseDocumentFrequentizer.characterTrigrams
    for key, value in all_character_trigrams.items():
        number_of_trigram_occurences = calc_ngram_occurence(character_trigrams, key)
        features.append(number_of_trigram_occurences)
    '''
    return features

