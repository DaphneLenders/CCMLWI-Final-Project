import pandas as pd
import re
import nltk
import numpy as np
from nltk import wordpunct_tokenize, pos_tag
from textblob import TextBlob
from nltk.corpus import stopwords
from collections import defaultdict
import InverseDocumentFrequentizer
from nltk.stem.snowball import SnowballStemmer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))

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
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    bag_of_words = [x for x in wordpunct_tokenize(text)]
    tokens = nltk.word_tokenize(text.lower())
    words = word_tokenizer.tokenize(text.lower())
    sentences = sentence_tokenizer.tokenize(text)
    vocab = set(words)
    vowels = ['a', 'e', 'o', 'i', 'u']
    words_per_sentence = np.array([len(word_tokenizer.tokenize(s))
                                   for s in sentences])


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


    # pos tags frequencies
    tagged_words = pos_tag(tokens)
    for posTag in possibleTags:
        features.append(len([x for x in tagged_words if x[1] == posTag]))


    # Example feature 1: count the number of words
    features.append(len(bag_of_words))
    # Example feature 2: count the number of words, excluded the stopwords
    features.append(len([x for x in bag_of_words if x.lower() not in stop_words]))

    # feature3: number of just the stopwords
    features.append(len([x for x in bag_of_words if x.lower() in stop_words]))
    # feature4: average number of words per sentence
    features.append(words_per_sentence.mean())
    # feature5: sentence length variation
    features.append(words_per_sentence.std())
    # feature6: lexical diversity
    features.append(len(vocab) / float(len(words)))
    tokens = nltk.word_tokenize(text)
    pos_text = [p[1] for p in nltk.pos_tag(tokens)]

    # count frequencies for common POS types
    pos_list = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS', 'VB', 'POS', 'RB', 'VBD', 'CC',
                'EX', 'WDT', 'WRB', 'WP']
    for pos in pos_list:
        count = pos_text.count(pos)
        features.append(count)
    # feature18: number of sentences
    features.append(len(sentences))
    # feature31:number of uppercase words not at the beginning of the sentence
    upper_list = re.findall(r'(?<!\.\s)\b[A-Z][a-z]*\b', text)
    features.append(len(upper_list))
    # feature32:number of whitespace
    features.append(sum(c.isspace() for c in text))
    # feature33: number of alphabetic chars
    features.append(sum(c.isalpha() for c in text))

    list = ['!', ',', '.', ';', ':', '"', '-', '#', '&', '%', '|',
                '(', ')', '*', '@']
    for thing in list:
        count = text.count(thing)
        tokens.append(count)

    # word frequencies
    IDF = InverseDocumentFrequentizer.getIDF()
    for key, value in IDF.items():
        #tf_idf = text.count(key) * value
        features.append(text.count(key))
    
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

    '''character_trigrams = make_trigrams(list(text))
    all_character_trigrams = InverseDocumentFrequentizer.characterTrigrams
    for key, value in all_character_trigrams.items():
        number_of_trigram_occurences = calc_ngram_occurence(character_trigrams, key)
        features.append(number_of_trigram_occurences)
'''
    return features

