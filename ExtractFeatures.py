import pandas as pd
import re
import nltk
import numpy as np
from nltk import wordpunct_tokenize, pos_tag
from textblob import TextBlob
from nltk.corpus import stopwords
import statistics
from nltk.stem.snowball import SnowballStemmer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))

possibleTags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'RB',
                'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
vowels = ['a', 'o', 'u', 'i', 'e']


# This function calculated how often a queriedRepeition occurs in a text. If e.g. queriedRepetition is equal to 1
# the function calculates how many unique words are in a text. If the queriedRepetition is equal to 2 it calculates
# how many words occur twice in a text (and so on)
def calcNumberOfRepetitions(tokens, text, queriedRepetition):
    numberOfQueriedRepetitions = 0
    tokensAlreadyChecked = []
    for token in tokens:
        if token not in tokensAlreadyChecked:
            numberOfOccurences = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(token), text))
            if (numberOfOccurences == queriedRepetition):
                numberOfQueriedRepetitions += 1
            tokensAlreadyChecked.append(token)
    return numberOfQueriedRepetitions



def extract_features(text):
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    # bag_of_words_and_characters includes characters
    bag_of_words_and_characters = [x for x in wordpunct_tokenize(text)]
    # words_only doesn't include characters
    words_only = word_tokenizer.tokenize(text.lower())
    # stemmedwords
    stemmer = SnowballStemmer("english")
    stemmed_words = [stemmer.stem(x) for x in words_only]
    # vocab doesn't contain double elements
    vocab = set(words_only)

    sentences = sentence_tokenizer.tokenize(text)
    words_per_sentence = np.array([len(word_tokenizer.tokenize(s))
                                   for s in sentences])
    features = []





    # Count the number of words_only
    features.append(len(bag_of_words_and_characters))
    # Count the number of words_only, excluded the stopwords
    features.append(len([x for x in bag_of_words_and_characters if x.lower() not in stop_words]))
    # Number of just the stopwords
    features.append(len([x for x in bag_of_words_and_characters if x.lower() in stop_words]))
    # Average number of words_only per sentence
    features.append(words_per_sentence.mean())
    # Sentence length variation
    features.append(words_per_sentence.std())
    # Lexical diversity
    features.append(len(vocab) / float(len(words_only)))
    # Average word length
    word_lengths = [len(word) for word in words_only]
    features.append(sum(word_lengths)/len(word_lengths))
    # Median word length
    features.append(statistics.median(word_lengths))
    # Standarddeviation in word length
    features.append(statistics.stdev(word_lengths))
    # number of words with word length 1 up to word length 10
    for i in range (1, 10):
        features.append(len([word for word in words_only if len(word)==i]))
    # Number of unique words_only, words_only occuring twice and words_only occuring thrice in a text
    for i in range (1, 4):
        features.append(calcNumberOfRepetitions(words_only, text, i))
    # Number of words without vowels
    features.append(len([word for word in words_only if not any(vowel in word for vowel in 'aoeui')]))
    # Number of sentences
    features.append(len(sentences))
    # Sentiment
    blob = TextBlob(text)
    features.append(blob.sentiment[0])
    # Polarity
    features.append(blob.sentiment[1])
    # POS-tag frequencies
    tagged_words = pos_tag(words_only)
    for posTag in possibleTags:
        features.append(len([x for x in tagged_words if x[1] == posTag]))
    # Number of uppercase words_only not at the beginning of the sentence
    upper_list = re.findall(r'(?<!\.\s)\b[A-Z][a-z]*\b', text)
    features.append(len(upper_list))
    # Number of whitespace
    features.append(sum(c.isspace() for c in text))
    # Number of alphabetic chars
    features.append(sum(c.isalpha() for c in text))
    # Number of digits
    features.append(sum(c.isdigit() for c in text))
    # Number of special characters
    special_characters = ['!', ',', '.', ';', ':', '"', '-', '#', '&', '%', '|',
                '(', ')', '*', '@']
    for special_character in special_characters:
        count = text.count(special_character)
        features.append(count)


    #TODO ####### INCLUDE FASTTEXT #######

    return features

