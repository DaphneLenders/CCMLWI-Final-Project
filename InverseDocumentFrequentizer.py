from collections import defaultdict
import re
import math
from nltk.stem.snowball import SnowballStemmer

stop_words_all = ['must', 'can', 'cannot', 'could', 'need', 'needs', 'should', 'is', 'are', 'would', 'aint', 'had', 'has',
                  'hasnt', 'have', 'hadnt', 'havent', 'having', 'it', 'its', 'itself',
               'than', 'where', 'does', 'doesnt', 'dont', 'do', 'didnt', 'did', 'were', 'werent', 'before',
              'doing', 'a', 'an', 'so', 'own', 'above', 'and', 'such', 'then', 'from',
              'for', 'any', 'how', 'again', 'very', 'but', 'once', 'because',
              'some', 'wont', 'to', 'these', 'with', 'the', 'them', 'both', 'few', 'after', 'wasnt',
              'be', 'been', 'against', 'on', 'off', 'too', 'under', 'why', 'who', 'our', 'in', 'if', 'other',
              'until', 'down', 'during', 'was', 'over', 'those', 'further', 'same', 'that', 'thats', 'of', 'up',
              'what', 'just', 'will', 'as', 'no', 'each', 'being', 'now', 'all', 'by', 'into', 'when', 'most',
              'not', 'theirs', 'they', 'at', 'out', 'below', 'whom', 'nor', 'here', 'through', 'more', 'or',
              'only', 'about', 'which', 'there', 'theres', 'this', 'between', 'while', 'might', 'she', 'shes', 'shell', 'her', 'herself',
              'he', 'hes', 'hell', 'him', 'his', 'himself', 'i', 'me', 'myself', 'am', 'you', 'youre', 'youll', 'your', 'yours',
              'yourself', 'youve', 'youd', 'we', 'want', 'wanted']

global IDF
global bigramFrequencies
global trigramFrequencies
global DF_stemmed
global characterTrigrams
global functionWordTrigrams

stemmer = SnowballStemmer("english")

def getIDF():
    return IDF

def getDF_stemmed():
    return DF_stemmed

def getBigrams():
    return bigramFrequencies

def getTrigrams():
    return trigramFrequencies

def getCharacterTrigrams():
    return characterTrigrams

def getFunctionWordTrigrams():
    return functionWordTrigrams

def idf_vectorizer(all_texts):
    DF = calc_document_frequencies(all_texts)
    global IDF
    IDF = inverse_document_frequencies(DF, len(all_texts))
    return IDF


def calc_document_frequencies(all_texts):
    wordsAlreadyChecked = []
    DF = defaultdict(int)

    for text in all_texts:
        text = str(text)
        text = text.lower()
        words = re.compile('\w+').findall(text)
        for word in words:
            if word not in wordsAlreadyChecked and not word.isdigit():
                DF[word] += 1
                wordsAlreadyChecked.append(word)
        wordsAlreadyChecked = []
    return DF

def inverse_document_frequencies(document_frequencies, number_of_documents):
    IDF = defaultdict(int)
    for key, value in document_frequencies.items():
        IDF[key] = math.log((number_of_documents/(1+value)), 10)

    return IDF

def calc_document_frequencies_stemmed_words(all_texts):
    wordsAlreadyChecked = []
    global DF_stemmed
    DF_stemmed = defaultdict(int)

    for text in all_texts:
        text = str(text)
        text = text.lower()
        words = re.compile('\w+').findall(text)
        stemmed_words = [stemmer.stem(x) for x in words]
        for word in stemmed_words:
            if word not in wordsAlreadyChecked and not word.isdigit():
                DF_stemmed[word] += 1
                wordsAlreadyChecked.append(word)
        wordsAlreadyChecked = []
    return DF_stemmed

#This function generates the bigrams of all texts of the training data set
def generate_bigrams(allTexts):
    global bigramFrequencies
    bigramFrequencies = defaultdict(int)

    for text in allTexts:
        text = str(text)
        text = text.lower()
        bigrams_already_checked = []
        words = re.compile('\w+').findall(text)
        length = len(words)
        for i in range(length-1):
            if not (words[i].isdigit() or words[i+1].isdigit()):
                bigram = (words[i], words[i+1])
                if (bigram not in bigrams_already_checked):
                    bigramFrequencies[bigram] += 1
                    bigrams_already_checked.append(bigram)
    return bigramFrequencies

#This function generates the bigrams of all texts of the training data set
def generate_stemmed_bigrams(allTexts):
    global bigramFrequencies
    bigramFrequencies = defaultdict(int)

    for text in allTexts:
        text = str(text)
        text = text.lower()
        bigrams_already_checked = []
        words = re.compile('\w+').findall(text)
        stemmed_words = [stemmer.stem(x) for x in words]
        length = len(stemmed_words)
        for i in range(length-1):
            if not (stemmed_words[i].isdigit() or stemmed_words[i+1].isdigit()):
                bigram = (stemmed_words[i], stemmed_words[i+1])
                if (bigram not in bigrams_already_checked):
                    bigramFrequencies[bigram] += 1
                    bigrams_already_checked.append(bigram)
    return bigramFrequencies

#This function generates the trigrams of all texts in the training dataset
def generate_trigrams(allTexts):
    global trigramFrequencies
    trigramFrequencies = defaultdict(int)
    for text in allTexts:
        text = str(text)
        text = text.lower()
        trigrams_already_checked = []
        words = re.compile('\w+').findall(text)
        length = len(words)
        for i in range(length - 2):
            if not (words[i].isdigit() or words[i+1].isdigit()):
                trigram = (words[i], words[i + 1], words[i+2])
                if (trigram not in trigrams_already_checked):
                    trigramFrequencies[trigram] += 1
                    trigrams_already_checked.append(trigram)
    return trigramFrequencies


def generate_character_trigrams(allTexts):
    global characterTrigrams
    characterTrigrams = defaultdict(int)
    for text in allTexts:
        text = str(text)
        text = text.lower()
        trigrams_already_checked = []
        characters = list(text)
        length = len(characters)
        for i in range(length - 2):
            if not (characters[i].isdigit() or characters[i+1].isdigit()):
                trigram = (characters[i], characters[i + 1], characters[i+2])
                if (trigram not in trigrams_already_checked):
                    characterTrigrams[trigram] += 1
                    trigrams_already_checked.append(trigram)
    return characterTrigrams

def generate_functionword_trigrams(allTexts):
    global functionWordTrigrams
    functionWordTrigrams = defaultdict(int)
    for text in allTexts:
        text = str(text)
        text = text.lower()
        trigrams_already_checked = []
        words = re.compile('\w+').findall(text)
        length = len(words)
        for i in range(length - 2):
            if (words[i] in stop_words_all and words[i + 1] in stop_words_all and words[i+2] in stop_words_all):
                trigram = (words[i], words[i + 1], words[i + 2])
                if (trigram not in trigrams_already_checked):
                    functionWordTrigrams[trigram] += 1
                    trigrams_already_checked.append(trigram)
    return functionWordTrigrams
