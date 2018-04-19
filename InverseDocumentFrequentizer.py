from collections import defaultdict
import re
import math

global IDF

def getIDF():
    return IDF

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