import pandas as pd
import os
import nltk
import numpy as np
from scipy.stats import entropy


def extract_test(path_name):
    text = ''
    for line in open(path_name):
        if 'End of the Project Gutenberg EBook' in line:
            break
        else:
            text = text + line
    return text

# When given a root directory (root_dir) and the name of the author whose influencers are checked, this
# function makes a dataframe consisting of the texts of influencers, along with their name and the author
# they've influenced
def read_directory(root_dir, influences):
    list = []
    for dir_name, sub_dir_list, file_list in os.walk(root_dir):
        print('Found directory: %s' % dir_name)
        author = os.path.basename(os.path.normpath(dir_name))
        for f_name in file_list:
            path_name = os.path.join(dir_name, f_name)
            text = extract_test(path_name)
            list.append([text, author, influences])

    dataframe = pd.DataFrame(list, columns=['Text', 'Author', 'Influences'])
    return dataframe



def calc_function_word_frequencies(groups):
    dataframe = pd.DataFrame({'function word': function_words['Word']})

    for (author, df) in groups:
        function_word_frequencies_total = np.zeros(len(function_words['Word']))
        for text in df['Text']:
            word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
            words_only = word_tokenizer.tokenize(text.lower())
            function_word_frequencies = np.array([words_only.count(function_word) for function_word in function_words['Word']])
            function_word_frequencies_total += function_word_frequencies

        dataframe[author] = function_word_frequencies_total.astype(int)
    return dataframe

def normalize_function_word_frequencies(dataframe):
    for column in dataframe.columns[1:]:
        sum_column = sum(dataframe[column])
        normalized_values = [value/sum_column for value in dataframe[column]]
        dataframe[column] = normalized_values
    return dataframe


def calc_kullback_leibner_divergence(author1, author2):
    return entropy(author1, author2)


path = 'Function Words.xlsx'
# function words are extracted from excel file
function_words = pd.read_excel(path, usecols = [0])

mary_shelley_influencers = read_directory('./Mary Shelley Influencers Data', "Mary Shelley")

function_word_frequencies = calc_function_word_frequencies(mary_shelley_influencers.groupby('Author'))



normalized_function_word_frequencies = normalize_function_word_frequencies(function_word_frequencies)
print(calc_kullback_leibner_divergence(normalized_function_word_frequencies['John Milton'], normalized_function_word_frequencies['Mary Wollstonecraft']))
#print(list(mary_shelley_influencers.groupby('Author')))