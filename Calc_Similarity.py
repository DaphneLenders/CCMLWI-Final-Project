import pandas as pd
import os
import nltk
import numpy as np
from scipy.stats import entropy
from math import exp
from nltk import pos_tag


# This function opens a file with a given path_name and puts the text of the file in
# a string. Only the text up until the line 'End of the Project Gutenberg EBook' is
# included, since everything after this line is not relevant anymore.
def extract_test(path_name):
    text = ''
    for line in open(path_name,encoding = "ISO-8859-1"):
        #print(line)
        if 'End of the Project Gutenberg EBook' in line:
            #print(line)
            break
        else:
            text = text + line   
    return text

# The data of the kaggle-authors is stored in a pandas dataframe. All author names
# are changed such that no abbreveations but their actual names are used
def read_author_data():
    data = pd.read_csv('train.csv')
    data.loc[data['Author'] == 'EAP', 'Author'] = 'Edgar Allan Poe'
    data.loc[data['Author'] == 'HPL', 'Author'] = 'H. P. Lovecraft'
    data.loc[data['Author'] == 'MWS', 'Author'] = 'Mary Shelley'
    return data

# A function to create a dataframe from influencer data that is stored in the input directory.
# Each person that has been influenced is stored as a key in the dictionary, and the values 
# in this dictionary are arrays that contain that person's infleuncers. 
# The created dataframe contains 2 fixed columns ('text' and 'author'). 
# Each row corresponds to one text sample.
# In addition, the dataframe contains one column per influenced author, which stores boolean values
# denoting whether or not the author has been influenced by the author of the text in that row. 
def create_influencer_dataframe(influencer_directory,influencer_dict):
    # First of all, create the dataframe with the fixed columns:
    dataframe = read_influencer_directory(influencer_directory)
    # Now, dynamically add columns to the dataframe: 
    for influenced_author in influencer_dict:
        # First, set the whole column to false by default.
        dataframe[influenced_author] = 0
        # Fill in true for each row who's author has influenced the influenced_author
        for influencer in influencer_dict[influenced_author]:
            dataframe.loc[dataframe['Author'] == influencer,influenced_author] = 1
    return dataframe                     

# When given a root directory (root_dir) and the name of the author whose influencers are checked, this
# function makes a dataframe consisting of the texts of influencers, along with their name and the author
# they've influenced
def read_influencer_directory(root_dir):
    list = []
    for dir_name, sub_dir_list, file_list in os.walk(root_dir):
        #print('Found directory: %s' % dir_name)
        author = os.path.basename(os.path.normpath(dir_name))
        for f_name in file_list:
            if f_name.endswith(".txt"):
                path_name = os.path.join(dir_name, f_name)
                text = extract_test(path_name)
                list.append([text, author])#, influences])
    dataframe = pd.DataFrame(list, columns=['Text', 'Author'])
    return dataframe

# Given a dataframe that is grouped by authors, the number of times an author uses certain function
# words (as stored in the list 'function_words') is calculated and stored in a dataframe
def calc_function_word_frequencies(groups):
    dataframe = pd.DataFrame({'function word': function_words['Word']})

    for (author, df) in groups:
        function_word_frequencies_total = np.zeros(len(function_words['Word']))
        for text in df['Text']:
            # split the text into tokens
            word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
            words_only = word_tokenizer.tokenize(text.lower())
            # count how many times the text contains the given function words
            function_word_frequencies = np.array([words_only.count(function_word) for function_word in function_words['Word']])
            # add the number of counted function words to the total number of function words for that author
            function_word_frequencies_total += function_word_frequencies
        # add 1 to each function_word_frequency to avoid zeros-values
        dataframe[author] = function_word_frequencies_total.astype(int) +1 
    return dataframe


def calc_pos_tag_frequencies(groups):
    dataframe = pd.DataFrame({'POS tag': possibleTags})

    for (author, df) in groups:
        pos_tag_frequencies_total = np.zeros(len(possibleTags))
        for text in df['Text']:
            # split the text into tokens
            word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
            words_only = word_tokenizer.tokenize(text.lower())
            tagged_words = pos_tag(words_only)
            # count the tag frequencies for the given text
            pos_tag_frequencies = np.array([tagged_words[1].count(tag) for tag in possibleTags])
            # add the tag frequencies for the given text to the total tag frequencies
            pos_tag_frequencies_total += pos_tag_frequencies
        #store tag frequencies in a dataframe and add 1 to every instance (to avoid zero values)
        dataframe[author] = pos_tag_frequencies_total.astype(int) + 1
    return dataframe

# Helper function for the Kullback Leibner Divergence computation: All frequencies have to be normalized (such that
# they sum up to 1) in order to calculate the divergence
def normalize_frequencies(to_be_normalized):
    for column in to_be_normalized.columns[1:]:
        sum_column = sum(to_be_normalized[column])
        normalized_values = [value / sum_column for value in to_be_normalized[column]]
        to_be_normalized[column] = normalized_values
    return to_be_normalized

# Given two vectors of normalized feature-vectors, this function calculates the kullback
# leibner divergence between two authors based on those feature-vectors
def calc_kullback_leibner_divergence(normalized_frequencies_author1, normalized_frequencies_author2):
    return entropy(normalized_frequencies_author1, normalized_frequencies_author2)

# The similarity is calculated, based on the kullback leibner divergence for two authors and a parameter omega
# (omega is by default 0.5)
def calc_similarity(k_l_divergence, omega=0.5):
    return exp(-k_l_divergence/omega)

#Given the normalized feature-vectors for the kag
def create_similarity_table(author_function_words, influencers_function_words):
    similarity_matrix = []
    # loop through all authors
    for author_column in author_function_words.columns[1:]:
        similarity_row = []
        # loop through each influencer to calculate the similarity (based on the kullback leibner divergence)
        # between each author and each influencer
        for influencer_column in influencers_function_words.columns[1:]:
            kl_divergence = calc_kullback_leibner_divergence(author_function_words[author_column], influencers_function_words[influencer_column] )
            similarity_row.append(calc_similarity(kl_divergence))
        # put the similarities between each author and each influencer in a matrx
        similarity_matrix.append(similarity_row)
    # convert the similarity matrix into a dataframe (where the columns refer to the influencers and the rows
    # to the kaggle authors)
    similarity_dataframe = pd.DataFrame(similarity_matrix, columns=influencers_function_words.columns[1:])
    similarity_dataframe['kaggle authors'] = author_function_words.columns[1:]
    similarity_dataframe.set_index('kaggle authors')
    return similarity_dataframe


################# Loading list of function words #################
print('Start loading data...')
path = 'Function Words.xlsx'
# function words are extracted from excel file
function_words = pd.read_excel(path, usecols = [0])

################## Data from kaggle authors is read ##################
print("Data of the kaggle authors is stored into a dataframe")
kaggle_authors = read_author_data()

################## Data from influencers is read and put into a dataframe ##################
# These are the stored influencers obtained utlizing google:
influencer_dict = {"Mary Shelley": ["Samuel Taylor Coleridge","John Milton","Mary Wollstonecraft","William Godwin"], "Edgar Allan Poe": ["John Keats","Lord Byron","Samuel Taylor Coleridge","Thomas de Quincey"], "H. P. Lovecraft": ["Friedrich Nietzsche", "H G Wells", "Robert E Howard"]}
# This is the directory of the influencer txt files:
influencer_directory = './Influencer Texts/'
# Next, read in all influencer data and create one big pd dataframe to store it:
influencer_texts = create_influencer_dataframe(influencer_directory, influencer_dict)


'''
################## Normalized function word frequencies are calculated ##################
print("Calculating normalized function word frequencies kaggle authors")
normalized_frequencies_kaggle_authors = normalize_frequencies(calc_function_word_frequencies(kaggle_authors.groupby('Author')))

print("Calculating normalized function word frequencies for influencers")
normalized_frequencies_influencers = normalize_frequencies(calc_function_word_frequencies(influencer_texts.groupby('Author')))

################## Calculate similarities between every kaggle author and every influencer ##################
print("Similarity table based on function words is created")
similarity_table = create_similarity_table(normalized_frequencies_kaggle_authors, normalized_frequencies_influencers)

################## Similarity Table is stored into an Excel file ##################
writer = pd.ExcelWriter('similarities.xlsx')
similarity_table.to_excel(writer,'Sheet1')
writer.save()
'''

################## Normalized POS-tag frequencies are calculated ##################
possibleTags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'RB',
                'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

print("Calculate normalized pos-tag frequencies for kaggle authors")
normalized_pos_tag_frequencies_kaggle = normalize_frequencies(calc_pos_tag_frequencies(kaggle_authors.groupby('Author')))

print("Calculate normalized pos-tag frequencies for influencer authors")
normalized_pos_tag_frequencies_influencer = normalize_frequencies(calc_pos_tag_frequencies(influencer_texts.groupby('Author')))

################## Calculate similarities between every kaggle author and every influencer ##################
print("Similarity table based on pos-tag frequencies is created")
similarity_table = create_similarity_table(normalized_pos_tag_frequencies_kaggle, normalized_pos_tag_frequencies_influencer)

################## Similarity Table is stored into an Excel file ##################
writer = pd.ExcelWriter('similarities_pos.xlsx')
similarity_table.to_excel(writer,'Sheet1')
writer.save()
