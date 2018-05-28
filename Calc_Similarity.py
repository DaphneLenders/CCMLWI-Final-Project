import pandas as pd
import os
import nltk
import numpy as np
from scipy.stats import entropy
from math import exp

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



def calc_function_word_frequencies(groups):
    dataframe = pd.DataFrame({'function word': function_words['Word']})

    for (author, df) in groups:
        function_word_frequencies_total = np.zeros(len(function_words['Word']))
        for text in df['Text']:
            word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
            words_only = word_tokenizer.tokenize(text.lower())
            function_word_frequencies = np.array([words_only.count(function_word) for function_word in function_words['Word']])
            function_word_frequencies_total += function_word_frequencies

        dataframe[author] = function_word_frequencies_total.astype(int) +1 
    return dataframe

# Helper function for the Kullback Leibner Divergence computation
def normalize_function_word_frequencies(dataframe):
    for column in dataframe.columns[1:]:
        sum_column = sum(dataframe[column])
        normalized_values = [value/sum_column for value in dataframe[column]]
        dataframe[column] = normalized_values
    return dataframe


def calc_kullback_leibner_divergence(text_dataframe ,author1, author2):
    normalized_frequencies = normalize_function_word_frequencies(calc_function_word_frequencies(text_dataframe.groupby('Author')))
    return entropy(normalized_frequencies[author1], normalized_frequencies[author2])


# The similarity is calculated, based on the kullback leibner divergence for two authors and a parameter omega
# (omega is by default 0.5)
def calc_similarity(k_l_divergence, omega=0.5):
    return exp(-k_l_divergence/omega)


#################                 LOADING DATA                #################
print('Start loading data...')
path = 'Function Words.xlsx'
# function words are extracted from excel file
function_words = pd.read_excel(path, usecols = [0])

# These are the stored influencers obtained utlizing google:
influencer_dict = {"Mary Shelly": ["Samuel Taylor Coleridge","John Milton","Mary Wollstonecraft","William Godwin"], "Edgar Alan Poe": ["John Keats","Lord Byron","Samuel Taylor Coleridge","Thomas de Quincey"]}
# This is the directory of the influencer txt files:
influencer_directory = './Influencer Texts/' 
# Next, read in all influencer data and create one big pd dataframe to store it:
influencer_texts = create_influencer_dataframe(influencer_directory,influencer_dict)
spooky_texts = pd.read_csv('train.csv') 
print('Loading successful.')
##################                   NEXT STEP                 #################
print('Start calculating divergence')
print(calc_kullback_leibner_divergence(influencer_texts,'William Godwin', 'Mary Wollstonecraft'))
print(calc_similarity(calc_kullback_leibner_divergence(influencer_texts,'William Godwin', 'Mary Wollstonecraft')))
print('Finished calculating divergence')

