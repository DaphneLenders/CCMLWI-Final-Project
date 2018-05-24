import collections
import random
import os
import pandas as pd


def extract_test(path_name):
    text = ''
    for line in open(path_name):
        if 'End of the Project Gutenberg EBook' in line:
            break
        else:
            text = text + line
    return text


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





mary_shelley_influencers = read_directory('./Mary Shelley Influencers Data', "Mary Shelley")
#print(mary_shelley_influencers.iloc[0]['Text'])