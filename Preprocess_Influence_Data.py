import collections
import random
import os
import pandas as pd


def put_in_dataframe(f_name, dir_name):

def read_directory(root_dir):
    for dir_name, sub_dir_list, file_list in os.walk(root_dir):
        print('Found directory: %s' % dir_name)
        for f_name in file_list:
            put_in_dataframe()
            print(f_name)





read_directory('./Mary Shelley Influencers Data')