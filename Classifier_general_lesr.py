import Classifier_NaiveBayes
import Classifier_XGBoost
import pandas as pd
import os
import re
import nltk
import numpy as np
import statistics
import time
import sklearn.model_selection
from nltk import wordpunct_tokenize, pos_tag, data
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from textblob import TextBlob
from gensim.models.wrappers import FastText
from sklearn import cross_validation,preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, recall_score, precision_score


'''
This is the 'general Classifier' class. In this class the features are extracted and the classifier can be trained and tested. 
'''

# This function reads the training data from a csv file and puts it into a panda dataframe
def read_csv():
    data = pd.read_csv('train.csv')
    return data

def read_char_2grams_csv():
    char_2grams = pd.read_csv('ngrams2.csv',keep_default_na=False)
    return char_2grams['2-gram'].apply(lambda x: x.lower())[0:30]

def read_char_3grams_csv():
    char_3grams = pd.read_csv('ngrams3.csv',keep_default_na=False)
    return char_3grams['3-gram'].apply(lambda x: x.lower())[0:30]

# This function creates a dataframe that contains the column names 'Text' and 'Author'
# by reading in the text stored at path_name and filling the column 'Text' with all
# individual sentences in that file. The column 'Author' will be filled with the
# author name specified in the input value author.
def convert_gutenberg_text_to_author_dataframe(path_name,author):
    # First, read and preprocess the text stored in the file:
    text = ''
    for line in open(path_name,encoding = "ISO-8859-1"):
        #print(line)
        if 'End of the Project Gutenberg EBook' in line:
            #print(line)
            break
        else:
            text = text + line  
    text = " ".join(text.replace("\n"," ").split())
    # Now, create the corresponding dataframe: 
    df = pd.DataFrame(tokenizer.tokenize(text))   
    df.columns = ['Text']
    df['Author'] = author
    return df

# This function will ensure that for a given author in the input dataframe data_authors,
# there will be amount_samples_per_author text samples for the author specified 
# in the input variable author_name. 
# This denotes deleting samples if there are more than amount_samples_per_author samples 
# in already.
# If there are less, however, the samples will be augmented by sentences extracted from
# the file specified in filename. 
def get_exactly_n_samples_for_given_author(data_authors, author_name, amount_samples_per_author,filename):
    # Case 1: We have too many samples in -> randomly delete those that are too much
    if(data_authors[data_authors['Author']==author_name].shape[0]>amount_samples_per_author):
        print("Too many samples for author", author_name)
        # Calculate how many samples we have to delete 
        amount_samples_to_be_deleted = data_authors[data_authors['Author']==author_name].shape[0] - amount_samples_per_author  
        # Randomly select which indices have to be delted
        indices_for_deletion = np.random.choice(data_authors[data_authors['Author']==author_name].index, amount_samples_to_be_deleted, replace=False)
        data_authors_n = data_authors.drop(indices_for_deletion)  
    # Case 2: We have to augment the data
    elif(data_authors[data_authors['Author']==author_name].shape[0]<amount_samples_per_author):
#        print("Not enough samples for author", author_name)
        # Read in the additional data to a dataframe
        augmenting_data = convert_gutenberg_text_to_author_dataframe(filename,author_name)    
        # Calculate how many samples have to be added
        amount_samples_to_be_added = amount_samples_per_author - data_authors[data_authors['Author']==author_name].shape[0]
        # Remove sentences that are already in the authors_data dataframe!
        # For debug purposes: uncomment next line to see which sentences occur in both dataframes
        excluded = augmenting_data[augmenting_data['Text'].isin(data_authors.loc[data_authors['Author'] == author_name]['Text'])]  
        # Remove sentences that already occur in our dataframe from the additional data
        augmenting_data = augmenting_data[~augmenting_data['Text'].isin(data_authors.loc[data_authors['Author'] == author_name]['Text'])]
        augmenting_data = augmenting_data.loc[~augmenting_data['Text'].str.contains('^[_\W]+$')] # remove texts that consist of special chars only from what can be chosen.
        augmenting_data = augmenting_data.loc[~augmenting_data['Text'].str.contains('^[^A-Za-z]*$')] # remove texts that have no letter inside
        #augmenting_data.loc[augmenting_data['Text']!='.'] # remove texts that consist of '.' only from what can be chosen.
        
        # Now, we can randomly add samples 
        # Caution: We cannot sample more than the actual number of available additional sentence
        data_authors_n = pd.DataFrame()
        if(amount_samples_to_be_added>augmenting_data.shape[0]):
            amount_samples_to_be_added = augmenting_data.shape[0]
            print("   Not enough samples available to add, adding ",amount_samples_to_be_added," samples")
            # Now, augment our dataframe:
            data_authors_n = pd.concat([data_authors,augmenting_data]).reset_index(drop=True)
        else:   
            # Now, augment our dataframe:
            data_authors_n = pd.concat([data_authors,augmenting_data.sample(amount_samples_to_be_added)]).reset_index(drop=True)  
#        print("   Now, the amount of text samples for the author ",author_name, "is: ", data_authors_n[data_authors_n['Author']==author_name].shape[0]) 
    # Case 3: We have exactly the right amount in the dataframe already
    else: 
        data_authors_n = data_authors
    return data_authors_n   

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
                text = extract_gutenberg_text_from_path(path_name)
                list.append([text, author])#, influences])
    
    dataframe = pd.DataFrame(list, columns=['Text', 'Author'])
    return dataframe

# In this function the folds for 10-fold cross validation are made. Each fold consists of the data used for
# training, the training labels, the data of the validation set and the labels of the validation set.
def create_folds(train_data, labels):
    splitted_indeces = cross_validation.KFold(len(labels), n_folds=10, shuffle=True, random_state=1)
    folds = []
    for train_index, validation_index in splitted_indeces:
        train_data_new, validation_data_new = train_data.reindex(train_index), train_data.reindex(validation_index)
        train_labels_new, validation_labels_new = labels.reindex(train_index), labels.reindex(validation_index)
        folds.append((train_data_new, train_labels_new, validation_data_new, validation_labels_new))
    return folds

# Split a dataframe into training and test data
def create_train_test_split(test_split, texts):
    # Step one: Randomly re-order tweets (and reindex)
    texts_reordered = texts.sample(frac=1).reset_index(drop=True)
    # Store the test and training tweets
    texts_test = texts_reordered[0:int(len(texts_reordered)*test_split)]
    texts_train = texts_reordered[int(len(texts_reordered)*test_split):]
    # Save both datasets
    return texts_test,texts_train

## Given the folds for 10-fold cross validation and a classifier the user wants to use, this function will
## evaluate the performance of the classifier over all folds and return the average performance of that
## classifier
#def classify(folds, classifier):
#    scores = []
#    # Loop through the training data, the training labels, the validation data and validation labels
#    # of each fold
#    for (train_data, train_labels, validation_data, validation_labels) in folds:
#
#        # The feature matrices for both the training data and the validation data are generated
#        feature_matrix_train = list(map(extract_features, train_data))
#        feature_matrix_validation = list(map(extract_features, validation_data))
#
#        # Feature selection could be applied here: Only the features with a higher variance than 0.3 are kept for training
#        #sel = VarianceThreshold(threshold=0.3)
#        #sel = sel.fit(feature_matrix_train)
#        #feature_matrix_train = sel.transform(feature_matrix_train)
#        #feature_matrix_validation = sel.transform(feature_matrix_validation)
#
#        # Oversampling could be used to balance out dataset
#        #feature_matrix_train, train_labels = SMOTE().fit_sample(feature_matrix_train, train_labels)
#
#        # The classifier is fitted on the training
#        validation_prediction = classifier.fit_and_predict(feature_matrix_train, train_labels, feature_matrix_validation)
#
#        # The confusion matrix for the validation data predictions is printed
#        print(confusion_matrix(validation_labels, validation_prediction, labels=['MWS', 'HPL', 'EAP']))
#
#        # The function 'evaluate' is called to obtain the precision, the recall and the F1-score for
#        # the classifier's performance on this specific fold
#        score = evaluate(validation_labels, validation_prediction)
#        # The performance measures are added to an array such that later on the average scores over all folds
#        # can be computed
#        scores.append(score)
#    # The average scores over all folds are computed and returned
#    average_performance = average_scores(scores)
#    return average_performance


def performClassification(data,train_set,test_set,leaveOuts,classifier,baseline_classifier,results_log_classifer,results_log_baseline):
    # Reset indices:
    train_set = train_set.sample(frac=1).reset_index(drop=True) 
    test_set = test_set.sample(frac=1).reset_index(drop=True)   
    indicesDict =  {"lexicaldiversity":list(range(0,20)),"specialchars":list(range(20,38)),"bi- and trigrams":list(range(38,78)),"postags":list(range(78,110)),"sentiments":list(range(110,112)),"fasttext":list(range(112,412)), "None":[]}
    #f1scores = []    

    print("   Extracting features")
    t = time.time()
    # Extract the features
    features_trainTw =  [extract_features(text,"None",data) for text in train_set['Text'].tolist()]   
    features_testTw =  [extract_features(text,"None",data) for text in test_set['Text'].tolist()] 
    print("   Elapsed time:", (time.time() - t))
    t = time.time()
    for i in range(0,len(leaveOuts)):
        leaveOut = leaveOuts[i]
        print('      Currently leaving out: '+leaveOut,file=results_log_classifer) 
        print('      Currently leaving out: '+leaveOut,file=results_log_baseline) 
        print('      Currently leaving out: '+leaveOut) 
        # Creating corresponding feature vector (dropping all positions we do not include)
        new_features_train = preprocessing.scale(np.asarray([[i for j, i in enumerate(featurevector) if j not in indicesDict[leaveOut]] for featurevector in features_trainTw]))
        new_features_test = preprocessing.scale(np.asarray([[i for j, i in enumerate(featurevector) if j not in indicesDict[leaveOut]] for featurevector in features_testTw]))
        print("Starting Classification ")
        # Classify and evaluate
        skf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        scores = []
        scores_baseline = []
#        ####################### Validation DATA ###############################
#        for fold_id, (train_indexes, validation_indexes) in enumerate(skf.split(train_set['Text'].tolist(), train_set['Author'].tolist())):
#            # Print the fold number
#            print("      Fold %d" % (fold_id + 1),file=results_log_classifer)
#            print("      Fold %d" % (fold_id + 1),file=results_log_baseline)
#            print("      Fold %d" % (fold_id + 1))
#            # Collect the data for this train/validation split
#            train_features = np.asarray([new_features_train[x] for x in train_indexes])
#            train_labels = np.asarray([train_set['Author'][x] for x in train_indexes])
#            validation_features = np.asarray([new_features_train[x] for x in validation_indexes])
#            validation_labels = np.asarray([train_set['Author'][x] for x in validation_indexes])
#
#            # The classifier is fitted on the training
#            print("      Running baseline classifier")
#            validation_prediction_baseline = baseline_classifier.fit_and_predict(train_features, train_labels, validation_features)
#            print("      Elapsed time:", (time.time() - t))
#            print("      Elapsed time:", (time.time() - t),file=results_log_baseline)
#            t = time.time()
#            print("      Running classifier")
#            validation_prediction = classifier.fit_and_predict(train_features, train_labels, validation_features)
#            print("      Elapsed time:", (time.time() - t))
#            print("      Elapsed time:", (time.time() - t),file=results_log_classifier)
#            t = time.time()            
#            ##################### THE BASELINE CLASSIFIER #####################
#            # The confusion matrix for the validation data predictions is printed
#            print("      Baseline: ")
#            print("      Baseline: ",file=results_log_baseline)
#            print(confusion_matrix(validation_labels, validation_prediction_baseline, labels=['MWS', 'HPL', 'EAP']))
#            print(confusion_matrix(validation_labels, validation_prediction_baseline, labels=['MWS', 'HPL', 'EAP']),file=results_log_baseline)
#            # The function 'evaluate' is called to obtain the precision, the recall and the F1-score for
#            # the classifier's performance on this specific fold
#            score_baseline = evaluate(validation_labels, validation_prediction_baseline,results_log_baseline)
#            # The performance measures are added to an array such that later on the average scores over all folds
#            # can be computed
#            scores_baseline.append(score_baseline)   
#            ##################### THE ACTUAL CLASSIFIER #######################
#            print("      Classifier: ")
#            print("      Classifier: ",file=results_log_classifer)
#            print(confusion_matrix(validation_labels, validation_prediction, labels=['MWS', 'HPL', 'EAP']))
#            print(confusion_matrix(validation_labels, validation_prediction, labels=['MWS', 'HPL', 'EAP']),file=results_log_classifer)
#            # The function 'evaluate' is called to obtain the precision, the recall and the F1-score for
#            # the classifier's performance on this specific fold
#            score = evaluate(validation_labels, validation_prediction,results_log_classifer)
#            # The performance measures are added to an array such that later on the average scores over all folds
#            # can be computed
#            scores.append(score)  
#        print("   Average baseline performance on validation set:")    
#        print("   Average baseline performance on validation set:" ,file=results_log_baseline) 
#        average_performance_baseline = average_scores(scores_baseline,results_log_baseline)
#        print("   Average classifier performance on validation set:")
#        print("   Average classifier performance on validation set:",file=results_log_classifer)
#        # The average scores over all folds are computed and returned
#        average_performance = average_scores(scores,results_log_classifer)
        ####################### TEST DATA #####################################
#        train_featuresAll = new_features_train
#        test_features = preprocessing.scale(np.asarray(features_testTw))
        train_labelsALL = np.asarray(train_set['Author'])
        test_labels = np.asarray(test_set['Author'])
        test_prediction_baseline = baseline_classifier.fit_and_predict(new_features_train, train_labelsALL, new_features_test)
        test_prediction = classifier.fit_and_predict(new_features_train, train_labelsALL, new_features_test)
        print("   Baseline performance on test data set:")
        print("   Baseline performance on test data set:",file=results_log_baseline)
        print(confusion_matrix(test_labels, test_prediction_baseline, labels=['MWS', 'HPL', 'EAP']))
        print(confusion_matrix(test_labels, test_prediction_baseline, labels=['MWS', 'HPL', 'EAP']),file=results_log_baseline)
        test_score_baseline = evaluate(test_labels, test_prediction_baseline,results_log_baseline)
        print("   Classifier performance on test data set:")
        print("   Classifier performance on test data set:",file=results_log_classifer)
        print(confusion_matrix(test_labels, test_prediction, labels=['MWS', 'HPL', 'EAP']))
        print(confusion_matrix(test_labels, test_prediction, labels=['MWS', 'HPL', 'EAP']),file=results_log_classifer)
        test_score = evaluate(test_labels, test_prediction,results_log_classifer)        
    return "finished"            

# Given the predicted class labels and the true class labels, this function prints out the recall, precision and f1-score
# of a classifier and also returns these values
def evaluate(y_true, y_pred,results_log):
    recall = recall_score(y_true, y_pred, average='macro')
    print("   Recall: %f" % recall,file=results_log)
    print("   Recall: %f" % recall)
    precision = precision_score(y_true, y_pred, average='macro')
    print("   Precision: %f" % precision,file=results_log)
    print("   Precision: %f" % precision)
    f1_score = 2 * (precision * recall)/(precision + recall)
    print("   F1-score: %f" % f1_score,file=results_log)
    print("   F1-score: %f" % f1_score)
    return recall, precision, f1_score

# Given a list of recall, precision and f1-score measurement, the average values for this measures are
# calculated and printed.
def average_scores(scores,results_log):
    average_recall = (sum(r for r, p, f in scores))/len(scores)
    average_prediction = (sum (p for r, p, f in scores))/len(scores)
    average_f1 = (sum(f for r, p, f in scores))/len(scores)
    print("Average Recall: %f" % average_recall)
    print("Average Prediction: %f" % average_prediction)
    print("Average F1-score: %f" % average_f1)
    print("Average Recall: %f" % average_recall,file=results_log)
    print("Average Prediction: %f" % average_prediction,file=results_log)
    print("Average F1-score: %f" % average_f1,file=results_log)    

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

def make_bigrams(words):
    length = len(words)
    bigrams = []
    for i in range(length - 1):
        if not (words[i].isdigit() or words[i + 1].isdigit()):
            bigrams.append((words[i], words[i + 1]))
    return bigrams

def make_stemmed_bigrams(words):
    length = len(words)
    bigrams = []
    for i in range(length - 1):
        if not (words[i].isdigit() or words[i + 1].isdigit()):
            bigrams.append((words[i], words[i + 1]))
    return bigrams

def make_trigrams(words):
    length = len(words)
    trigrams = []
    for i in range(length - 2):
        if not (words[i].isdigit() or words[i + 1].isdigit() or words[i + 2].isdigit()):
            trigrams.append((words[i], words[i + 1], words[i + 2]))
    return trigrams

# Given all n-grams in a song, and some queried n-gram this function calculates how often the queried n-gram occurs
# in the code
def calc_ngram_occurence(ngrams, queriedNgram):
    numberOfngramOccurences = 0
    for ngram in ngrams:
        if ngram == queriedNgram:
            numberOfngramOccurences += 1
    return numberOfngramOccurences

# A function to actually extract all needed features from a single given text sample.  
def extract_features(text,leaveOut,data_authors):
    text_lower = text.lower()
#    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#    word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    # bag_of_words_and_characters includes characters
    bag_of_words_and_characters = [x for x in wordpunct_tokenize(text)]
    # words_only doesn't include characters
    words_only = tokenizer.tokenize(text_lower)
#    # stemmedwords
#    stemmer = SnowballStemmer("english")
#    stemmed_words = [stemmer.stem(x) for x in words_only]
    # vocab doesn't contain double elements
    vocab = set(words_only)
#    sentences = sentence_tokenizer.tokenize(text)
#    words_per_sentence = np.array([len(tokenizer.tokenize(s))
#                                  for s in sentences])
    features = []  # Initialize the empty feature vector
    if leaveOut != 'lexicaldiversity':
#        print("Beginning of lexicaldiversity ", len(features))
        # Count the number of words_only
        features.append(len(bag_of_words_and_characters))
        # Count the number of words_only, excluded the stopwords
        features.append(len([x for x in bag_of_words_and_characters if x.lower() not in stop_words]))
        # Number of just the stopwords
        features.append(len([x for x in bag_of_words_and_characters if x.lower() in stop_words]))
#        # Average number of words_only per sentence
#        features.append(words_per_sentence.mean())
#        # Sentence length variation
#        features.append(words_per_sentence.std())
        # Lexical diversity
        features.append(len(vocab) / float(len(words_only)))
        # Average word length
        word_lengths = [len(word) for word in words_only]
        features.append(sum(word_lengths) / len(word_lengths))
        # Median word length
        features.append(statistics.median(word_lengths))
        # Standarddeviation in word length
        if(len(word_lengths)==1): # not enough data points inside!
          features.append(word_lengths[0])
        else:    
          features.append(statistics.stdev(word_lengths))
        # number of words with word length 1 up to word length 10
        for i in range(1, 10):
            features.append(len([word for word in words_only if len(word) == i]))
        # Number of unique words_only, words_only occuring twice and words_only occuring thrice in a text
        for i in range(1, 4):
            features.append(calcNumberOfRepetitions(words_only, text, i))
        # Number of words without vowels (to catch abbreviations)
        features.append(len([word for word in words_only if not any(vowel in word for vowel in 'aoeui')]))
    if leaveOut != 'specialchars':
#        print("Beginning of specialchars ", len(features))
        # Number of uppercase words_only not at the beginning of the sentence
        upper_list = re.findall(r'(?<!\.\s)\b[A-Z][a-z]*\b', text)
        features.append(len(upper_list))
        # Number of whitespace
        features.append(sum(c.isspace() for c in text))
        # Number of alphabetic chars
        #             features.append(sum(c.isalpha() for c in text))
        # Number of digits
        features.append(sum(c.isdigit() for c in text))
        # Number of special characters
        special_characters = ['!', ',', '.', ';', ':', '"', '-', '#', '&', '%', '|', '(', ')', '*', '@']
        for special_character in special_characters:
            count = text.count(special_character)
            features.append(count)
            ## Number of sentences
            # features.append(len(sentences))
    if leaveOut != 'biandtrigrams':
#        print("Beginning of bi- and trigrams ", len(features))
        #Character 2-gram frequencies
        for two_gram in char_2grams:
            #print(two_gram)
            features.append(text_lower.count(two_gram.lower()))
        #Character 3-gram frequencies
        for three_gram in char_3grams:
            features.append(text_lower.count(three_gram.lower()))
    if leaveOut != 'postags':
#        print("Beginning of postags ", len(features))
        # POS-tag frequencies
        tagged_words = pos_tag(words_only)
        for posTag in possibleTags:
            features.append(len([x for x in tagged_words if x[1] == posTag]))
    if leaveOut!= 'sentiments':
#            print("Beginning of sentiments ", len(features))
            # Sentiment
            blob = TextBlob(text)
            features.append(blob.sentiment[0])
            # Polarity
            features.append(blob.sentiment[1])    
    if leaveOut!= 'fasttext': 
#            print("Beginning of fasttext ", len(features))   
            fasttext_sentencevector = np.zeros(300, )
            sum_tfidf_values = 0
            # Digits are not included in the model:
            #print(text)
            for word in re.sub('[^A-Za-z\s]+', '', text).lower().split():#text.lower().split():  # TODO: think about preprocessing
#                print(word)      
#                print(text)
                # Extract the tf-idf value of this word in the given tweet:
                tfidf_value = sklearn_representation[data_authors["Text"].tolist().index(text), sklearn_tfidf.vocabulary_[word]]
                # Id the tfidf_value is 0, we change it into a small scalar:
                if (tfidf_value == 0):
                    tfidf_value = 0.001
                # For each word, add the corresponding fasttext vector:
                fasttext_sentencevector = fasttext_sentencevector + fasttext_model[word] * tfidf_value
                # Add up the tfidf values
                sum_tfidf_values += tfidf_value
                # print("sum_tfidf_values = "+str(sum_tfidf_values))
            if (sum_tfidf_values == 0):
                # This is an error that might occur, but it might not be there in our dataset
                print("ERROR IN TFIDF! SAMPLE TEXT IS :")
                print(text)
                #print(data_authors.loc[data_authors['Text'==text]])
            # Finally, compute the weighted average vector of the sample:
            fasttext_sentencevector = fasttext_sentencevector / sum_tfidf_values
            for fasttext_value in fasttext_sentencevector:
                features.append(fasttext_value)
    return features



###############################     MAIN STUFF    #############################
################ LOAD / PREPARE RESOURCES
print("Loading resources ...")
# Data stands for the actual amount of data for 
data_authors = read_csv().drop('id',axis=1) # we don't need the id values for our works
# If you don't have them already, please uncomment ( -> commented to save time)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english')) # The stopwords we will need
possibleTags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS',
               'RB','RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB'] # possible POS tags
vowels = ['a', 'o', 'u', 'i', 'e'] # vowels (really need to save them?)
leaveOuts = ['None'] 
char_2grams = read_char_2grams_csv()
char_3grams= read_char_3grams_csv()
# Next, load the fasttext model
print("Loading pretrained fasttext model")
fasttext_model = FastText.load_fasttext_format('wiki.simple')
print("Finished loading fasttext model")
print("Finished loading resources")

print("Balancing out dataset")
# We decided to set the number of samples per author to 6000
amount_samples_per_author = 6000  
# Equalize the amount of samples per author:
data_authors = get_exactly_n_samples_for_given_author(data_authors, 'HPL', amount_samples_per_author,"Influencer Texts/Lovecraft.txt")
data_authors = get_exactly_n_samples_for_given_author(data_authors, 'EAP', amount_samples_per_author,"Influencer Texts/Poe.txt")  
data_authors = get_exactly_n_samples_for_given_author(data_authors, 'MWS', amount_samples_per_author,"Influencer Texts/Shelley.txt")  
print("Now, the amount of text samples for the author EAP is: ", data_authors[data_authors['Author']=='EAP'].shape[0])  
print("Now, the amount of text samples for the author HPL is: ", data_authors[data_authors['Author']=='HPL'].shape[0])    
print("Now, the amount of text samples for the author MWS is: ", data_authors[data_authors['Author']=='MWS'].shape[0])   
print("Finished balacing out dataset")

################## PERFORM CLASSIFICATION
print("Looping over data ratio")
# First, we create a train test split that will be used throughout the analysis
test_set,train_set = create_train_test_split(0.2, data_authors)
amount_of_data_needed = [1,1.25,1.5,1.75,2]
# Do a loop until end of code:
for ratio in amount_of_data_needed:
    print("   Increasing the author data to ", ratio," times the amount of original samples for each author.")
    # We want to store all important outputs of the console in a corresponding txt file
    filename_classifier = "results_log_classifier_for_data_amount_" + str(ratio) + ".txt"
    results_log_classifier = open(filename_classifier,'w')
    filename_baseline = "results_log_baseline_for_data_amount_" + str(ratio) + ".txt"
    results_log_baseline = open(filename_baseline,'w')    
    # Next, we have to augment our data -> calculate amount of samples needed
    amount_samples_per_author_plus_influencer = int(4800*ratio)  
    print("   Now, the amount of text samples needed per author is: ", amount_samples_per_author_plus_influencer)
    # Augment the amount of samples per author:
    data_authors_plus_influencer = get_exactly_n_samples_for_given_author(train_set, 'HPL', amount_samples_per_author_plus_influencer,"Final Influencers/Lovecraft_influencer/Lovecraft_influencer.txt")
    data_authors_plus_influencer = get_exactly_n_samples_for_given_author(data_authors_plus_influencer, 'EAP', amount_samples_per_author_plus_influencer,"Final Influencers/Poe_influencer.txt")  
    data_authors_plus_influencer = get_exactly_n_samples_for_given_author(data_authors_plus_influencer, 'MWS', amount_samples_per_author_plus_influencer,"Final Influencers/MaryShelley_influencer/Mary Shelley_influencer.txt")  
    print("   Now, the amount of text samples for the author EAP is: ", data_authors_plus_influencer[data_authors_plus_influencer['Author']=='EAP'].shape[0],file=results_log_classifier)  
    print("   Now, the amount of text samples for the author HPL is: ", data_authors_plus_influencer[data_authors_plus_influencer['Author']=='HPL'].shape[0],file=results_log_classifier)    
    print("   Now, the amount of text samples for the author MWS is: ", data_authors_plus_influencer[data_authors_plus_influencer['Author']=='MWS'].shape[0],file=results_log_classifier) 
    print("   Now, the amount of text samples for the author EAP is: ", data_authors_plus_influencer[data_authors_plus_influencer['Author']=='EAP'].shape[0],file=results_log_baseline)  
    print("   Now, the amount of text samples for the author HPL is: ", data_authors_plus_influencer[data_authors_plus_influencer['Author']=='HPL'].shape[0],file=results_log_baseline)    
    print("   Now, the amount of text samples for the author MWS is: ", data_authors_plus_influencer[data_authors_plus_influencer['Author']=='MWS'].shape[0],file=results_log_baseline)     
    print("   Now, the amount of text samples for the author EAP is: ", data_authors_plus_influencer[data_authors_plus_influencer['Author']=='EAP'].shape[0])  
    print("   Now, the amount of text samples for the author HPL is: ", data_authors_plus_influencer[data_authors_plus_influencer['Author']=='HPL'].shape[0])    
    print("   Now, the amount of text samples for the author MWS is: ", data_authors_plus_influencer[data_authors_plus_influencer['Author']=='MWS'].shape[0])     
    # Compute corresponding tf-idf values
    print("    Preparing additional resources ...")
    new_train_set = data_authors_plus_influencer
    data_authors_plus_influencer = pd.concat([test_set,new_train_set]).reset_index(drop=True)
    tokenize = lambda doc: doc.lower().split(" ")
    sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
    all_documents = [re.sub('[^A-Za-z\s]+', '', text).lower() for text in data_authors_plus_influencer["Text"].tolist()]
    sklearn_representation = sklearn_tfidf.fit_transform(all_documents)    
    print("   Finished preparing additional resources")
    print("   Starting classification")
    # Now, we can perform the actual classification:
    performClassification(data_authors_plus_influencer,new_train_set,test_set,leaveOuts, Classifier_XGBoost, Classifier_NaiveBayes,results_log_classifier,results_log_baseline)  
    results_log_classifier.close()
    results_log_baseline.close()
    print("   Finished classification")
print("Finished looping over data ratio")

################# PERFORM LEAVE-ONE-OUT Analysis
# All Options for the leave one out analysis:
leaveOuts = ['lexicaldiversity','specialchars','bi- and trigrams','postags','fasttext','sentiments']
print("Looping over leave out options and ratios")
amount_of_data_needed = [1,1.25] # We decided to compare the option without influencer data to the best classified version with influencer data
# Do a loop until end of code:
for ratio in amount_of_data_needed:
    print("   Increasing the author data to ", ratio," times the amount of original samples for each author.")
    # We want to store all important outputs of the console in a corresponding txt file
    filename_classifier = "results_log_classifier_leaveOut_for_data_amount_" + str(ratio) + ".txt"
    results_log_classifier = open(filename_classifier,'w')
    filename_baseline = "results_log_baseline_leaveOut_for_data_amount_" + str(ratio) + ".txt"
    results_log_baseline = open(filename_baseline,'w')    
    # Next, we have to augment our data -> calculate amount of samples needed
    amount_samples_per_author_plus_influencer = int(4800*ratio)  
    print("   Now, the amount of text samples needed per author is: ", amount_samples_per_author_plus_influencer)
    # Augment the amount of samples per author:
    data_authors_plus_influencer = get_exactly_n_samples_for_given_author(train_set, 'HPL', amount_samples_per_author_plus_influencer,"Final Influencers/Lovecraft_influencer/Lovecraft_influencer.txt")
    data_authors_plus_influencer = get_exactly_n_samples_for_given_author(data_authors_plus_influencer, 'EAP', amount_samples_per_author_plus_influencer,"Final Influencers/Poe_influencer.txt")  
    data_authors_plus_influencer = get_exactly_n_samples_for_given_author(data_authors_plus_influencer, 'MWS', amount_samples_per_author_plus_influencer,"Final Influencers/MaryShelley_influencer/Mary Shelley_influencer.txt")  
    print("   Now, the amount of text samples for the author EAP is: ", data_authors_plus_influencer[data_authors_plus_influencer['Author']=='EAP'].shape[0],file=results_log_classifier)  
    print("   Now, the amount of text samples for the author HPL is: ", data_authors_plus_influencer[data_authors_plus_influencer['Author']=='HPL'].shape[0],file=results_log_classifier)    
    print("   Now, the amount of text samples for the author MWS is: ", data_authors_plus_influencer[data_authors_plus_influencer['Author']=='MWS'].shape[0],file=results_log_classifier) 
    print("   Now, the amount of text samples for the author EAP is: ", data_authors_plus_influencer[data_authors_plus_influencer['Author']=='EAP'].shape[0],file=results_log_baseline)  
    print("   Now, the amount of text samples for the author HPL is: ", data_authors_plus_influencer[data_authors_plus_influencer['Author']=='HPL'].shape[0],file=results_log_baseline)    
    print("   Now, the amount of text samples for the author MWS is: ", data_authors_plus_influencer[data_authors_plus_influencer['Author']=='MWS'].shape[0],file=results_log_baseline)     
    print("   Now, the amount of text samples for the author EAP is: ", data_authors_plus_influencer[data_authors_plus_influencer['Author']=='EAP'].shape[0])  
    print("   Now, the amount of text samples for the author HPL is: ", data_authors_plus_influencer[data_authors_plus_influencer['Author']=='HPL'].shape[0])    
    print("   Now, the amount of text samples for the author MWS is: ", data_authors_plus_influencer[data_authors_plus_influencer['Author']=='MWS'].shape[0])     
    # Compute corresponding tf-idf values
    print("    Preparing additional resources ...")
    new_train_set = data_authors_plus_influencer
    data_authors_plus_influencer = pd.concat([test_set,new_train_set]).reset_index(drop=True)
    tokenize = lambda doc: doc.lower().split(" ")
    sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
    all_documents = [re.sub('[^A-Za-z\s]+', '', text).lower() for text in data_authors_plus_influencer["Text"].tolist()]
    sklearn_representation = sklearn_tfidf.fit_transform(all_documents)    
    print("   Finished preparing additional resources")
    print("   Starting leave-one-out analysis")
    # Now, we can perform the actual classification:
    performClassification(data_authors_plus_influencer,new_train_set,test_set,leaveOuts, Classifier_XGBoost, Classifier_NaiveBayes,results_log_classifier,results_log_baseline)  
    results_log_classifier.close()
    results_log_baseline.close()
    print("   Finished leave-one-out analysis")
print("Finished looping over leave out options and ratios")



