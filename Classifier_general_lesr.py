import Classifier_SVM
import pandas as pd
import re
import nltk
import numpy as np
import statistics
import sklearn.model_selection
from nltk import wordpunct_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from textblob import TextBlob
from gensim.models.wrappers import FastText
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, recall_score, precision_score

'''
This is the 'general Classifier' class. In this class the features are extracted and the classifier can be trained and tested. 
'''

# This function reads the training data from a csv file and puts it into a panda dataframe
def read_csv():
	data = pd.read_csv('train.csv')
	return data

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

# Split a twitter .csv into training and test data
def create_train_test_split(test_split, tweets):
	# Step one: Randomly re-order tweets (and reindex)
	tweets_reordered = tweets.sample(frac=1).reset_index(drop=True)
	# Store the test and training tweets
	tweets_test = tweets_reordered[0:int(len(tweets_reordered)*test_split)]
	tweets_train = tweets_reordered[int(len(tweets_reordered)*test_split):]
	# Save both datasets
	return tweets_test,tweets_train

# Given the folds for 10-fold cross validation and a classifier the user wants to use, this function will
# evaluate the performance of the classifier over all folds and return the average performance of that
# classifier
def classify(folds, classifier):
	scores = []
	# Loop through the training data, the training labels, the validation data and validation labels
	# of each fold
	for (train_data, train_labels, validation_data, validation_labels) in folds:

		# The feature matrices for both the training data and the validation data are generated
		feature_matrix_train = list(map(extract_features, train_data))
		feature_matrix_validation = list(map(extract_features, validation_data))

		# Feature selection could be applied here: Only the features with a higher variance than 0.3 are kept for training
		#sel = VarianceThreshold(threshold=0.3)
		#sel = sel.fit(feature_matrix_train)
		#feature_matrix_train = sel.transform(feature_matrix_train)
		#feature_matrix_validation = sel.transform(feature_matrix_validation)

		# Oversampling could be used to balance out dataset
		#feature_matrix_train, train_labels = SMOTE().fit_sample(feature_matrix_train, train_labels)

		# The classifier is fitted on the training
		validation_prediction = classifier.fit_and_predict(feature_matrix_train, train_labels, feature_matrix_validation)

		# The confusion matrix for the validation data predictions is printed
		print(confusion_matrix(validation_labels, validation_prediction, labels=['MWS', 'HPL', 'EAP']))

		# The function 'evaluate' is called to obtain the precision, the recall and the F1-score for
		# the classifier's performance on this specific fold
		score = evaluate(validation_labels, validation_prediction)
		# The performance measures are added to an array such that later on the average scores over all folds
		# can be computed
		scores.append(score)

	# The average scores over all folds are computed and returned
	average_performance = average_scores(scores)
	return average_performance


def performClassification(data,leaveOuts,classifier):
	# First, we create a train test split that will be used throughout the analysis
	test_set,train_set = create_train_test_split(0.25, data)
	# Reset indices:
	train_set = train_set.sample(frac=1).reset_index(drop=True)	
	test_set = test_set.sample(frac=1).reset_index(drop=True)	
    indicesDict =  {"lexicalsyntactic":list(range(0,73)), "sentiments":list(range(73,75)),"fasttext":list(range(75,375)), "None":[]}
	#f1scores = []    
	scores_per_leaveout = []

	print("Extracting features")
	# Extract the features
	features_trainTw =  [extract_features(text,"None") for text in train_set['Text'].tolist()]   
	features_testTw =  [extract_features(text,"None") for text in test_set['Text'].tolist()] 

	for i in range(0,len(leaveOuts)):
		leaveOut = leaveOuts[i]
		print('Currently leaving out: '+leaveOut) 
		# Creating corresponding feature vector (dropping all positions we do not include)
		new_features_train = np.asarray([[i for j, i in enumerate(featurevector) if j not in indicesDict[leaveOut]] for featurevector in features_trainTw])
		new_features_test = np.asarray([[i for j, i in enumerate(featurevector) if j not in indicesDict[leaveOut]] for featurevector in features_testTw])
		print("Starting Classification ")
		# Classify and evaluate
		skf = sklearn.model_selection.StratifiedKFold(n_splits=10)
		scores = []
		for fold_id, (train_indexes, validation_indexes) in enumerate(skf.split(train_set['Text'].tolist(), train_set['Author'].tolist())):
			# Print the fold number
			print("Fold %d" % (fold_id + 1))
			# Collect the data for this train/validation split
			train_features = np.asarray([new_features_train[x] for x in train_indexes])
			train_labels = np.asarray([train_set['Author'][x] for x in train_indexes])
			validation_features = np.asarray([new_features_train[x] for x in validation_indexes])
			validation_labels = np.asarray([train_set['Author'][x] for x in validation_indexes])

			# The classifier is fitted on the training
			validation_prediction = classifier.fit_and_predict(train_features, train_labels, validation_features)

			# The confusion matrix for the validation data predictions is printed
			print(confusion_matrix(validation_labels, validation_prediction, labels=['MWS', 'HPL', 'EAP']))

			# The function 'evaluate' is called to obtain the precision, the recall and the F1-score for
			# the classifier's performance on this specific fold
			score = evaluate(validation_labels, validation_prediction)
			# The performance measures are added to an array such that later on the average scores over all folds
			# can be computed
			scores.append(score)            
				
		# The average scores over all folds are computed and returned
		average_performance = average_scores(scores)
	return "FOTZE"            
			
			
			
			





# Given the predicted class labels and the true class labels, this function prints out the recall, precision and f1-score
# of a classifier and also returns these values
def evaluate(y_true, y_pred):
	recall = recall_score(y_true, y_pred, average='macro')
	print("Recall: %f" % recall)

	precision = precision_score(y_true, y_pred, average='macro')
	print("Precision: %f" % precision)

	f1_score = 2 * (precision * recall)/(precision + recall)
	print("F1-score: %f" % f1_score)

	return recall, precision, f1_score

# Given a list of recall, precision and f1-score measurement, the average values for this measures are
# calculated and printed.
def average_scores(scores):
	average_recall = (sum(r for r, p, f in scores))/len(scores)
	average_prediction = (sum (p for r, p, f in scores))/len(scores)
	average_f1 = (sum(f for r, p, f in scores))/len(scores)

	print("Average Recall: %f" % average_recall)
	print("Average Prediction: %f" % average_prediction)
	print("Average F1-score: %f" % average_f1)

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
# 
def extract_features(text,leaveOut):
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
	features = [] # Initialize the empty feature vector
	if leaveOut!= 'lexicalsyntactic':
			print("Beginning of lexicalsyntactic ", len(features))
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
			features.append(sum(word_lengths) / len(word_lengths))
			# Median word length
			features.append(statistics.median(word_lengths))
			# Standarddeviation in word length
			features.append(statistics.stdev(word_lengths))
			# number of words with word length 1 up to word length 10
			for i in range(1, 10):
				features.append(len([word for word in words_only if len(word) == i]))
			# Number of unique words_only, words_only occuring twice and words_only occuring thrice in a text
			for i in range(1, 4):
				features.append(calcNumberOfRepetitions(words_only, text, i))
			# Number of words without vowels
			features.append(len([word for word in words_only if not any(vowel in word for vowel in 'aoeui')]))
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
			## Number of sentences
			#features.append(len(sentences))
	
	if leaveOut!= 'sentiments':
			print("Beginning of sentiments ", len(features))
			# Sentiment
			blob = TextBlob(text)
			features.append(blob.sentiment[0])
			# Polarity
			features.append(blob.sentiment[1])

		
	if leaveOut!= 'fasttext': 
			print("Beginning of fasttext ", len(features))   
			fasttext_sentencevector = np.zeros(300, )
			sum_tfidf_values = 0
			# Digits are not included in the model:
			#print(text)
			for word in re.sub('[^A-Za-z\s]+', '', text).lower().split():#text.lower().split():  # TODO: think about preprocessing
				#print(word)
			
				# Extract the tf-idf value of this word in the given tweet:
				tfidf_value = sklearn_representation[data["Text"].tolist().index(text), sklearn_tfidf.vocabulary_[word]]
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
			# Finally, compute the weighted average vector of the sample:
			fasttext_sentencevector = fasttext_sentencevector / sum_tfidf_values
			for fasttext_value in fasttext_sentencevector:
				features.append(fasttext_value)

	return features

##############################     MAIN STUFF    ##############################

################# LOAD / PREPARE RESOURCES
#print("Loading resources ...")
#data = read_csv()
## If you don't have them already, please uncomment ( -> commented to save time)
##nltk.download('stopwords')
##nltk.download('punkt')
##nltk.download('averaged_perceptron_tagger')
#stop_words = set(stopwords.words('english')) # The stopwords we will need
#possibleTags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS',
#				'RB','RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB'] # possible POS tags
#vowels = ['a', 'o', 'u', 'i', 'e'] # vowels (really need to save them?)
## All Options for the leave one out analysis:
## 'bagOfWords','numberOfWords','specialCharacterCounts','wordLengths',functionWordCounts','vocabularyRichness','letterNgrams'
##leaveOuts = ['lexicalsyntactic','fasttext','sentiments','None']
#leaveOuts = ['None'] # debug purpose
##print("Perform analysis on whole dataset")
##predictions,recalls,precisions,f1_scores,binary_cross_entropys,y_pred = performClassification(train_tweets,test_tweets,leaveOuts,True)
#
## Next, load the fasttext model
#print("Loading pretrained fasttext model")
#fasttext_model = FastText.load_fasttext_format('wiki.simple')
#print("Finished loading fasttext model")
#print("Finished loading resources")
## Compute tf-idf values
#print("Preparing resources ...")
#tokenize = lambda doc: doc.lower().split(" ")
#sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
#all_documents = [re.sub('[^A-Za-z\s]+', '', text).lower() for text in data["Text"].tolist()]
#sklearn_representation = sklearn_tfidf.fit_transform(all_documents)    
#print("Finished preparing resources")


print("Staring classification")
#leaveOut = "lexicalsyntactic"
# We need this dict to easier access the features we want to include for each leave-out run
#indicesDict =  {"lexicalsyntactic":list(range(0,73)), "sentiments":list(range(73,75)),"fasttext":list(range(75,375)), "None":[]}
#currentfeatures = [[i for j, i in enumerate(featurevector) if j not in indicesDict[leaveOut]] for featurevector in features_trainTw]  


#classifier = SVM
## First, we create a train test split that will be used throughout the analysis
#test_set,train_set = create_train_test_split(0.25, data)
## Reset indices:
#train_set = train_set.sample(frac=1).reset_index(drop=True)	
#test_set = test_set.sample(frac=1).reset_index(drop=True)	
#
##f1scores = []    
#scores_per_leaveout = []
#for i in range(0,len(leaveOuts)):
#	leaveOut = leaveOuts[i]
#	print('Currently leaving out: '+leaveOut) 
#	print("Extracting features")
#	# Extract the features
#	features_trainTw =  [extract_features(text,leaveOut) for text in train_set['Text'].tolist()]   
#	features_testTw =  [extract_features(text,leaveOut) for text in test_set['Text'].tolist()]   
#
#	new_features_train = np.asarray(features_trainTw)
#	new_features_test = np.asarray(features_testTw)
#	print("Starting Classification ")
#	# Classify and evaluate
#	skf = sklearn.model_selection.StratifiedKFold(n_splits=10)
#	scores = []
#	for fold_id, (train_indexes, validation_indexes) in enumerate(skf.split(train_set['Text'].tolist(), train_set['Author'].tolist())):
#		# Print the fold number
#		print("Fold %d" % (fold_id + 1))
#		# Collect the data for this train/validation split
#		train_features = np.asarray([new_features_train[x] for x in train_indexes])
#		train_labels = np.asarray([train_set['Author'][x] for x in train_indexes])
#		validation_features = np.asarray([new_features_train[x] for x in validation_indexes])
#		validation_labels = np.asarray([train_set['Author'][x] for x in validation_indexes])
#
#		# The classifier is fitted on the training
#		validation_prediction = classifier.fit_and_predict(train_features, train_labels, validation_features)
#
#		# The confusion matrix for the validation data predictions is printed
#		print(confusion_matrix(validation_labels, validation_prediction, labels=['MWS', 'HPL', 'EAP']))
#
#		# The function 'evaluate' is called to obtain the precision, the recall and the F1-score for
#		# the classifier's performance on this specific fold
#		score = evaluate(validation_labels, validation_prediction)
#		# The performance measures are added to an array such that later on the average scores over all folds
#		# can be computed
#		scores.append(score)            
#			
#	# The average scores over all folds are computed and returned
#	average_performance = average_scores(scores)
#scores_per_leaveout.append(average_performance)  


#folds = create_folds(data.Text, data.Author)
#classify(folds, Classifier_SVM)
performClassification(data,leaveOuts,Classifier_SVM)
#file = open("Coleridge1.txt", "r") 
#text = data["Text"][2]
#feature_vector2 = extract_features(text, "None")
print("Finished classification")

















## In the main function the data is read from the csv file, the folds for 10-fold cross validation are
## created and finally the classifier is trained and validated on all of these folds
#def main():
#    data = read_csv()
#    #folds = create_folds(data.text, data.author)
#    #classify(folds, Classifier_SVM)
#
#
#
#
#if __name__ == '__main__':
#    main()