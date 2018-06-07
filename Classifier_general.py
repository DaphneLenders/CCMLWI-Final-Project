import Classifier_SVM
import pandas as pd
from ExtractFeatures import extract_features
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, recall_score, precision_score

'''
This is the 'general Classifier' class. In this class the classifier can be trained and tested. The features
the classifier are trained on can be found in the 'ExtractFeatures' class.
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



# In the main function the data is read from the csv file, the folds for 10-fold cross validation are
# created and finally the classifier is trained and validated on all of these folds
def main():
    data = read_csv()
    folds = create_folds(data.text, data.author)
    classify(folds, Classifier_SVM)




if __name__ == '__main__':
    main()